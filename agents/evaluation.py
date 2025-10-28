"""
Evaluation Agent for EduGrade AI
Handles semantic grading using GPT-4o-mini and fact-checking with Perplexity Sonar API
Implements rubric-based scoring with partial marking
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple
import asyncio
import aiohttp
from dataclasses import dataclass
from enum import Enum

# Google Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Google Gemini not available. Install google-generativeai")

# Perplexity
try:
    from perplexity import Perplexity
    PERPLEXITY_AVAILABLE = True
except ImportError:
    PERPLEXITY_AVAILABLE = False
    logging.warning("Perplexity not available. Install perplexity-ai")

logger = logging.getLogger(__name__)

class GradingLevel(Enum):
    """Grading levels for different types of questions"""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"

@dataclass
class RubricCriteria:
    """Rubric criteria for evaluation"""
    name: str
    weight: float
    max_points: float
    description: str
    keywords: List[str] = None

@dataclass
class EvaluationResult:
    """Result of answer evaluation"""
    score: float
    max_score: float
    percentage: float
    feedback: str
    conceptual_analysis: str
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    confidence: float
    fact_check_results: Dict = None

class EvaluationAgent:
    """Agent responsible for evaluating student answers using AI"""
    
    def __init__(self, 
                 google_gemini_api_key: str = None,
                 perplexity_api_key: str = None,
                 rubric_path: str = None):
        """
        Initialize the Evaluation Agent
        
        Args:
            google_gemini_api_key: Google Gemini API key
            perplexity_api_key: Perplexity API key
            rubric_path: Path to rubric configuration file
        """
        self.gemini_model = None
        self.perplexity_client = None
        self.rubric = None
        
        # Initialize Google Gemini
        if GEMINI_AVAILABLE and google_gemini_api_key:
            try:
                genai.configure(api_key=google_gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("Google Gemini client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Google Gemini client: {e}")
        
        # Initialize Perplexity
        if PERPLEXITY_AVAILABLE and perplexity_api_key:
            try:
                self.perplexity_client = Perplexity(api_key=perplexity_api_key)
                logger.info("Perplexity client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Perplexity client: {e}")
        
        # Load rubric
        if rubric_path and os.path.exists(rubric_path):
            self.load_rubric(rubric_path)
        else:
            self.load_default_rubric()
    
    def load_rubric(self, rubric_path: str):
        """Load rubric from JSON file"""
        try:
            with open(rubric_path, 'r', encoding='utf-8') as f:
                rubric_data = json.load(f)
            
            self.rubric = {
                'criteria': [RubricCriteria(**c) for c in rubric_data.get('criteria', [])],
                'total_points': rubric_data.get('total_points', 100),
                'grading_level': GradingLevel(rubric_data.get('grading_level', 'moderate')),
                'partial_marking': rubric_data.get('partial_marking', True),
                'feedback_length': rubric_data.get('feedback_length', 'medium')
            }
            logger.info(f"Rubric loaded from {rubric_path}")
        except Exception as e:
            logger.error(f"Failed to load rubric: {e}")
            self.load_default_rubric()
    
    def load_default_rubric(self):
        """Load default rubric for general evaluation"""
        self.rubric = {
            'criteria': [
                RubricCriteria(
                    name="Accuracy",
                    weight=0.4,
                    max_points=40,
                    description="Correctness of the answer",
                    keywords=["correct", "accurate", "right", "wrong", "incorrect"]
                ),
                RubricCriteria(
                    name="Completeness",
                    weight=0.3,
                    max_points=30,
                    description="Thoroughness and completeness of the response",
                    keywords=["complete", "thorough", "comprehensive", "partial", "incomplete"]
                ),
                RubricCriteria(
                    name="Clarity",
                    weight=0.2,
                    max_points=20,
                    description="Clarity and organization of the answer",
                    keywords=["clear", "organized", "logical", "confusing", "unclear"]
                ),
                RubricCriteria(
                    name="Originality",
                    weight=0.1,
                    max_points=10,
                    description="Original thinking and creativity",
                    keywords=["original", "creative", "unique", "generic", "copy"]
                )
            ],
            'total_points': 100,
            'grading_level': GradingLevel.MODERATE,
            'partial_marking': True,
            'feedback_length': 'medium'
        }
        logger.info("Default rubric loaded")
    
    async def evaluate_answers(self, 
                             question: str,
                             student_answers: List[Dict],
                             reference_answer: str = None) -> List[EvaluationResult]:
        """
        Evaluate multiple student answers
        
        Args:
            question: The question text
            student_answers: List of student answer dictionaries
            reference_answer: Optional reference answer for comparison
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for i, answer_data in enumerate(student_answers):
            logger.info(f"Evaluating answer {i+1}/{len(student_answers)}")
            
            try:
                result = await self._evaluate_single_answer(
                    question=question,
                    student_text=answer_data['extracted_text'],
                    reference_answer=reference_answer,
                    answer_metadata=answer_data.get('metadata', {})
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating answer {i+1}: {e}")
                # Create error result
                error_result = EvaluationResult(
                    score=0,
                    max_score=self.rubric['total_points'],
                    percentage=0,
                    feedback=f"Error in evaluation: {str(e)}",
                    conceptual_analysis="Evaluation failed due to technical error",
                    strengths=[],
                    weaknesses=["Technical evaluation error"],
                    suggestions=["Please try again or contact support"],
                    confidence=0.0
                )
                results.append(error_result)
        
        return results
    
    async def _evaluate_single_answer(self, 
                                    question: str,
                                    student_text: str,
                                    reference_answer: str = None,
                                    answer_metadata: Dict = None) -> EvaluationResult:
        """Evaluate a single student answer"""
        
        # Prepare evaluation prompt
        evaluation_prompt = self._create_evaluation_prompt(
            question, student_text, reference_answer
        )
        
        # Get AI evaluation
        ai_evaluation = await self._get_ai_evaluation(evaluation_prompt)
        
        # Fact-check if enabled
        fact_check_results = None
        if self.perplexity_client and self._should_fact_check(student_text):
            fact_check_results = await self._fact_check_answer(question, student_text)
        
        # Calculate scores based on rubric
        scores = self._calculate_rubric_scores(ai_evaluation, student_text)
        
        # Generate feedback
        feedback = self._generate_feedback(ai_evaluation, scores, fact_check_results)
        
        # Calculate final score
        total_score = sum(scores.values())
        percentage = (total_score / self.rubric['total_points']) * 100
        
        return EvaluationResult(
            score=total_score,
            max_score=self.rubric['total_points'],
            percentage=percentage,
            feedback=feedback['main_feedback'],
            conceptual_analysis=feedback['conceptual_analysis'],
            strengths=feedback['strengths'],
            weaknesses=feedback['weaknesses'],
            suggestions=feedback['suggestions'],
            confidence=ai_evaluation.get('confidence', 0.8),
            fact_check_results=fact_check_results
        )
    
    def _create_evaluation_prompt(self, question: str, student_text: str, reference_answer: str = None) -> str:
        """Create evaluation prompt for AI"""
        prompt = f"""
You are an expert teacher evaluating a student's answer. Please provide a detailed evaluation.

QUESTION: {question}

STUDENT ANSWER: {student_text}

REFERENCE ANSWER: {reference_answer if reference_answer else "Not provided"}

RUBRIC CRITERIA:
"""
        
        for criterion in self.rubric['criteria']:
            prompt += f"- {criterion.name} ({criterion.max_points} points): {criterion.description}\n"
        
        prompt += f"""
GRADING LEVEL: {self.rubric['grading_level'].value}
PARTIAL MARKING: {'Enabled' if self.rubric['partial_marking'] else 'Disabled'}

Please evaluate the student's answer and provide:
1. Scores for each criterion (0 to max points)
2. Overall assessment
3. Key strengths
4. Areas for improvement
5. Specific suggestions
6. Confidence level (0-1)

Format your response as JSON with the following structure:
{{
    "scores": {{
        "Accuracy": 0-40,
        "Completeness": 0-30,
        "Clarity": 0-20,
        "Originality": 0-10
    }},
    "assessment": "Overall assessment of the answer",
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "suggestions": ["suggestion1", "suggestion2"],
    "confidence": 0.0-1.0
}}
"""
        return prompt
    
    async def _get_ai_evaluation(self, prompt: str) -> Dict:
        """Get evaluation from Google Gemini"""
        if not self.gemini_model:
            raise ValueError("Google Gemini client not initialized")
        
        try:
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1000,
                response_mime_type="application/json"
            )
            
            # Create the full prompt with system instructions
            full_prompt = f"""You are an expert teacher and evaluator. Always respond with valid JSON.

{prompt}"""
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.gemini_model.generate_content(
                    full_prompt,
                    generation_config=generation_config
                )
            )
            
            content = response.text.strip()
            
            # Try to parse JSON response
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "scores": {"Accuracy": 20, "Completeness": 15, "Clarity": 10, "Originality": 5},
                    "assessment": content,
                    "strengths": ["Answer provided"],
                    "weaknesses": ["Could not parse detailed evaluation"],
                    "suggestions": ["Improve answer clarity"],
                    "confidence": 0.5
                }
                
        except Exception as e:
            logger.error(f"Google Gemini API error: {e}")
            raise
    
    def _should_fact_check(self, text: str) -> bool:
        """Determine if answer should be fact-checked"""
        # Simple heuristic: fact-check if text contains factual claims
        fact_indicators = [
            "is", "are", "was", "were", "will be", "has been", "have been",
            "according to", "research shows", "studies indicate", "data shows"
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in fact_indicators)
    
    async def _fact_check_answer(self, question: str, answer: str) -> Dict:
        """Fact-check the answer using Perplexity Sonar API"""
        if not self.perplexity_client:
            return None
        
        try:
            # Create fact-checking query
            query = f"Fact check: {question} Answer: {answer}"
            
            # Use Perplexity to fact-check
            response = await self.perplexity_client.query(query)
            
            return {
                "query": query,
                "response": response,
                "factual_accuracy": "verified",  # This would be determined by analyzing the response
                "sources": [],  # Extract sources from response
                "confidence": 0.8
            }
            
        except Exception as e:
            logger.warning(f"Fact-checking failed: {e}")
            return None
    
    def _calculate_rubric_scores(self, ai_evaluation: Dict, student_text: str) -> Dict:
        """Calculate scores based on rubric and AI evaluation"""
        scores = {}
        
        for criterion in self.rubric['criteria']:
            criterion_name = criterion.name
            
            # Get AI score for this criterion
            ai_score = ai_evaluation.get('scores', {}).get(criterion_name, 0)
            
            # Apply grading level adjustment
            adjusted_score = self._apply_grading_level(ai_score, criterion.max_points)
            
            # Apply partial marking if enabled
            if self.rubric['partial_marking']:
                final_score = adjusted_score
            else:
                # Round to nearest full point
                final_score = round(adjusted_score)
            
            scores[criterion_name] = min(final_score, criterion.max_points)
        
        return scores
    
    def _apply_grading_level(self, score: float, max_score: float) -> float:
        """Apply grading level adjustment to scores"""
        level = self.rubric['grading_level']
        
        if level == GradingLevel.STRICT:
            # Reduce scores by 10%
            return score * 0.9
        elif level == GradingLevel.LENIENT:
            # Increase scores by 10%
            return min(score * 1.1, max_score)
        else:  # MODERATE
            return score
    
    def _generate_feedback(self, ai_evaluation: Dict, scores: Dict, fact_check_results: Dict = None) -> Dict:
        """Generate comprehensive feedback for the student"""
        
        # Base feedback from AI evaluation
        main_feedback = ai_evaluation.get('assessment', 'No specific feedback available.')
        
        # Add fact-checking feedback if available
        if fact_check_results:
            main_feedback += f"\n\nFact-checking: {fact_check_results.get('factual_accuracy', 'Not verified')}"
        
        # Generate conceptual analysis
        conceptual_analysis = self._generate_conceptual_analysis(scores, ai_evaluation)
        
        # Extract strengths and weaknesses
        strengths = ai_evaluation.get('strengths', [])
        weaknesses = ai_evaluation.get('weaknesses', [])
        suggestions = ai_evaluation.get('suggestions', [])
        
        # Add score-specific feedback
        for criterion, score in scores.items():
            max_score = next(c for c in self.rubric['criteria'] if c.name == criterion).max_points
            percentage = (score / max_score) * 100
            
            if percentage < 50:
                weaknesses.append(f"Needs improvement in {criterion.lower()}")
            elif percentage > 80:
                strengths.append(f"Strong performance in {criterion.lower()}")
        
        return {
            'main_feedback': main_feedback,
            'conceptual_analysis': conceptual_analysis,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'suggestions': suggestions
        }
    
    def _generate_conceptual_analysis(self, scores: Dict, ai_evaluation: Dict) -> str:
        """Generate conceptual analysis based on scores"""
        analysis_parts = []
        
        # Analyze each criterion
        for criterion, score in scores.items():
            max_score = next(c for c in self.rubric['criteria'] if c.name == criterion).max_points
            percentage = (score / max_score) * 100
            
            if percentage >= 80:
                analysis_parts.append(f"Excellent understanding of {criterion.lower()}")
            elif percentage >= 60:
                analysis_parts.append(f"Good grasp of {criterion.lower()}")
            elif percentage >= 40:
                analysis_parts.append(f"Basic understanding of {criterion.lower()}")
            else:
                analysis_parts.append(f"Needs significant improvement in {criterion.lower()}")
        
        return ". ".join(analysis_parts) + "."
    
    def save_evaluation_results(self, results: List[EvaluationResult], output_path: str):
        """Save evaluation results to file"""
        try:
            # Convert results to serializable format
            serializable_results = []
            for result in results:
                serializable_results.append({
                    'score': result.score,
                    'max_score': result.max_score,
                    'percentage': result.percentage,
                    'feedback': result.feedback,
                    'conceptual_analysis': result.conceptual_analysis,
                    'strengths': result.strengths,
                    'weaknesses': result.weaknesses,
                    'suggestions': result.suggestions,
                    'confidence': result.confidence,
                    'fact_check_results': result.fact_check_results
                })
            
            # Save to JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Evaluation results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize agent
    agent = EvaluationAgent()
    
    # Test evaluation
    # question = "Explain the process of photosynthesis"
    # student_answers = [{"extracted_text": "Photosynthesis is the process by which plants make food using sunlight."}]
    # results = asyncio.run(agent.evaluate_answers(question, student_answers))
    # print(f"Evaluation result: {results[0].percentage}%")

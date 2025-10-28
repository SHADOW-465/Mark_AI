"""
Unit tests for Evaluation Agent
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import json
import os

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.evaluation import EvaluationAgent, GradingLevel, RubricCriteria, EvaluationResult

class TestEvaluationAgent:
    """Test cases for EvaluationAgent"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.agent = EvaluationAgent()
        self.sample_question = "What is photosynthesis?"
        self.sample_answer = "Photosynthesis is the process by which plants convert light energy into chemical energy."
        self.sample_reference = "Photosynthesis is the process by which plants use sunlight to synthesize foods with the help of chlorophyll."
    
    def test_initialization(self):
        """Test agent initialization"""
        assert self.agent.gemini_model is None  # No API key provided
        assert self.agent.perplexity_client is None  # No API key provided
        assert self.agent.rubric is not None
        assert len(self.agent.rubric['criteria']) == 4  # Default rubric has 4 criteria
    
    def test_initialization_with_api_keys(self):
        """Test initialization with API keys"""
        with patch('google.generativeai.configure') as mock_configure:
            with patch('google.generativeai.GenerativeModel') as mock_model:
                agent = EvaluationAgent(google_gemini_api_key="test_key")
                mock_configure.assert_called_once_with(api_key="test_key")
                mock_model.assert_called_once_with('gemini-1.5-flash')
    
    def test_load_default_rubric(self):
        """Test loading default rubric"""
        self.agent.load_default_rubric()
        
        assert self.agent.rubric is not None
        assert self.agent.rubric['total_points'] == 100
        assert self.agent.rubric['grading_level'] == GradingLevel.MODERATE
        assert self.agent.rubric['partial_marking'] is True
        
        criteria = self.agent.rubric['criteria']
        assert len(criteria) == 4
        
        # Check specific criteria
        accuracy_criteria = next(c for c in criteria if c.name == "Accuracy")
        assert accuracy_criteria.weight == 0.4
        assert accuracy_criteria.max_points == 40
    
    def test_load_rubric_from_file(self):
        """Test loading rubric from JSON file"""
        rubric_data = {
            "criteria": [
                {
                    "name": "Test Criteria",
                    "weight": 0.5,
                    "max_points": 50,
                    "description": "Test description",
                    "keywords": ["test", "example"]
                }
            ],
            "total_points": 100,
            "grading_level": "strict",
            "partial_marking": False,
            "feedback_length": "short"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(rubric_data, f)
            temp_path = f.name
        
        try:
            self.agent.load_rubric(temp_path)
            
            assert self.agent.rubric['total_points'] == 100
            assert self.agent.rubric['grading_level'] == GradingLevel.STRICT
            assert self.agent.rubric['partial_marking'] is False
            assert len(self.agent.rubric['criteria']) == 1
        finally:
            os.unlink(temp_path)
    
    def test_create_evaluation_prompt(self):
        """Test evaluation prompt creation"""
        prompt = self.agent._create_evaluation_prompt(
            self.sample_question,
            self.sample_answer,
            self.sample_reference
        )
        
        assert isinstance(prompt, str)
        assert self.sample_question in prompt
        assert self.sample_answer in prompt
        assert self.sample_reference in prompt
        assert "JSON" in prompt
        assert "scores" in prompt
    
    def test_should_fact_check(self):
        """Test fact-checking decision logic"""
        # Should fact-check
        factual_text = "According to research, the Earth is round."
        assert self.agent._should_fact_check(factual_text) is True
        
        # Should not fact-check
        opinion_text = "I think this is a good answer."
        assert self.agent._should_fact_check(opinion_text) is False
    
    def test_calculate_rubric_scores(self):
        """Test rubric score calculation"""
        ai_evaluation = {
            'scores': {
                'Accuracy': 35,
                'Completeness': 25,
                'Clarity': 18,
                'Originality': 8
            }
        }
        
        scores = self.agent._calculate_rubric_scores(ai_evaluation, self.sample_answer)
        
        assert 'Accuracy' in scores
        assert 'Completeness' in scores
        assert 'Clarity' in scores
        assert 'Originality' in scores
        
        # Scores should be within expected ranges
        assert 0 <= scores['Accuracy'] <= 40
        assert 0 <= scores['Completeness'] <= 30
        assert 0 <= scores['Clarity'] <= 20
        assert 0 <= scores['Originality'] <= 10
    
    def test_apply_grading_level_strict(self):
        """Test strict grading level adjustment"""
        self.agent.rubric['grading_level'] = GradingLevel.STRICT
        score = self.agent._apply_grading_level(80, 100)
        assert score == 72.0  # 80 * 0.9
    
    def test_apply_grading_level_lenient(self):
        """Test lenient grading level adjustment"""
        self.agent.rubric['grading_level'] = GradingLevel.LENIENT
        score = self.agent._apply_grading_level(80, 100)
        assert score == 88.0  # 80 * 1.1
    
    def test_apply_grading_level_moderate(self):
        """Test moderate grading level (no adjustment)"""
        self.agent.rubric['grading_level'] = GradingLevel.MODERATE
        score = self.agent._apply_grading_level(80, 100)
        assert score == 80.0  # No adjustment
    
    def test_generate_feedback(self):
        """Test feedback generation"""
        ai_evaluation = {
            'assessment': 'Good understanding of the concept',
            'strengths': ['Clear explanation', 'Good examples'],
            'weaknesses': ['Could be more detailed'],
            'suggestions': ['Add more examples']
        }
        
        scores = {
            'Accuracy': 35,
            'Completeness': 25,
            'Clarity': 18,
            'Originality': 8
        }
        
        feedback = self.agent._generate_feedback(ai_evaluation, scores)
        
        assert 'main_feedback' in feedback
        assert 'conceptual_analysis' in feedback
        assert 'strengths' in feedback
        assert 'weaknesses' in feedback
        assert 'suggestions' in feedback
        
        assert isinstance(feedback['strengths'], list)
        assert isinstance(feedback['weaknesses'], list)
        assert isinstance(feedback['suggestions'], list)
    
    def test_generate_conceptual_analysis(self):
        """Test conceptual analysis generation"""
        scores = {
            'Accuracy': 35,  # 87.5%
            'Completeness': 15,  # 50%
            'Clarity': 18,  # 90%
            'Originality': 5  # 50%
        }
        
        ai_evaluation = {'assessment': 'Good work'}
        
        analysis = self.agent._generate_conceptual_analysis(scores, ai_evaluation)
        
        assert isinstance(analysis, str)
        assert len(analysis) > 0
        assert analysis.endswith('.')
    
    @pytest.mark.asyncio
    async def test_evaluate_answers_success(self):
        """Test successful answer evaluation"""
        student_answers = [
            {'extracted_text': self.sample_answer, 'metadata': {}}
        ]
        
        # Mock the evaluation process
        with patch.object(self.agent, '_evaluate_single_answer') as mock_eval:
            mock_result = EvaluationResult(
                score=85.0,
                max_score=100.0,
                percentage=85.0,
                feedback="Good answer",
                conceptual_analysis="Shows understanding",
                strengths=["Clear explanation"],
                weaknesses=["Could be more detailed"],
                suggestions=["Add examples"],
                confidence=0.8
            )
            mock_eval.return_value = mock_result
            
            results = await self.agent.evaluate_answers(
                self.sample_question,
                student_answers,
                self.sample_reference
            )
            
            assert len(results) == 1
            assert results[0].score == 85.0
            assert results[0].percentage == 85.0
            assert results[0].feedback == "Good answer"
    
    @pytest.mark.asyncio
    async def test_evaluate_answers_error_handling(self):
        """Test error handling in answer evaluation"""
        student_answers = [
            {'extracted_text': self.sample_answer, 'metadata': {}}
        ]
        
        # Mock evaluation to raise an exception
        with patch.object(self.agent, '_evaluate_single_answer') as mock_eval:
            mock_eval.side_effect = Exception("Evaluation error")
            
            results = await self.agent.evaluate_answers(
                self.sample_question,
                student_answers
            )
            
            assert len(results) == 1
            assert results[0].score == 0
            assert results[0].percentage == 0
            assert "Error in evaluation" in results[0].feedback
    
    @pytest.mark.asyncio
    async def test_get_ai_evaluation_success(self):
        """Test successful AI evaluation"""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps({
            "scores": {"Accuracy": 35, "Completeness": 25, "Clarity": 18, "Originality": 8},
            "assessment": "Good answer",
            "strengths": ["Clear explanation"],
            "weaknesses": ["Could be more detailed"],
            "suggestions": ["Add examples"],
            "confidence": 0.8
        })
        mock_model.generate_content.return_value = mock_response
        
        self.agent.gemini_model = mock_model
        
        result = await self.agent._get_ai_evaluation("Test prompt")
        
        assert result['scores']['Accuracy'] == 35
        assert result['assessment'] == "Good answer"
        assert result['confidence'] == 0.8
    
    @pytest.mark.asyncio
    async def test_get_ai_evaluation_json_error(self):
        """Test AI evaluation with JSON parsing error"""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Invalid JSON response"
        mock_model.generate_content.return_value = mock_response
        
        self.agent.gemini_model = mock_model
        
        result = await self.agent._get_ai_evaluation("Test prompt")
        
        # Should return fallback response
        assert 'scores' in result
        assert 'assessment' in result
        assert result['confidence'] == 0.5
    
    @pytest.mark.asyncio
    async def test_fact_check_answer(self):
        """Test fact-checking functionality"""
        mock_client = Mock()
        mock_client.query.return_value = "Fact-checking response"
        
        self.agent.perplexity_client = mock_client
        
        result = await self.agent._fact_check_answer(self.sample_question, self.sample_answer)
        
        assert result is not None
        assert 'query' in result
        assert 'response' in result
        assert 'factual_accuracy' in result
    
    @pytest.mark.asyncio
    async def test_fact_check_answer_error(self):
        """Test fact-checking error handling"""
        mock_client = Mock()
        mock_client.query.side_effect = Exception("API Error")
        
        self.agent.perplexity_client = mock_client
        
        result = await self.agent._fact_check_answer(self.sample_question, self.sample_answer)
        
        assert result is None
    
    def test_save_evaluation_results(self):
        """Test saving evaluation results"""
        results = [
            EvaluationResult(
                score=85.0,
                max_score=100.0,
                percentage=85.0,
                feedback="Good answer",
                conceptual_analysis="Shows understanding",
                strengths=["Clear explanation"],
                weaknesses=["Could be more detailed"],
                suggestions=["Add examples"],
                confidence=0.8
            )
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            self.agent.save_evaluation_results(results, temp_path)
            
            # Check if file was created and contains data
            assert os.path.exists(temp_path)
            with open(temp_path, 'r') as f:
                data = json.load(f)
                assert len(data) == 1
                assert data[0]['score'] == 85.0
        finally:
            os.unlink(temp_path)
    
    def test_partial_marking_disabled(self):
        """Test scoring when partial marking is disabled"""
        self.agent.rubric['partial_marking'] = False
        
        ai_evaluation = {
            'scores': {
                'Accuracy': 35.7,  # Should be rounded
                'Completeness': 25.3,
                'Clarity': 18.8,
                'Originality': 7.9
            }
        }
        
        scores = self.agent._calculate_rubric_scores(ai_evaluation, self.sample_answer)
        
        # All scores should be rounded to integers
        assert all(isinstance(score, (int, float)) and score == int(score) for score in scores.values())
    
    def test_rubric_criteria_validation(self):
        """Test rubric criteria validation"""
        criteria = RubricCriteria(
            name="Test",
            weight=0.5,
            max_points=50,
            description="Test description",
            keywords=["test", "example"]
        )
        
        assert criteria.name == "Test"
        assert criteria.weight == 0.5
        assert criteria.max_points == 50
        assert criteria.description == "Test description"
        assert criteria.keywords == ["test", "example"]

if __name__ == "__main__":
    pytest.main([__file__])

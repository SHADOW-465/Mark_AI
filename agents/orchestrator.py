"""
Multi-Agent Orchestrator for EduGrade AI
Coordinates the workflow between all agents using LangGraph
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

# Import agents
from .image_preprocessing import ImagePreprocessingAgent
from .ocr_extraction import OCRExtractionAgent
from .evaluation import EvaluationAgent
from .grade_storage import GradeStorageAgent

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolExecutor
    LANGRAPH_AVAILABLE = True
except ImportError:
    LANGRAPH_AVAILABLE = False
    logging.warning("LangGraph not available. Install langgraph")

logger = logging.getLogger(__name__)

@dataclass
class ProcessingState:
    """State object for the multi-agent processing pipeline"""
    # Input data
    image_path: str
    question: str
    reference_answer: Optional[str] = None
    student_id: str = None
    exam_id: str = None
    
    # Processing results
    preprocessed_data: Dict = None
    ocr_results: List[Dict] = None
    evaluation_results: List[Dict] = None
    stored_grades: List[Dict] = None
    
    # Metadata
    processing_start_time: str = None
    processing_end_time: str = None
    errors: List[str] = None
    success: bool = False

class MultiAgentOrchestrator:
    """Orchestrates the multi-agent processing pipeline"""
    
    def __init__(self, 
                 config: Dict[str, Any] = None):
        """
        Initialize the Multi-Agent Orchestrator
        
        Args:
            config: Configuration dictionary for all agents
        """
        self.config = config or {}
        
        # Initialize agents
        self.image_agent = ImagePreprocessingAgent(
            yolo_model_path=self.config.get('yolo_model_path'),
            confidence_threshold=self.config.get('confidence_threshold', 0.5)
        )
        
        self.ocr_agent = OCRExtractionAgent(
            google_credentials_path=self.config.get('google_credentials_path'),
            trocr_model_name=self.config.get('trocr_model_name', 'microsoft/trocr-base-handwritten'),
            languages=self.config.get('languages', ['en', 'hi', 'ta'])
        )
        
        self.evaluation_agent = EvaluationAgent(
            google_gemini_api_key=self.config.get('google_gemini_api_key'),
            perplexity_api_key=self.config.get('perplexity_api_key'),
            rubric_path=self.config.get('rubric_path')
        )
        
        self.storage_agent = GradeStorageAgent(
            db_path=self.config.get('db_path', 'grades.db'),
            storage_dir=self.config.get('storage_dir', './grades')
        )
        
        # Initialize workflow
        if LANGRAPH_AVAILABLE:
            self.workflow = self._create_workflow()
        else:
            self.workflow = None
            logger.warning("LangGraph not available, using sequential processing")
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(ProcessingState)
        
        # Add nodes
        workflow.add_node("preprocess", self._preprocess_node)
        workflow.add_node("ocr_extract", self._ocr_extract_node)
        workflow.add_node("evaluate", self._evaluate_node)
        workflow.add_node("store_grades", self._store_grades_node)
        
        # Add edges
        workflow.add_edge("preprocess", "ocr_extract")
        workflow.add_edge("ocr_extract", "evaluate")
        workflow.add_edge("evaluate", "store_grades")
        workflow.add_edge("store_grades", END)
        
        # Set entry point
        workflow.set_entry_point("preprocess")
        
        return workflow.compile()
    
    async def process_answer_sheet(self, 
                                 image_path: str,
                                 question: str,
                                 reference_answer: str = None,
                                 student_id: str = None,
                                 exam_id: str = None) -> ProcessingState:
        """
        Process an answer sheet through the complete pipeline
        
        Args:
            image_path: Path to the answer sheet image
            question: The question text
            reference_answer: Optional reference answer
            student_id: Student identifier
            exam_id: Exam identifier
            
        Returns:
            ProcessingState with all results
        """
        # Create initial state
        state = ProcessingState(
            image_path=image_path,
            question=question,
            reference_answer=reference_answer,
            student_id=student_id or f"STU_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            exam_id=exam_id or f"EXAM_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            processing_start_time=datetime.now().isoformat(),
            errors=[],
            success=False
        )
        
        try:
            if self.workflow:
                # Use LangGraph workflow
                result = await self.workflow.ainvoke(state)
                return result
            else:
                # Use sequential processing
                return await self._sequential_processing(state)
                
        except Exception as e:
            logger.error(f"Error in processing pipeline: {e}")
            state.errors.append(str(e))
            state.success = False
            state.processing_end_time = datetime.now().isoformat()
            return state
    
    async def _sequential_processing(self, state: ProcessingState) -> ProcessingState:
        """Sequential processing fallback when LangGraph is not available"""
        try:
            # Step 1: Image preprocessing
            logger.info("Starting image preprocessing...")
            state.preprocessed_data = self.image_agent.preprocess_image(state.image_path)
            
            if not state.preprocessed_data.get('preprocessing_success', False):
                raise Exception("Image preprocessing failed")
            
            # Step 2: OCR extraction
            logger.info("Starting OCR extraction...")
            patches = state.preprocessed_data.get('answer_patches', [])
            state.ocr_results = self.ocr_agent.extract_text_from_patches(patches)
            
            # Step 3: Evaluation
            logger.info("Starting evaluation...")
            student_answers = [{'extracted_text': result['extracted_text'], 'metadata': result.get('metadata', {})} 
                             for result in state.ocr_results]
            state.evaluation_results = await self.evaluation_agent.evaluate_answers(
                state.question, student_answers, state.reference_answer
            )
            
            # Step 4: Store grades
            logger.info("Starting grade storage...")
            state.stored_grades = []
            for i, eval_result in enumerate(state.evaluation_results):
                grade_record = self.storage_agent.store_grade(
                    student_id=state.student_id,
                    exam_id=state.exam_id,
                    question_id=f"Q{i+1:03d}",
                    answer_text=state.ocr_results[i]['extracted_text'],
                    score=eval_result.score,
                    max_score=eval_result.max_score,
                    percentage=eval_result.percentage,
                    feedback=eval_result.feedback,
                    metadata={
                        'question': state.question,
                        'reference_answer': state.reference_answer,
                        'ocr_confidence': state.ocr_results[i].get('confidence', 0.0),
                        'evaluation_confidence': eval_result.confidence,
                        'processing_timestamp': state.processing_start_time
                    }
                )
                state.stored_grades.append(grade_record)
            
            state.success = True
            state.processing_end_time = datetime.now().isoformat()
            
            logger.info("Processing completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error in sequential processing: {e}")
            state.errors.append(str(e))
            state.success = False
            state.processing_end_time = datetime.now().isoformat()
            return state
    
    # LangGraph node functions
    async def _preprocess_node(self, state: ProcessingState) -> ProcessingState:
        """Image preprocessing node"""
        try:
            logger.info("Processing image preprocessing node...")
            state.preprocessed_data = self.image_agent.preprocess_image(state.image_path)
            
            if not state.preprocessed_data.get('preprocessing_success', False):
                raise Exception("Image preprocessing failed")
            
            logger.info("Image preprocessing completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error in preprocessing node: {e}")
            state.errors.append(f"Preprocessing error: {str(e)}")
            return state
    
    async def _ocr_extract_node(self, state: ProcessingState) -> ProcessingState:
        """OCR extraction node"""
        try:
            logger.info("Processing OCR extraction node...")
            
            if not state.preprocessed_data or not state.preprocessed_data.get('preprocessing_success'):
                raise Exception("No preprocessed data available")
            
            patches = state.preprocessed_data.get('answer_patches', [])
            state.ocr_results = self.ocr_agent.extract_text_from_patches(patches)
            
            logger.info(f"OCR extraction completed: {len(state.ocr_results)} answers extracted")
            return state
            
        except Exception as e:
            logger.error(f"Error in OCR extraction node: {e}")
            state.errors.append(f"OCR extraction error: {str(e)}")
            return state
    
    async def _evaluate_node(self, state: ProcessingState) -> ProcessingState:
        """Evaluation node"""
        try:
            logger.info("Processing evaluation node...")
            
            if not state.ocr_results:
                raise Exception("No OCR results available")
            
            student_answers = [{'extracted_text': result['extracted_text'], 'metadata': result.get('metadata', {})} 
                             for result in state.ocr_results]
            
            state.evaluation_results = await self.evaluation_agent.evaluate_answers(
                state.question, student_answers, state.reference_answer
            )
            
            logger.info(f"Evaluation completed: {len(state.evaluation_results)} answers evaluated")
            return state
            
        except Exception as e:
            logger.error(f"Error in evaluation node: {e}")
            state.errors.append(f"Evaluation error: {str(e)}")
            return state
    
    async def _store_grades_node(self, state: ProcessingState) -> ProcessingState:
        """Grade storage node"""
        try:
            logger.info("Processing grade storage node...")
            
            if not state.evaluation_results or not state.ocr_results:
                raise Exception("No evaluation results available")
            
            state.stored_grades = []
            for i, eval_result in enumerate(state.evaluation_results):
                grade_record = self.storage_agent.store_grade(
                    student_id=state.student_id,
                    exam_id=state.exam_id,
                    question_id=f"Q{i+1:03d}",
                    answer_text=state.ocr_results[i]['extracted_text'],
                    score=eval_result.score,
                    max_score=eval_result.max_score,
                    percentage=eval_result.percentage,
                    feedback=eval_result.feedback,
                    metadata={
                        'question': state.question,
                        'reference_answer': state.reference_answer,
                        'ocr_confidence': state.ocr_results[i].get('confidence', 0.0),
                        'evaluation_confidence': eval_result.confidence,
                        'processing_timestamp': state.processing_start_time
                    }
                )
                state.stored_grades.append(grade_record)
            
            state.success = True
            state.processing_end_time = datetime.now().isoformat()
            
            logger.info(f"Grade storage completed: {len(state.stored_grades)} grades stored")
            return state
            
        except Exception as e:
            logger.error(f"Error in grade storage node: {e}")
            state.errors.append(f"Grade storage error: {str(e)}")
            return state
    
    def get_processing_summary(self, state: ProcessingState) -> Dict[str, Any]:
        """Get a summary of the processing results"""
        if not state.success:
            return {
                'success': False,
                'errors': state.errors,
                'processing_time': None
            }
        
        # Calculate processing time
        if state.processing_start_time and state.processing_end_time:
            start = datetime.fromisoformat(state.processing_start_time)
            end = datetime.fromisoformat(state.processing_end_time)
            processing_time = (end - start).total_seconds()
        else:
            processing_time = None
        
        # Calculate average scores
        if state.evaluation_results:
            avg_score = sum(r.percentage for r in state.evaluation_results) / len(state.evaluation_results)
            total_answers = len(state.evaluation_results)
        else:
            avg_score = 0
            total_answers = 0
        
        return {
            'success': True,
            'student_id': state.student_id,
            'exam_id': state.exam_id,
            'total_answers': total_answers,
            'average_score': round(avg_score, 2),
            'processing_time_seconds': processing_time,
            'preprocessing_success': state.preprocessed_data.get('preprocessing_success', False) if state.preprocessed_data else False,
            'ocr_success': len(state.ocr_results) > 0 if state.ocr_results else False,
            'evaluation_success': len(state.evaluation_results) > 0 if state.evaluation_results else False,
            'storage_success': len(state.stored_grades) > 0 if state.stored_grades else False,
            'errors': state.errors
        }
    
    def save_processing_report(self, state: ProcessingState, output_path: str):
        """Save a detailed processing report"""
        try:
            report = {
                'processing_summary': self.get_processing_summary(state),
                'preprocessed_data': state.preprocessed_data,
                'ocr_results': state.ocr_results,
                'evaluation_results': [
                    {
                        'score': r.score,
                        'max_score': r.max_score,
                        'percentage': r.percentage,
                        'feedback': r.feedback,
                        'conceptual_analysis': r.conceptual_analysis,
                        'strengths': r.strengths,
                        'weaknesses': r.weaknesses,
                        'suggestions': r.suggestions,
                        'confidence': r.confidence
                    } for r in state.evaluation_results
                ] if state.evaluation_results else [],
                'stored_grades': [
                    {
                        'student_id': g.student_id,
                        'exam_id': g.exam_id,
                        'question_id': g.question_id,
                        'score': g.score,
                        'max_score': g.max_score,
                        'percentage': g.percentage,
                        'feedback': g.feedback,
                        'timestamp': g.timestamp,
                        'current_hash': g.current_hash
                    } for g in state.stored_grades
                ] if state.stored_grades else [],
                'metadata': {
                    'image_path': state.image_path,
                    'question': state.question,
                    'reference_answer': state.reference_answer,
                    'processing_start_time': state.processing_start_time,
                    'processing_end_time': state.processing_end_time
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Processing report saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save processing report: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize orchestrator
    config = {
        'yolo_model_path': None,  # Will use OpenCV fallback
        'confidence_threshold': 0.5,
        'google_credentials_path': None,  # Will use TrOCR fallback
        'trocr_model_name': 'microsoft/trocr-base-handwritten',
        'languages': ['en', 'hi', 'ta'],
        'google_gemini_api_key': None,  # Set your API key
        'perplexity_api_key': None,  # Set your API key
        'rubric_path': None,  # Will use default rubric
        'db_path': 'test_grades.db',
        'storage_dir': './test_grades'
    }
    
    orchestrator = MultiAgentOrchestrator(config)
    
    # Test processing
    # result = asyncio.run(orchestrator.process_answer_sheet(
    #     image_path="sample_answer_sheet.jpg",
    #     question="Explain the process of photosynthesis",
    #     reference_answer="Photosynthesis is the process by which plants convert light energy into chemical energy...",
    #     student_id="STU001",
    #     exam_id="EXAM001"
    # ))
    # 
    # summary = orchestrator.get_processing_summary(result)
    # print(f"Processing summary: {summary}")

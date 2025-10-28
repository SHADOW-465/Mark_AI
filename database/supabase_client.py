"""
Supabase client for EduGrade AI
Handles database operations using Supabase
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from supabase import create_client, Client
from config.settings import settings

logger = logging.getLogger(__name__)

class SupabaseClient:
    """Supabase client for database operations"""
    
    def __init__(self):
        """Initialize Supabase client"""
        self.url = settings.supabase_url
        self.key = settings.supabase_key
        
        if not self.url or not self.key:
            raise ValueError("Supabase URL and key must be provided")
        
        self.client: Client = create_client(self.url, self.key)
        logger.info("Supabase client initialized successfully")
    
    def create_grade(self, grade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new grade record
        
        Args:
            grade_data: Dictionary containing grade information
            
        Returns:
            Created grade record
        """
        try:
            result = self.client.table('grades').insert(grade_data).execute()
            if result.data:
                logger.info(f"Grade created successfully: {result.data[0]['id']}")
                return result.data[0]
            else:
                raise Exception("Failed to create grade")
        except Exception as e:
            logger.error(f"Error creating grade: {e}")
            raise
    
    def get_grade(self, grade_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a grade by ID
        
        Args:
            grade_id: Grade ID
            
        Returns:
            Grade record or None
        """
        try:
            result = self.client.table('grades').select('*').eq('id', grade_id).execute()
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            logger.error(f"Error getting grade: {e}")
            return None
    
    def get_student_grades(self, student_id: str, exam_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get grades for a specific student
        
        Args:
            student_id: Student identifier
            exam_id: Optional exam identifier to filter
            
        Returns:
            List of grade records
        """
        try:
            query = self.client.table('grades').select('*').eq('student_id', student_id)
            
            if exam_id:
                query = query.eq('exam_id', exam_id)
            
            result = query.order('created_at', desc=False).execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Error getting student grades: {e}")
            return []
    
    def get_exam_grades(self, exam_id: str) -> List[Dict[str, Any]]:
        """
        Get all grades for a specific exam
        
        Args:
            exam_id: Exam identifier
            
        Returns:
            List of grade records
        """
        try:
            result = self.client.table('grades').select('*').eq('exam_id', exam_id).order('created_at', desc=False).execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Error getting exam grades: {e}")
            return []
    
    def update_grade(self, grade_id: int, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update a grade record
        
        Args:
            grade_id: Grade ID
            update_data: Data to update
            
        Returns:
            Updated grade record or None
        """
        try:
            result = self.client.table('grades').update(update_data).eq('id', grade_id).execute()
            if result.data:
                logger.info(f"Grade updated successfully: {grade_id}")
                return result.data[0]
            return None
        except Exception as e:
            logger.error(f"Error updating grade: {e}")
            return None
    
    def delete_grade(self, grade_id: int) -> bool:
        """
        Delete a grade record
        
        Args:
            grade_id: Grade ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = self.client.table('grades').delete().eq('id', grade_id).execute()
            logger.info(f"Grade deleted successfully: {grade_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting grade: {e}")
            return False
    
    def create_student(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new student record
        
        Args:
            student_data: Dictionary containing student information
            
        Returns:
            Created student record
        """
        try:
            result = self.client.table('students').insert(student_data).execute()
            if result.data:
                logger.info(f"Student created successfully: {result.data[0]['id']}")
                return result.data[0]
            else:
                raise Exception("Failed to create student")
        except Exception as e:
            logger.error(f"Error creating student: {e}")
            raise
    
    def get_student(self, student_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a student by ID
        
        Args:
            student_id: Student identifier
            
        Returns:
            Student record or None
        """
        try:
            result = self.client.table('students').select('*').eq('student_id', student_id).execute()
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            logger.error(f"Error getting student: {e}")
            return None
    
    def create_exam(self, exam_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new exam record
        
        Args:
            exam_data: Dictionary containing exam information
            
        Returns:
            Created exam record
        """
        try:
            result = self.client.table('exams').insert(exam_data).execute()
            if result.data:
                logger.info(f"Exam created successfully: {result.data[0]['id']}")
                return result.data[0]
            else:
                raise Exception("Failed to create exam")
        except Exception as e:
            logger.error(f"Error creating exam: {e}")
            raise
    
    def get_exam(self, exam_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an exam by ID
        
        Args:
            exam_id: Exam identifier
            
        Returns:
            Exam record or None
        """
        try:
            result = self.client.table('exams').select('*').eq('exam_id', exam_id).execute()
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            logger.error(f"Error getting exam: {e}")
            return None
    
    def create_processing_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new processing job record
        
        Args:
            job_data: Dictionary containing job information
            
        Returns:
            Created job record
        """
        try:
            result = self.client.table('processing_jobs').insert(job_data).execute()
            if result.data:
                logger.info(f"Processing job created successfully: {result.data[0]['id']}")
                return result.data[0]
            else:
                raise Exception("Failed to create processing job")
        except Exception as e:
            logger.error(f"Error creating processing job: {e}")
            raise
    
    def update_processing_job(self, job_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update a processing job record
        
        Args:
            job_id: Job identifier
            update_data: Data to update
            
        Returns:
            Updated job record or None
        """
        try:
            result = self.client.table('processing_jobs').update(update_data).eq('job_id', job_id).execute()
            if result.data:
                logger.info(f"Processing job updated successfully: {job_id}")
                return result.data[0]
            return None
        except Exception as e:
            logger.error(f"Error updating processing job: {e}")
            return None
    
    def get_processing_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a processing job by ID
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job record or None
        """
        try:
            result = self.client.table('processing_jobs').select('*').eq('job_id', job_id).execute()
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            logger.error(f"Error getting processing job: {e}")
            return None
    
    def create_rubric(self, rubric_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new rubric record
        
        Args:
            rubric_data: Dictionary containing rubric information
            
        Returns:
            Created rubric record
        """
        try:
            result = self.client.table('rubrics').insert(rubric_data).execute()
            if result.data:
                logger.info(f"Rubric created successfully: {result.data[0]['id']}")
                return result.data[0]
            else:
                raise Exception("Failed to create rubric")
        except Exception as e:
            logger.error(f"Error creating rubric: {e}")
            raise
    
    def get_rubric(self, rubric_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a rubric by ID
        
        Args:
            rubric_id: Rubric identifier
            
        Returns:
            Rubric record or None
        """
        try:
            result = self.client.table('rubrics').select('*').eq('rubric_id', rubric_id).execute()
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            logger.error(f"Error getting rubric: {e}")
            return None
    
    def get_exam_analytics(self, exam_id: str) -> Dict[str, Any]:
        """
        Get analytics for a specific exam
        
        Args:
            exam_id: Exam identifier
            
        Returns:
            Dictionary with exam analytics
        """
        try:
            # Get all grades for the exam
            grades = self.get_exam_grades(exam_id)
            
            if not grades:
                return {
                    'exam_id': exam_id,
                    'total_answers': 0,
                    'unique_students': 0,
                    'average_percentage': 0,
                    'min_percentage': 0,
                    'max_percentage': 0,
                    'grade_distribution': {}
                }
            
            # Calculate statistics
            total_answers = len(grades)
            unique_students = len(set(grade['student_id'] for grade in grades))
            percentages = [grade['percentage'] for grade in grades]
            
            average_percentage = sum(percentages) / len(percentages) if percentages else 0
            min_percentage = min(percentages) if percentages else 0
            max_percentage = max(percentages) if percentages else 0
            
            # Calculate grade distribution
            grade_distribution = {}
            for grade in grades:
                percentage = grade['percentage']
                if percentage >= 90:
                    grade_letter = 'A'
                elif percentage >= 80:
                    grade_letter = 'B'
                elif percentage >= 70:
                    grade_letter = 'C'
                elif percentage >= 60:
                    grade_letter = 'D'
                else:
                    grade_letter = 'F'
                
                grade_distribution[grade_letter] = grade_distribution.get(grade_letter, 0) + 1
            
            return {
                'exam_id': exam_id,
                'total_answers': total_answers,
                'unique_students': unique_students,
                'average_percentage': round(average_percentage, 2),
                'min_percentage': round(min_percentage, 2),
                'max_percentage': round(max_percentage, 2),
                'grade_distribution': grade_distribution
            }
            
        except Exception as e:
            logger.error(f"Error getting exam analytics: {e}")
            return {}
    
    def verify_chain_integrity(self) -> Dict[str, Any]:
        """
        Verify the integrity of the grade chain
        
        Returns:
            Dictionary with verification results
        """
        try:
            # Get all grades ordered by creation time
            result = self.client.table('grades').select('*').order('created_at', desc=False).execute()
            grades = result.data or []
            
            verification_result = {
                'is_valid': True,
                'total_records': len(grades),
                'invalid_records': [],
                'chain_breaks': []
            }
            
            for i, grade in enumerate(grades):
                # Verify current record hash
                expected_hash = self._calculate_hash(grade)
                if grade.get('current_hash') != expected_hash:
                    verification_result['invalid_records'].append({
                        'index': i,
                        'student_id': grade['student_id'],
                        'question_id': grade['question_id'],
                        'expected_hash': expected_hash,
                        'actual_hash': grade.get('current_hash')
                    })
                    verification_result['is_valid'] = False
                
                # Verify chain continuity
                if i > 0:
                    if grade.get('previous_hash') != grades[i-1].get('current_hash'):
                        verification_result['chain_breaks'].append({
                            'index': i,
                            'student_id': grade['student_id'],
                            'question_id': grade['question_id'],
                            'expected_previous_hash': grades[i-1].get('current_hash'),
                            'actual_previous_hash': grade.get('previous_hash')
                        })
                        verification_result['is_valid'] = False
            
            logger.info(f"Chain verification completed. Valid: {verification_result['is_valid']}")
            return verification_result
            
        except Exception as e:
            logger.error(f"Chain verification failed: {e}")
            return {
                'is_valid': False,
                'total_records': 0,
                'invalid_records': [],
                'chain_breaks': [],
                'error': str(e)
            }
    
    def _calculate_hash(self, grade: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash for grade record"""
        import hashlib
        
        # Create data string for hashing
        data_string = f"{grade['student_id']}{grade['exam_id']}{grade['question_id']}"
        data_string += f"{grade['answer_text']}{grade['score']}{grade['max_score']}"
        data_string += f"{grade['percentage']}{grade['feedback']}{grade['timestamp']}"
        data_string += f"{grade.get('previous_hash', '')}{json.dumps(grade.get('metadata', {}), sort_keys=True)}"
        
        # Calculate SHA-256 hash
        hash_object = hashlib.sha256(data_string.encode('utf-8'))
        return hash_object.hexdigest()

# Global Supabase client instance
supabase_client = None

def get_supabase_client() -> SupabaseClient:
    """Get or create Supabase client instance"""
    global supabase_client
    if supabase_client is None:
        supabase_client = SupabaseClient()
    return supabase_client

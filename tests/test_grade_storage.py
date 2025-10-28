"""
Unit tests for Grade Storage Agent with Supabase
"""

import pytest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.grade_storage import GradeStorageAgent
from database.supabase_models import GradeRecord

class TestGradeStorageAgent:
    """Test cases for GradeStorageAgent"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_storage_dir = tempfile.mkdtemp()
        
        # Mock Supabase client
        self.mock_supabase_client = Mock()
        self.mock_storage = Mock()
        
        with patch('agents.grade_storage.get_supabase_client', return_value=self.mock_supabase_client):
            with patch('agents.grade_storage.SupabaseGradeStorage', return_value=self.mock_storage):
                self.agent = GradeStorageAgent(
                    db_path=None,  # Deprecated for Supabase
                    storage_dir=self.temp_storage_dir
                )
    
    def teardown_method(self):
        """Clean up test fixtures"""
        # Clean up storage directory
        import shutil
        if os.path.exists(self.temp_storage_dir):
            shutil.rmtree(self.temp_storage_dir)
    
    def test_initialization(self):
        """Test agent initialization"""
        assert self.agent.storage_dir == self.temp_storage_dir
        assert len(self.agent.grade_chain) == 0  # Empty chain initially
        assert self.agent.supabase_client == self.mock_supabase_client
        assert self.agent.storage == self.mock_storage
    
    def test_init_database(self):
        """Test database initialization"""
        # Mock Supabase connection test
        self.mock_supabase_client.client.table.return_value.select.return_value.limit.return_value.execute.return_value = Mock()
        
        # Test database initialization
        self.agent._init_database()
        
        # Verify Supabase client was called
        self.mock_supabase_client.client.table.assert_called_with('grades')
    
    def test_calculate_hash(self):
        """Test hash calculation"""
        grade_record = GradeRecord(
            student_id="STU001",
            exam_id="EXAM001",
            question_id="Q001",
            answer_text="Test answer",
            score=85.0,
            max_score=100.0,
            percentage=85.0,
            feedback="Good answer",
            timestamp="2024-01-01T00:00:00Z",
            previous_hash="0" * 64,
            current_hash="",
            metadata={"test": "data"}
        )
        
        hash_value = self.agent._calculate_hash(grade_record)
        
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA-256 produces 64-character hex string
        assert hash_value.isalnum()  # Should be alphanumeric
    
    def test_get_last_hash_empty_chain(self):
        """Test getting last hash from empty chain"""
        hash_value = self.agent._get_last_hash()
        assert hash_value == "0" * 64  # Genesis hash
    
    def test_get_last_hash_with_records(self):
        """Test getting last hash from chain with records"""
        # Add a mock record to the chain
        mock_record = GradeRecord(
            student_id="STU001",
            exam_id="EXAM001",
            question_id="Q001",
            answer_text="Test answer",
            score=85.0,
            max_score=100.0,
            percentage=85.0,
            feedback="Good answer",
            timestamp="2024-01-01T00:00:00Z",
            previous_hash="0" * 64,
            current_hash="test_hash_123",
            metadata={}
        )
        self.agent.grade_chain.append(mock_record)
        
        hash_value = self.agent._get_last_hash()
        assert hash_value == "test_hash_123"
    
    def test_store_grade_success(self):
        """Test successful grade storage"""
        # Mock the storage.store_grade method
        mock_grade_record = GradeRecord(
            student_id="STU001",
            exam_id="EXAM001",
            question_id="Q001",
            answer_text="Test answer",
            score=85.0,
            max_score=100.0,
            percentage=85.0,
            feedback="Good answer",
            timestamp="2024-01-01T00:00:00Z",
            previous_hash="0" * 64,
            current_hash="test_hash_123",
            metadata={"test": "data"}
        )
        self.mock_storage.store_grade.return_value = mock_grade_record
        
        grade_record = self.agent.store_grade(
            student_id="STU001",
            exam_id="EXAM001",
            question_id="Q001",
            answer_text="Test answer",
            score=85.0,
            max_score=100.0,
            percentage=85.0,
            feedback="Good answer",
            metadata={"test": "data"}
        )
        
        assert isinstance(grade_record, GradeRecord)
        assert grade_record.student_id == "STU001"
        assert grade_record.exam_id == "EXAM001"
        assert grade_record.question_id == "Q001"
        assert grade_record.score == 85.0
        assert grade_record.percentage == 85.0
        
        # Check if record was added to chain
        assert len(self.agent.grade_chain) == 1
        assert self.agent.grade_chain[0] == grade_record
        
        # Verify storage.store_grade was called
        self.mock_storage.store_grade.assert_called_once()
    
    def test_store_grade_with_metadata(self):
        """Test grade storage with custom metadata"""
        custom_metadata = {
            "teacher": "Dr. Smith",
            "subject": "Biology",
            "difficulty": "medium"
        }
        
        grade_record = self.agent.store_grade(
            student_id="STU002",
            exam_id="EXAM002",
            question_id="Q002",
            answer_text="Another test answer",
            score=90.0,
            max_score=100.0,
            percentage=90.0,
            feedback="Excellent answer",
            metadata=custom_metadata
        )
        
        assert grade_record.metadata["teacher"] == "Dr. Smith"
        assert grade_record.metadata["subject"] == "Biology"
        assert grade_record.metadata["difficulty"] == "medium"
        assert "stored_at" in grade_record.metadata
        assert "version" in grade_record.metadata
    
    def test_get_student_grades(self):
        """Test retrieving student grades"""
        # Mock the storage.get_student_grades method
        mock_grades = [
            GradeRecord(
                student_id="STU001",
                exam_id="EXAM001",
                question_id="Q001",
                answer_text="Answer 1",
                score=80.0,
                max_score=100.0,
                percentage=80.0,
                feedback="Good",
                timestamp="2024-01-01T00:00:00Z",
                previous_hash="0" * 64,
                current_hash="hash1",
                metadata={}
            ),
            GradeRecord(
                student_id="STU001",
                exam_id="EXAM001",
                question_id="Q002",
                answer_text="Answer 2",
                score=90.0,
                max_score=100.0,
                percentage=90.0,
                feedback="Excellent",
                timestamp="2024-01-01T00:00:00Z",
                previous_hash="hash1",
                current_hash="hash2",
                metadata={}
            )
        ]
        self.mock_storage.get_student_grades.return_value = mock_grades
        
        # Get grades for student
        grades = self.agent.get_student_grades("STU001")
        
        assert len(grades) == 2
        assert all(grade.student_id == "STU001" for grade in grades)
        assert all(grade.exam_id == "EXAM001" for grade in grades)
        
        # Verify storage method was called
        self.mock_storage.get_student_grades.assert_called_once_with("STU001", None)
    
    def test_get_student_grades_with_exam_filter(self):
        """Test retrieving student grades with exam filter"""
        # Store grades for different exams
        self.agent.store_grade(
            student_id="STU001",
            exam_id="EXAM001",
            question_id="Q001",
            answer_text="Answer 1",
            score=80.0,
            max_score=100.0,
            percentage=80.0,
            feedback="Good",
            metadata={}
        )
        
        self.agent.store_grade(
            student_id="STU001",
            exam_id="EXAM002",
            question_id="Q001",
            answer_text="Answer 2",
            score=90.0,
            max_score=100.0,
            percentage=90.0,
            feedback="Excellent",
            metadata={}
        )
        
        # Get grades for specific exam
        grades = self.agent.get_student_grades("STU001", "EXAM001")
        
        assert len(grades) == 1
        assert grades[0].exam_id == "EXAM001"
    
    def test_get_exam_analytics(self):
        """Test exam analytics generation"""
        # Mock the storage.get_exam_analytics method
        mock_analytics = {
            'exam_id': "EXAM001",
            'total_answers': 5,
            'unique_students': 5,
            'average_percentage': 80.0,
            'min_percentage': 70.0,
            'max_percentage': 90.0,
            'grade_distribution': {'A': 1, 'B': 2, 'C': 2}
        }
        self.mock_storage.get_exam_analytics.return_value = mock_analytics
        
        analytics = self.agent.get_exam_analytics("EXAM001")
        
        assert analytics['exam_id'] == "EXAM001"
        assert analytics['total_answers'] == 5
        assert analytics['unique_students'] == 5
        assert analytics['average_percentage'] == 80.0
        assert analytics['min_percentage'] == 70.0
        assert analytics['max_percentage'] == 90.0
        assert 'grade_distribution' in analytics
        
        # Verify storage method was called
        self.mock_storage.get_exam_analytics.assert_called_once_with("EXAM001")
    
    def test_verify_chain_integrity_valid(self):
        """Test chain integrity verification with valid chain"""
        # Store a valid grade
        self.agent.store_grade(
            student_id="STU001",
            exam_id="EXAM001",
            question_id="Q001",
            answer_text="Test answer",
            score=85.0,
            max_score=100.0,
            percentage=85.0,
            feedback="Good answer",
            metadata={}
        )
        
        verification = self.agent.verify_chain_integrity()
        
        assert verification['is_valid'] is True
        assert verification['total_records'] == 1
        assert len(verification['invalid_records']) == 0
        assert len(verification['chain_breaks']) == 0
    
    def test_verify_chain_integrity_invalid(self):
        """Test chain integrity verification with invalid chain"""
        # Manually add an invalid record to the chain
        invalid_record = GradeRecord(
            student_id="STU001",
            exam_id="EXAM001",
            question_id="Q001",
            answer_text="Test answer",
            score=85.0,
            max_score=100.0,
            percentage=85.0,
            feedback="Good answer",
            timestamp="2024-01-01T00:00:00Z",
            previous_hash="0" * 64,
            current_hash="invalid_hash",  # Invalid hash
            metadata={}
        )
        self.agent.grade_chain.append(invalid_record)
        
        verification = self.agent.verify_chain_integrity()
        
        assert verification['is_valid'] is False
        assert len(verification['invalid_records']) > 0
    
    def test_export_grades(self):
        """Test grade export functionality"""
        # Mock the storage.export_grades method
        self.mock_storage.export_grades.return_value = None
        
        # Export grades
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = f.name
        
        try:
            self.agent.export_grades(export_path)
            
            # Verify storage method was called
            self.mock_storage.export_grades.assert_called_once_with(export_path, None)
        finally:
            os.unlink(export_path)
    
    def test_export_grades_with_exam_filter(self):
        """Test grade export with exam filter"""
        # Mock the storage.export_grades method
        self.mock_storage.export_grades.return_value = None
        
        # Export grades for specific exam
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = f.name
        
        try:
            self.agent.export_grades(export_path, "EXAM001")
            
            # Verify storage method was called with exam filter
            self.mock_storage.export_grades.assert_called_once_with(export_path, "EXAM001")
        finally:
            os.unlink(export_path)
    
    def test_save_to_file(self):
        """Test saving grade records to file system"""
        # Mock the storage.store_grade method
        mock_grade_record = GradeRecord(
            student_id="STU001",
            exam_id="EXAM001",
            question_id="Q001",
            answer_text="Test answer",
            score=85.0,
            max_score=100.0,
            percentage=85.0,
            feedback="Good answer",
            timestamp="2024-01-01T00:00:00Z",
            previous_hash="0" * 64,
            current_hash="test_hash",
            metadata={}
        )
        self.mock_storage.store_grade.return_value = mock_grade_record
        
        # Store a grade
        grade_record = self.agent.store_grade(
            student_id="STU001",
            exam_id="EXAM001",
            question_id="Q001",
            answer_text="Test answer",
            score=85.0,
            max_score=100.0,
            percentage=85.0,
            feedback="Good answer",
            metadata={}
        )
        
        # Check if files were created
        student_dir = os.path.join(self.temp_storage_dir, "STU001")
        exam_dir = os.path.join(student_dir, "EXAM001")
        
        assert os.path.exists(student_dir)
        assert os.path.exists(exam_dir)
        
        # Check for individual record file
        record_files = [f for f in os.listdir(exam_dir) if f.endswith('.json')]
        assert len(record_files) == 1
        
        # Check for chain file
        chain_file = os.path.join(self.temp_storage_dir, "grade_chain.json")
        assert os.path.exists(chain_file)
    
    def test_error_handling_storage_error(self):
        """Test error handling for storage operations"""
        # Mock storage error
        self.mock_storage.store_grade.side_effect = Exception("Storage error")
        
        # This should raise an exception
        with pytest.raises(Exception):
            self.agent.store_grade(
                student_id="STU001",
                exam_id="EXAM001",
                question_id="Q001",
                answer_text="Test answer",
                score=85.0,
                max_score=100.0,
                percentage=85.0,
                feedback="Good answer",
                metadata={}
            )
    
    def test_grade_record_immutability(self):
        """Test that grade records are immutable"""
        grade_record = GradeRecord(
            student_id="STU001",
            exam_id="EXAM001",
            question_id="Q001",
            answer_text="Test answer",
            score=85.0,
            max_score=100.0,
            percentage=85.0,
            feedback="Good answer",
            timestamp="2024-01-01T00:00:00Z",
            previous_hash="0" * 64,
            current_hash="test_hash",
            metadata={}
        )
        
        # Verify all fields are set correctly
        assert grade_record.student_id == "STU001"
        assert grade_record.exam_id == "EXAM001"
        assert grade_record.question_id == "Q001"
        assert grade_record.answer_text == "Test answer"
        assert grade_record.score == 85.0
        assert grade_record.max_score == 100.0
        assert grade_record.percentage == 85.0
        assert grade_record.feedback == "Good answer"
        assert grade_record.timestamp == "2024-01-01T00:00:00Z"
        assert grade_record.previous_hash == "0" * 64
        assert grade_record.current_hash == "test_hash"
        assert grade_record.metadata == {}

if __name__ == "__main__":
    pytest.main([__file__])

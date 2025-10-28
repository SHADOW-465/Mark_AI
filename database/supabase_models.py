"""
Supabase models for EduGrade AI
Data models and schemas for Supabase database operations
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from dataclasses import dataclass, asdict
from pydantic import BaseModel, Field

class GradeModel(BaseModel):
    """Grade model for Supabase"""
    student_id: str = Field(..., description="Student identifier")
    exam_id: str = Field(..., description="Exam identifier")
    question_id: str = Field(..., description="Question identifier")
    answer_text: str = Field(..., description="Student's answer text")
    score: float = Field(..., description="Achieved score")
    max_score: float = Field(..., description="Maximum possible score")
    percentage: float = Field(..., description="Score percentage")
    feedback: str = Field(..., description="Teacher feedback")
    timestamp: str = Field(..., description="Timestamp of grading")
    previous_hash: str = Field(..., description="Previous record hash")
    current_hash: str = Field(..., description="Current record hash")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    is_override: bool = Field(default=False, description="Whether this is an override")
    override_reason: Optional[str] = Field(None, description="Reason for override")

class StudentModel(BaseModel):
    """Student model for Supabase"""
    student_id: str = Field(..., description="Unique student identifier")
    name: str = Field(..., description="Student name")
    email: Optional[str] = Field(None, description="Student email")
    class_name: Optional[str] = Field(None, description="Class name")

class ExamModel(BaseModel):
    """Exam model for Supabase"""
    exam_id: str = Field(..., description="Unique exam identifier")
    title: str = Field(..., description="Exam title")
    description: Optional[str] = Field(None, description="Exam description")
    subject: Optional[str] = Field(None, description="Subject")
    total_questions: Optional[int] = Field(None, description="Total number of questions")
    max_score: Optional[float] = Field(None, description="Maximum possible score")

class ProcessingJobModel(BaseModel):
    """Processing job model for Supabase"""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(default="pending", description="Job status")
    image_path: str = Field(..., description="Path to image file")
    question: str = Field(..., description="Question text")
    reference_answer: Optional[str] = Field(None, description="Reference answer")
    student_id: Optional[str] = Field(None, description="Student identifier")
    exam_id: Optional[str] = Field(None, description="Exam identifier")
    result_data: Optional[Dict[str, Any]] = Field(None, description="Processing results")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    started_at: Optional[str] = Field(None, description="Job start time")
    completed_at: Optional[str] = Field(None, description="Job completion time")

class RubricModel(BaseModel):
    """Rubric model for Supabase"""
    rubric_id: str = Field(..., description="Unique rubric identifier")
    name: str = Field(..., description="Rubric name")
    description: Optional[str] = Field(None, description="Rubric description")
    criteria: List[Dict[str, Any]] = Field(..., description="Grading criteria")
    total_points: float = Field(..., description="Total possible points")
    grading_level: str = Field(default="moderate", description="Grading strictness level")
    partial_marking: bool = Field(default=True, description="Allow partial marks")

@dataclass
class GradeRecord:
    """Immutable grade record with cryptographic verification"""
    student_id: str
    exam_id: str
    question_id: str
    answer_text: str
    score: float
    max_score: float
    percentage: float
    feedback: str
    timestamp: str
    previous_hash: str
    current_hash: str
    metadata: Dict[str, Any]
    is_override: bool = False
    override_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GradeRecord':
        """Create from dictionary"""
        return cls(**data)

class SupabaseGradeStorage:
    """Grade storage using Supabase"""
    
    def __init__(self, supabase_client):
        """Initialize with Supabase client"""
        self.client = supabase_client
    
    def store_grade(self, grade_data: Dict[str, Any]) -> GradeRecord:
        """
        Store a grade record with cryptographic verification
        
        Args:
            grade_data: Grade data dictionary
            
        Returns:
            GradeRecord object
        """
        try:
            # Create grade record
            grade_record = GradeRecord(
                student_id=grade_data['student_id'],
                exam_id=grade_data['exam_id'],
                question_id=grade_data['question_id'],
                answer_text=grade_data['answer_text'],
                score=grade_data['score'],
                max_score=grade_data['max_score'],
                percentage=grade_data['percentage'],
                feedback=grade_data['feedback'],
                timestamp=grade_data.get('timestamp', datetime.now().isoformat()),
                previous_hash=grade_data.get('previous_hash', '0' * 64),
                current_hash=grade_data.get('current_hash', ''),
                metadata=grade_data.get('metadata', {}),
                is_override=grade_data.get('is_override', False),
                override_reason=grade_data.get('override_reason')
            )
            
            # Calculate hash if not provided
            if not grade_record.current_hash:
                grade_record.current_hash = self._calculate_hash(grade_record)
            
            # Store in Supabase
            result = self.client.create_grade(grade_record.to_dict())
            
            return grade_record
            
        except Exception as e:
            raise Exception(f"Failed to store grade: {e}")
    
    def get_student_grades(self, student_id: str, exam_id: Optional[str] = None) -> List[GradeRecord]:
        """
        Retrieve grades for a specific student
        
        Args:
            student_id: Student identifier
            exam_id: Optional exam identifier to filter
            
        Returns:
            List of grade records
        """
        try:
            grades_data = self.client.get_student_grades(student_id, exam_id)
            return [GradeRecord.from_dict(grade) for grade in grades_data]
        except Exception as e:
            raise Exception(f"Failed to retrieve student grades: {e}")
    
    def get_exam_analytics(self, exam_id: str) -> Dict[str, Any]:
        """
        Get analytics for a specific exam
        
        Args:
            exam_id: Exam identifier
            
        Returns:
            Dictionary with exam analytics
        """
        try:
            return self.client.get_exam_analytics(exam_id)
        except Exception as e:
            raise Exception(f"Failed to get exam analytics: {e}")
    
    def verify_chain_integrity(self) -> Dict[str, Any]:
        """
        Verify the integrity of the grade chain
        
        Returns:
            Dictionary with verification results
        """
        try:
            return self.client.verify_chain_integrity()
        except Exception as e:
            raise Exception(f"Failed to verify chain integrity: {e}")
    
    def export_grades(self, output_path: str, exam_id: Optional[str] = None):
        """
        Export grades to JSON file
        
        Args:
            output_path: Output file path
            exam_id: Optional exam ID to filter
        """
        try:
            if exam_id:
                grades_data = self.client.get_exam_grades(exam_id)
            else:
                # Get all grades
                result = self.client.client.table('grades').select('*').execute()
                grades_data = result.data or []
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(grades_data, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            raise Exception(f"Failed to export grades: {e}")
    
    def _calculate_hash(self, grade_record: GradeRecord) -> str:
        """Calculate SHA-256 hash for grade record"""
        import hashlib
        
        # Create data string for hashing
        data_string = f"{grade_record.student_id}{grade_record.exam_id}{grade_record.question_id}"
        data_string += f"{grade_record.answer_text}{grade_record.score}{grade_record.max_score}"
        data_string += f"{grade_record.percentage}{grade_record.feedback}{grade_record.timestamp}"
        data_string += f"{grade_record.previous_hash}{json.dumps(grade_record.metadata, sort_keys=True)}"
        
        # Calculate SHA-256 hash
        hash_object = hashlib.sha256(data_string.encode('utf-8'))
        return hash_object.hexdigest()

# SQL schema for Supabase tables
SUPABASE_SCHEMA = """
-- Create grades table
CREATE TABLE IF NOT EXISTS grades (
    id BIGSERIAL PRIMARY KEY,
    student_id TEXT NOT NULL,
    exam_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    answer_text TEXT NOT NULL,
    score REAL NOT NULL,
    max_score REAL NOT NULL,
    percentage REAL NOT NULL,
    feedback TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    previous_hash TEXT NOT NULL,
    current_hash TEXT NOT NULL UNIQUE,
    metadata JSONB NOT NULL DEFAULT '{}',
    is_override BOOLEAN DEFAULT FALSE,
    override_reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create students table
CREATE TABLE IF NOT EXISTS students (
    id BIGSERIAL PRIMARY KEY,
    student_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    email TEXT,
    class_name TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create exams table
CREATE TABLE IF NOT EXISTS exams (
    id BIGSERIAL PRIMARY KEY,
    exam_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    subject TEXT,
    total_questions INTEGER,
    max_score REAL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create processing_jobs table
CREATE TABLE IF NOT EXISTS processing_jobs (
    id BIGSERIAL PRIMARY KEY,
    job_id TEXT UNIQUE NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    image_path TEXT NOT NULL,
    question TEXT NOT NULL,
    reference_answer TEXT,
    student_id TEXT,
    exam_id TEXT,
    result_data JSONB,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Create rubrics table
CREATE TABLE IF NOT EXISTS rubrics (
    id BIGSERIAL PRIMARY KEY,
    rubric_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    criteria JSONB NOT NULL,
    total_points REAL NOT NULL,
    grading_level TEXT NOT NULL DEFAULT 'moderate',
    partial_marking BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_grades_student_exam ON grades(student_id, exam_id);
CREATE INDEX IF NOT EXISTS idx_grades_hash ON grades(current_hash);
CREATE INDEX IF NOT EXISTS idx_grades_timestamp ON grades(created_at);
CREATE INDEX IF NOT EXISTS idx_students_student_id ON students(student_id);
CREATE INDEX IF NOT EXISTS idx_exams_exam_id ON exams(exam_id);
CREATE INDEX IF NOT EXISTS idx_processing_jobs_job_id ON processing_jobs(job_id);
CREATE INDEX IF NOT EXISTS idx_processing_jobs_status ON processing_jobs(status);
CREATE INDEX IF NOT EXISTS idx_rubrics_rubric_id ON rubrics(rubric_id);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_grades_updated_at BEFORE UPDATE ON grades FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_students_updated_at BEFORE UPDATE ON students FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_exams_updated_at BEFORE UPDATE ON exams FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_rubrics_updated_at BEFORE UPDATE ON rubrics FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
"""

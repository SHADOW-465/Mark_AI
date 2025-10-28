"""
Database models for EduGrade AI
SQLAlchemy models for the grade storage system
"""

from sqlalchemy import Column, Integer, String, Float, Text, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import json

Base = declarative_base()

class Grade(Base):
    """Grade model for storing student grades"""
    __tablename__ = "grades"
    
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String(100), nullable=False, index=True)
    exam_id = Column(String(100), nullable=False, index=True)
    question_id = Column(String(100), nullable=False, index=True)
    answer_text = Column(Text, nullable=False)
    score = Column(Float, nullable=False)
    max_score = Column(Float, nullable=False)
    percentage = Column(Float, nullable=False)
    feedback = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    previous_hash = Column(String(64), nullable=False)
    current_hash = Column(String(64), nullable=False, unique=True, index=True)
    metadata = Column(Text, nullable=False)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    is_override = Column(Boolean, default=False)
    override_reason = Column(Text, nullable=True)
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'student_id': self.student_id,
            'exam_id': self.exam_id,
            'question_id': self.question_id,
            'answer_text': self.answer_text,
            'score': self.score,
            'max_score': self.max_score,
            'percentage': self.percentage,
            'feedback': self.feedback,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'previous_hash': self.previous_hash,
            'current_hash': self.current_hash,
            'metadata': json.loads(self.metadata) if self.metadata else {},
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'is_override': self.is_override,
            'override_reason': self.override_reason
        }

class Student(Base):
    """Student model for storing student information"""
    __tablename__ = "students"
    
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    email = Column(String(200), nullable=True)
    class_name = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    grades = relationship("Grade", back_populates="student")
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'student_id': self.student_id,
            'name': self.name,
            'email': self.email,
            'class_name': self.class_name,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class Exam(Base):
    """Exam model for storing exam information"""
    __tablename__ = "exams"
    
    id = Column(Integer, primary_key=True, index=True)
    exam_id = Column(String(100), unique=True, nullable=False, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    subject = Column(String(100), nullable=True)
    total_questions = Column(Integer, nullable=True)
    max_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    grades = relationship("Grade", back_populates="exam")
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'exam_id': self.exam_id,
            'title': self.title,
            'description': self.description,
            'subject': self.subject,
            'total_questions': self.total_questions,
            'max_score': self.max_score,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class ProcessingJob(Base):
    """Processing job model for tracking background tasks"""
    __tablename__ = "processing_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(100), unique=True, nullable=False, index=True)
    status = Column(String(50), nullable=False, default="pending")  # pending, processing, completed, failed
    image_path = Column(String(500), nullable=False)
    question = Column(Text, nullable=False)
    reference_answer = Column(Text, nullable=True)
    student_id = Column(String(100), nullable=True)
    exam_id = Column(String(100), nullable=True)
    result_data = Column(Text, nullable=True)  # JSON string
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'job_id': self.job_id,
            'status': self.status,
            'image_path': self.image_path,
            'question': self.question,
            'reference_answer': self.reference_answer,
            'student_id': self.student_id,
            'exam_id': self.exam_id,
            'result_data': json.loads(self.result_data) if self.result_data else None,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }

class Rubric(Base):
    """Rubric model for storing grading rubrics"""
    __tablename__ = "rubrics"
    
    id = Column(Integer, primary_key=True, index=True)
    rubric_id = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    criteria = Column(Text, nullable=False)  # JSON string
    total_points = Column(Float, nullable=False)
    grading_level = Column(String(50), nullable=False, default="moderate")
    partial_marking = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'rubric_id': self.rubric_id,
            'name': self.name,
            'description': self.description,
            'criteria': json.loads(self.criteria) if self.criteria else [],
            'total_points': self.total_points,
            'grading_level': self.grading_level,
            'partial_marking': self.partial_marking,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

# Add relationships
Grade.student = relationship("Student", back_populates="grades")
Grade.exam = relationship("Exam", back_populates="grades")

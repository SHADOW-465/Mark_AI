"""
Grade Storage Agent for EduGrade AI
Handles secure storage of grades with SHA-256 cryptographic hashing
Creates tamper-proof grade records with timestamps
"""

import hashlib
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.firebaseservice import FirebaseService
from services.devdockservice import DevDockService
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GradeRecord:
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
    metadata: dict
    is_override: bool = False
    override_reason: str | None = None

class GradeStorageAgent:
    """Agent responsible for secure grade storage and verification"""
    
    def __init__(self, db_path: str = None, storage_dir: str = "./grades"):
        """
        Initialize the Grade Storage Agent
        
        Args:
            db_path: Deprecated - kept for compatibility
            storage_dir: Directory for file-based storage
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Firebase and DevDock
        self.firebase = FirebaseService()
        self.devdock = DevDockService()
        
        # Chain of grade records for verification
        self.grade_chain = []
        self._load_existing_chain()
    
    def _init_database(self):
        """No-op: using Firebase as the primary store."""
        return
    
    def _load_existing_chain(self):
        """Start with an empty chain; Firebase is the source of truth."""
        self.grade_chain = []
    
    def store_grade(self, 
                   student_id: str,
                   exam_id: str,
                   question_id: str,
                   answer_text: str,
                   score: float,
                   max_score: float,
                   percentage: float,
                   feedback: str,
                   metadata: Dict[str, Any] = None) -> GradeRecord:
        """
        Store a grade record with cryptographic verification
        
        Args:
            student_id: Unique student identifier
            exam_id: Unique exam identifier
            question_id: Unique question identifier
            answer_text: The student's answer text
            score: Achieved score
            max_score: Maximum possible score
            percentage: Score percentage
            feedback: Teacher feedback
            metadata: Additional metadata
            
        Returns:
            GradeRecord object with cryptographic hashes
        """
        try:
            # Prepare metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'stored_at': datetime.now(timezone.utc).isoformat(),
                'version': '1.0',
                'agent': 'grade_storage_agent'
            })
            
            # Get previous hash (last record in chain)
            previous_hash = self._get_last_hash()
            
            # Create timestamp
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # Create grade data document
            grade_data = {
                'student_id': student_id,
                'exam_id': exam_id,
                'question_id': question_id,
                'answer_text': answer_text,
                'score': score,
                'max_score': max_score,
                'percentage': percentage,
                'feedback': feedback,
                'timestamp': timestamp,
                'previous_hash': previous_hash,
                'metadata': metadata
            }
            
            # Compute current hash and build record
            temp = GradeRecord(
                student_id=student_id,
                exam_id=exam_id,
                question_id=question_id,
                answer_text=answer_text,
                score=score,
                max_score=max_score,
                percentage=percentage,
                feedback=feedback,
                timestamp=timestamp,
                previous_hash=previous_hash,
                current_hash="",
                metadata=metadata
            )
            current_hash = self._calculate_hash(temp)
            grade_record = GradeRecord(
                student_id=student_id,
                exam_id=exam_id,
                question_id=question_id,
                answer_text=answer_text,
                score=score,
                max_score=max_score,
                percentage=percentage,
                feedback=feedback,
                timestamp=timestamp,
                previous_hash=previous_hash,
                current_hash=current_hash,
                metadata=metadata
            )

            # Store in Firebase (system of record)
            try:
                self.firebase.store_grade({ **grade_data, 'current_hash': current_hash })
            except Exception as fe:
                logger.warning(f"Firebase store failed: {fe}")
            # Persist hash to DevDock (stub)
            try:
                self.devdock.verify_credential({'grade_hash': current_hash, 'student_id': student_id, 'exam_id': exam_id, 'question_id': question_id})
            except Exception as de:
                logger.warning(f"DevDock persist failed: {de}")
            
            # Add to chain
            self.grade_chain.append(grade_record)
            
            # Save to file system
            self._save_to_file(grade_record)
            
            logger.info(f"Grade stored successfully for student {student_id}, question {question_id}")
            return grade_record
            
        except Exception as e:
            logger.error(f"Failed to store grade: {e}")
            raise
    
    def _get_last_hash(self) -> str:
        """Get hash of the last record in the chain"""
        if not self.grade_chain:
            return "0" * 64  # Genesis hash
        
        return self.grade_chain[-1].current_hash
    
    def _calculate_hash(self, grade_record: GradeRecord) -> str:
        """Calculate SHA-256 hash for grade record"""
        # Create data string for hashing
        data_string = f"{grade_record.student_id}{grade_record.exam_id}{grade_record.question_id}"
        data_string += f"{grade_record.answer_text}{grade_record.score}{grade_record.max_score}"
        data_string += f"{grade_record.percentage}{grade_record.feedback}{grade_record.timestamp}"
        data_string += f"{grade_record.previous_hash}{json.dumps(grade_record.metadata, sort_keys=True)}"
        
        # Calculate SHA-256 hash
        hash_object = hashlib.sha256(data_string.encode('utf-8'))
        return hash_object.hexdigest()
    
    def _store_in_database(self, grade_record: GradeRecord):
        """No-op: Firebase is used for storage."""
        return
    
    def _save_to_file(self, grade_record: GradeRecord):
        """Save grade record to file system for backup"""
        try:
            # Create directory structure
            student_dir = self.storage_dir / grade_record.student_id
            exam_dir = student_dir / grade_record.exam_id
            exam_dir.mkdir(parents=True, exist_ok=True)
            
            # Save individual record
            record_file = exam_dir / f"{grade_record.question_id}_{grade_record.timestamp}.json"
            
            with open(record_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(grade_record), f, indent=2, ensure_ascii=False)
            
            # Update chain file
            chain_file = self.storage_dir / "grade_chain.json"
            chain_data = [asdict(record) for record in self.grade_chain]
            
            with open(chain_file, 'w', encoding='utf-8') as f:
                json.dump(chain_data, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.warning(f"Failed to save to file: {e}")
    
    def verify_chain_integrity(self) -> Dict[str, Any]:
        """
        Verify the integrity of the grade chain
        
        Returns:
            Dictionary with verification results
        """
        verification_result = {
            'is_valid': True,
            'total_records': len(self.grade_chain),
            'invalid_records': [],
            'chain_breaks': []
        }
        
        try:
            for i, record in enumerate(self.grade_chain):
                # Verify current record hash
                expected_hash = self._calculate_hash(record)
                if record.current_hash != expected_hash:
                    verification_result['invalid_records'].append({
                        'index': i,
                        'student_id': record.student_id,
                        'question_id': record.question_id,
                        'expected_hash': expected_hash,
                        'actual_hash': record.current_hash
                    })
                    verification_result['is_valid'] = False
                
                # Verify chain continuity
                if i > 0:
                    if record.previous_hash != self.grade_chain[i-1].current_hash:
                        verification_result['chain_breaks'].append({
                            'index': i,
                            'student_id': record.student_id,
                            'question_id': record.question_id,
                            'expected_previous_hash': self.grade_chain[i-1].current_hash,
                            'actual_previous_hash': record.previous_hash
                        })
                        verification_result['is_valid'] = False
            
            logger.info(f"Chain verification completed. Valid: {verification_result['is_valid']}")
            
        except Exception as e:
            logger.error(f"Chain verification failed: {e}")
            verification_result['is_valid'] = False
            verification_result['error'] = str(e)
        
        return verification_result
    
    def get_student_grades(self, student_id: str, exam_id: str = None) -> List[GradeRecord]:
        """
        Retrieve grades for a specific student
        
        Args:
            student_id: Student identifier
            exam_id: Optional exam identifier to filter
            
        Returns:
            List of grade records
        """
        try:
            docs = self.firebase.get_grades_by_student(student_id, exam_id)
            records = []
            for d in docs:
                records.append(GradeRecord(
                    student_id=d['student_id'],
                    exam_id=d['exam_id'],
                    question_id=d['question_id'],
                    answer_text=d.get('answer_text', ''),
                    score=d['score'],
                    max_score=d['max_score'],
                    percentage=d['percentage'],
                    feedback=d['feedback'],
                    timestamp=d['timestamp'],
                    previous_hash=d.get('previous_hash', ''),
                    current_hash=d.get('current_hash', ''),
                    metadata=d.get('metadata', {}),
                    is_override=d.get('is_override', False),
                    override_reason=d.get('override_reason')
                ))
            return records
        except Exception as e:
            logger.error(f"Failed to retrieve student grades: {e}")
            return []
    
    def get_exam_analytics(self, exam_id: str) -> Dict[str, Any]:
        """
        Get analytics for a specific exam
        
        Args:
            exam_id: Exam identifier
            
        Returns:
            Dictionary with exam analytics
        """
        try:
            grades = self.firebase.get_grades_by_exam(exam_id)
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
            total_answers = len(grades)
            unique_students = len(set(g['student_id'] for g in grades))
            percentages = [g['percentage'] for g in grades if 'percentage' in g]
            average_percentage = sum(percentages) / len(percentages) if percentages else 0
            min_percentage = min(percentages) if percentages else 0
            max_percentage = max(percentages) if percentages else 0
            dist = {}
            for g in grades:
                p = g.get('percentage', 0)
                if p >= 90:
                    letter = 'A'
                elif p >= 80:
                    letter = 'B'
                elif p >= 70:
                    letter = 'C'
                elif p >= 60:
                    letter = 'D'
                else:
                    letter = 'F'
                dist[letter] = dist.get(letter, 0) + 1
            return {
                'exam_id': exam_id,
                'total_answers': total_answers,
                'unique_students': unique_students,
                'average_percentage': round(average_percentage, 2),
                'min_percentage': round(min_percentage, 2),
                'max_percentage': round(max_percentage, 2),
                'grade_distribution': dist
            }
        except Exception as e:
            logger.error(f"Failed to get exam analytics: {e}")
            return {}
    
    def export_grades(self, output_path: str, exam_id: str = None):
        """Export grades to JSON file"""
        try:
            if exam_id:
                grades = self.firebase.get_grades_by_exam(exam_id)
            else:
                grades = []
                for ex in self.firebase.list_exam_ids():
                    grades.extend(self.firebase.get_grades_by_exam(ex))
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(grades, f, indent=2, ensure_ascii=False)
            logger.info(f"Grades exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export grades: {e}")
            raise

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize agent
    agent = GradeStorageAgent()
    
    # Test storing a grade
    # grade_record = agent.store_grade(
    #     student_id="STU001",
    #     exam_id="EXAM001",
    #     question_id="Q001",
    #     answer_text="The answer is photosynthesis",
    #     score=8.5,
    #     max_score=10.0,
    #     percentage=85.0,
    #     feedback="Good understanding of the concept",
    #     metadata={"teacher": "Dr. Smith", "subject": "Biology"}
    # )
    # print(f"Grade stored with hash: {grade_record.current_hash}")
    
    # Verify chain integrity
    # verification = agent.verify_chain_integrity()
    # print(f"Chain integrity: {verification['is_valid']}")

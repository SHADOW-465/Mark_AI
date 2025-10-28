from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
import logging

from services.firebaseservice import FirebaseService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/exams", tags=["exams"])

class CreateExamRequest(BaseModel):
    exam_id: str = Field(..., description="Unique exam identifier")
    title: str
    description: Optional[str] = None
    subject: Optional[str] = None
    total_questions: Optional[int] = None
    max_score: Optional[float] = None
    rubric_id: Optional[str] = None

@router.post("/")
async def create_exam(payload: CreateExamRequest):
    try:
        fb = FirebaseService()
        exam_data = {
            'exam_id': payload.exam_id,
            'title': payload.title,
            'description': payload.description,
            'subject': payload.subject,
            'total_questions': payload.total_questions,
            'max_score': payload.max_score,
            'created_at': datetime.utcnow().isoformat()
        }
        fb.store_exam(exam_data)
        return { 'message': 'Exam created', 'exam': exam_data }
    except Exception as e:
        logger.error(f"Create exam failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))



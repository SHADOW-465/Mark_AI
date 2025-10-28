from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
import logging

from database.supabase_client import get_supabase_client

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
        client = get_supabase_client()
        exam_data = {
            'exam_id': payload.exam_id,
            'title': payload.title,
            'description': payload.description,
            'subject': payload.subject,
            'total_questions': payload.total_questions,
            'max_score': payload.max_score,
            'created_at': datetime.utcnow().isoformat()
        }
        created = client.create_exam(exam_data)
        # Optionally associate rubric by rubric_id via separate table or metadata
        return { 'message': 'Exam created', 'exam': created }
    except Exception as e:
        logger.error(f"Create exam failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))



from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from agents.grade_storage import GradeStorageAgent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/grades", tags=["grades"])

class OverrideGradeRequest(BaseModel):
    student_id: str
    exam_id: str
    question_id: str
    new_score: float
    new_feedback: str
    override_reason: str

@router.put("/{grade_id}/override")
async def override_grade(grade_id: str, payload: OverrideGradeRequest):
    try:
        storage = GradeStorageAgent()
        grades = storage.get_student_grades(payload.student_id, payload.exam_id)
        existing = next((g for g in grades if g.question_id == payload.question_id), None)
        if not existing:
            raise HTTPException(status_code=404, detail="Grade not found")
        new_grade = storage.store_grade(
            student_id=payload.student_id,
            exam_id=payload.exam_id,
            question_id=f"{payload.question_id}_OVERRIDE",
            answer_text=existing.answer_text,
            score=payload.new_score,
            max_score=existing.max_score,
            percentage=(payload.new_score / existing.max_score) * 100.0,
            feedback=payload.new_feedback,
            metadata={
                'original_grade_hash': existing.current_hash,
                'override_reason': payload.override_reason,
                'is_override': True
            }
        )
        return { 'message': 'Grade overridden', 'hash': new_grade.current_hash }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Override grade failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))



from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging

from agents.grade_storage import GradeStorageAgent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])

@router.get("/{exam_id}")
async def get_exam_analytics(exam_id: str):
    try:
        storage = GradeStorageAgent()
        analytics = storage.get_exam_analytics(exam_id)
        if not analytics:
            raise HTTPException(status_code=404, detail="Exam not found or no data")
        return analytics
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get analytics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))



from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import os
import uuid
import logging

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import MultiAgentOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/submissions", tags=["submissions"])

# Simple in-memory store; replace with DB as needed
SUBMISSIONS: Dict[str, Dict[str, Any]] = {}

UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

class SubmissionResponse(BaseModel):
    submission_id: str
    status: str
    message: str

class TransformsResponse(BaseModel):
    submission_id: str
    alignment: Optional[Dict[str, Any]]

def _get_orchestrator() -> MultiAgentOrchestrator:
    # Lazy import to avoid circulars with main
    from agents.orchestrator import MultiAgentOrchestrator
    config = {
        'yolo_model_path': os.getenv('YOLO_MODEL_PATH'),
        'confidence_threshold': float(os.getenv('CONFIDENCE_THRESHOLD', '0.5')),
        'google_credentials_path': os.getenv('GOOGLE_CREDENTIALS_PATH'),
        'trocr_model_name': os.getenv('TROCR_MODEL_NAME', 'microsoft/trocr-base-handwritten'),
        'languages': os.getenv('LANGUAGES', 'en,hi,ta').split(','),
        'google_gemini_api_key': os.getenv('GOOGLE_GEMINI_API_KEY'),
        'perplexity_api_key': os.getenv('PERPLEXITY_API_KEY'),
        'rubric_path': os.getenv('RUBRIC_PATH'),
        'supabase_url': os.getenv('SUPABASE_URL'),
        'supabase_key': os.getenv('SUPABASE_KEY'),
        'storage_dir': os.getenv('STORAGE_DIR', './grades')
    }
    return MultiAgentOrchestrator(config)

@router.post("/", response_model=SubmissionResponse)
async def create_submission(
    background_tasks: BackgroundTasks,
    sheet: UploadFile = File(...),
    template: UploadFile = File(...),
    question: Optional[str] = None,
    reference_answer: Optional[str] = None,
    student_id: Optional[str] = None,
    exam_id: Optional[str] = None
):
    if not sheet.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Sheet must be an image")
    if not template.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Template must be an image")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_id = f"SUB_{timestamp}_{uuid.uuid4().hex[:8]}"

    sheet_path = UPLOAD_DIR / f"{submission_id}_sheet_{sheet.filename}"
    template_path = UPLOAD_DIR / f"{submission_id}_template_{template.filename}"

    content_sheet = await sheet.read()
    with open(sheet_path, "wb") as f:
        f.write(content_sheet)
    content_template = await template.read()
    with open(template_path, "wb") as f:
        f.write(content_template)

    SUBMISSIONS[submission_id] = {
        'status': 'processing',
        'sheet_path': str(sheet_path),
        'template_path': str(template_path),
        'aligned_image_path': None,
        'alignment': None,
        'result': None,
        'created_at': datetime.now().isoformat()
    }

    background_tasks.add_task(_process_submission, submission_id, question, reference_answer, student_id, exam_id)

    return SubmissionResponse(submission_id=submission_id, status='processing', message='Submission received')

async def _process_submission(submission_id: str, question: Optional[str], reference_answer: Optional[str], student_id: Optional[str], exam_id: Optional[str]):
    try:
        data = SUBMISSIONS.get(submission_id)
        if not data:
            return
        orch = _get_orchestrator()

        # Run preprocessing with template alignment only to capture aligned image & transforms
        pre = orch.image_agent.preprocess_image(data['sheet_path'], template_path=data['template_path'])

        # Save aligned image
        aligned_out = PROCESSED_DIR / f"{submission_id}_aligned.jpg"
        import cv2
        if 'processed_image' in pre:
            cv2.imwrite(str(aligned_out), pre['processed_image'])
        data['aligned_image_path'] = str(aligned_out)
        data['alignment'] = pre.get('metadata', {}).get('alignment')

        # Proceed with full pipeline using orchestrator (no template param in orchestrator yet; alignment already applied above)
        result = await orch.process_answer_sheet(
            image_path=data['sheet_path'],
            question=question or "Evaluate the answers as per rubric.",
            reference_answer=reference_answer,
            student_id=student_id,
            exam_id=exam_id
        )
        data['result'] = orch.get_processing_summary(result)
        data['status'] = 'completed' if data['result'].get('success') else 'failed'
    except Exception as e:
        logger.error(f"Submission {submission_id} failed: {e}")
        if submission_id in SUBMISSIONS:
            SUBMISSIONS[submission_id]['status'] = 'failed'
            SUBMISSIONS[submission_id]['error'] = str(e)

@router.get("/{submission_id}/aligned")
async def get_aligned_image(submission_id: str):
    data = SUBMISSIONS.get(submission_id)
    if not data:
        raise HTTPException(status_code=404, detail="Submission not found")
    if not data.get('aligned_image_path') or not os.path.exists(data['aligned_image_path']):
        raise HTTPException(status_code=404, detail="Aligned image not available yet")
    return FileResponse(path=data['aligned_image_path'], filename=Path(data['aligned_image_path']).name, media_type='image/jpeg')

@router.get("/{submission_id}/transforms", response_model=TransformsResponse)
async def get_transforms(submission_id: str):
    data = SUBMISSIONS.get(submission_id)
    if not data:
        raise HTTPException(status_code=404, detail="Submission not found")
    return TransformsResponse(submission_id=submission_id, alignment=data.get('alignment'))

@router.get("/{submission_id}/grades")
async def get_submission_grades(submission_id: str):
    data = SUBMISSIONS.get(submission_id)
    if not data:
        raise HTTPException(status_code=404, detail="Submission not found")
    if data.get('status') != 'completed':
        return {"submission_id": submission_id, "status": data.get('status')}
    return {"submission_id": submission_id, "status": data.get('status'), "summary": data.get('result')}



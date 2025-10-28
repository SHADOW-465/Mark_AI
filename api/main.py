"""
FastAPI main application for EduGrade AI
Provides REST API endpoints for the multi-agent answer sheet evaluation system
"""

import os
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn
from pathlib import Path
import asyncio
import json
from datetime import datetime

# Import orchestrator and agents
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import MultiAgentOrchestrator, ProcessingState
from agents.grade_storage import GradeStorageAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import APIRouter

# Initialize FastAPI app
app = FastAPI(
    title="EduGrade AI API",
    description="Multi-Agentic Answer Sheet Evaluator System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from .endpoints import submissions as submissions_router
from .endpoints import exams as exams_router
from .endpoints import grades as grades_router
from .endpoints import analytics as analytics_router
from .endpoints import devdock as devdock_router

# Global variables
orchestrator = None
storage_agent = None

# Pydantic models
class ProcessingRequest(BaseModel):
    question: str
    reference_answer: Optional[str] = None
    student_id: Optional[str] = None
    exam_id: Optional[str] = None
    rubric_path: Optional[str] = None

class GradeOverrideRequest(BaseModel):
    student_id: str
    exam_id: str
    question_id: str
    new_score: float
    new_feedback: str
    override_reason: str

class BatchProcessingRequest(BaseModel):
    image_paths: List[str]
    question: str
    reference_answer: Optional[str] = None
    exam_id: Optional[str] = None

class ProcessingResponse(BaseModel):
    success: bool
    processing_id: str
    message: str
    results: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None

class GradeResponse(BaseModel):
    student_id: str
    exam_id: str
    question_id: str
    score: float
    max_score: float
    percentage: float
    feedback: str
    timestamp: str
    hash: str

class AnalyticsResponse(BaseModel):
    exam_id: str
    total_answers: int
    unique_students: int
    average_percentage: float
    min_percentage: float
    max_percentage: float
    grade_distribution: Dict[str, int]

# Dependency to get orchestrator
def get_orchestrator():
    global orchestrator
    if orchestrator is None:
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
        orchestrator = MultiAgentOrchestrator(config)
    return orchestrator

def get_storage_agent():
    global storage_agent
    if storage_agent is None:
        storage_agent = GradeStorageAgent(
            db_path=None,  # Deprecated for Supabase
            storage_dir=os.getenv('STORAGE_DIR', './grades')
        )
    return storage_agent

# Create upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Background task storage
processing_tasks = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    logger.info("Starting EduGrade AI API...")
    
    # Initialize orchestrator
    get_orchestrator()
    get_storage_agent()
    
    logger.info("EduGrade AI API started successfully")

    # Include routers
    app.include_router(submissions_router.router)
    app.include_router(exams_router.router)
    app.include_router(grades_router.router)
    app.include_router(analytics_router.router)
    app.include_router(devdock_router.router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "EduGrade AI API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "orchestrator": orchestrator is not None,
            "storage": storage_agent is not None
        }
    }

@app.post("/upload", response_model=ProcessingResponse)
async def upload_and_process(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    question: str = None,
    reference_answer: str = None,
    student_id: str = None,
    exam_id: str = None,
    rubric_path: str = None
):
    """
    Upload and process an answer sheet image
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = UPLOAD_DIR / filename
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Generate processing ID
        processing_id = f"PROC_{timestamp}_{hash(file.filename) % 10000}"
        
        # Start background processing
        background_tasks.add_task(
            process_answer_sheet_background,
            processing_id,
            str(file_path),
            question or "Please evaluate this answer sheet",
            reference_answer,
            student_id,
            exam_id,
            rubric_path
        )
        
        return ProcessingResponse(
            success=True,
            processing_id=processing_id,
            message="File uploaded and processing started",
            results={
                "file_path": str(file_path),
                "file_size": len(content),
                "content_type": file.content_type
            }
        )
        
    except Exception as e:
        logger.error(f"Error in upload endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_answer_sheet_background(
    processing_id: str,
    file_path: str,
    question: str,
    reference_answer: str,
    student_id: str,
    exam_id: str,
    rubric_path: str
):
    """Background task for processing answer sheets"""
    try:
        logger.info(f"Starting background processing for {processing_id}")
        
        # Get orchestrator
        orch = get_orchestrator()
        
        # Process the answer sheet
        result = await orch.process_answer_sheet(
            image_path=file_path,
            question=question,
            reference_answer=reference_answer,
            student_id=student_id,
            exam_id=exam_id
        )
        
        # Store result
        processing_tasks[processing_id] = {
            'status': 'completed',
            'result': result,
            'summary': orch.get_processing_summary(result),
            'completed_at': datetime.now().isoformat()
        }
        
        logger.info(f"Background processing completed for {processing_id}")
        
    except Exception as e:
        logger.error(f"Error in background processing {processing_id}: {e}")
        processing_tasks[processing_id] = {
            'status': 'failed',
            'error': str(e),
            'failed_at': datetime.now().isoformat()
        }

@app.get("/processing/{processing_id}")
async def get_processing_status(processing_id: str):
    """Get the status of a processing task"""
    if processing_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Processing task not found")
    
    task = processing_tasks[processing_id]
    
    if task['status'] == 'completed':
        return {
            "processing_id": processing_id,
            "status": "completed",
            "summary": task['summary'],
            "completed_at": task['completed_at']
        }
    elif task['status'] == 'failed':
        return {
            "processing_id": processing_id,
            "status": "failed",
            "error": task['error'],
            "failed_at": task['failed_at']
        }
    else:
        return {
            "processing_id": processing_id,
            "status": "processing",
            "message": "Task is still being processed"
        }

@app.get("/grades/{student_id}")
async def get_student_grades(student_id: str, exam_id: Optional[str] = None):
    """Get grades for a specific student"""
    try:
        storage = get_storage_agent()
        grades = storage.get_student_grades(student_id, exam_id)
        
        return {
            "student_id": student_id,
            "exam_id": exam_id,
            "grades": [
                {
                    "question_id": grade.question_id,
                    "score": grade.score,
                    "max_score": grade.max_score,
                    "percentage": grade.percentage,
                    "feedback": grade.feedback,
                    "timestamp": grade.timestamp,
                    "hash": grade.current_hash
                }
                for grade in grades
            ],
            "total_grades": len(grades)
        }
        
    except Exception as e:
        logger.error(f"Error getting student grades: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/grades/override")
async def override_grade(request: GradeOverrideRequest):
    """Override a grade with teacher correction"""
    try:
        storage = get_storage_agent()
        
        # Get existing grade
        grades = storage.get_student_grades(request.student_id, request.exam_id)
        existing_grade = next(
            (g for g in grades if g.question_id == request.question_id), 
            None
        )
        
        if not existing_grade:
            raise HTTPException(status_code=404, detail="Grade not found")
        
        # Store override as new grade record
        override_grade = storage.store_grade(
            student_id=request.student_id,
            exam_id=request.exam_id,
            question_id=f"{request.question_id}_OVERRIDE",
            answer_text=existing_grade.answer_text,
            score=request.new_score,
            max_score=existing_grade.max_score,
            percentage=(request.new_score / existing_grade.max_score) * 100,
            feedback=request.new_feedback,
            metadata={
                'original_grade_hash': existing_grade.current_hash,
                'override_reason': request.override_reason,
                'override_timestamp': datetime.now().isoformat(),
                'is_override': True
            }
        )
        
        return {
            "message": "Grade overridden successfully",
            "original_grade": {
                "score": existing_grade.score,
                "percentage": existing_grade.percentage,
                "hash": existing_grade.current_hash
            },
            "new_grade": {
                "score": override_grade.score,
                "percentage": override_grade.percentage,
                "hash": override_grade.current_hash
            }
        }
        
    except Exception as e:
        logger.error(f"Error overriding grade: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/{exam_id}")
async def get_exam_analytics(exam_id: str):
    """Get analytics for a specific exam"""
    try:
        storage = get_storage_agent()
        analytics = storage.get_exam_analytics(exam_id)
        
        if not analytics:
            raise HTTPException(status_code=404, detail="Exam not found")
        
        return AnalyticsResponse(**analytics)
        
    except Exception as e:
        logger.error(f"Error getting exam analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics")
async def get_all_analytics():
    """Get analytics for all exams"""
    try:
        storage = get_storage_agent()
        
        # Get all unique exam IDs from Supabase
        result = storage.supabase_client.client.table('grades').select('exam_id').execute()
        exam_ids = list(set(grade['exam_id'] for grade in result.data or []))
        
        analytics = {}
        for exam_id in exam_ids:
            analytics[exam_id] = storage.get_exam_analytics(exam_id)
        
        return {
            "exams": analytics,
            "total_exams": len(exam_ids)
        }
        
    except Exception as e:
        logger.error(f"Error getting all analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/verify-integrity")
async def verify_grade_integrity():
    """Verify the integrity of the grade chain"""
    try:
        storage = get_storage_agent()
        verification = storage.verify_chain_integrity()
        
        return {
            "chain_integrity": verification['is_valid'],
            "total_records": verification['total_records'],
            "invalid_records": verification['invalid_records'],
            "chain_breaks": verification['chain_breaks']
        }
        
    except Exception as e:
        logger.error(f"Error verifying integrity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export/grades")
async def export_grades(exam_id: Optional[str] = None, format: str = "json"):
    """Export grades to file"""
    try:
        storage = get_storage_agent()
        
        # Generate export filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if exam_id:
            filename = f"grades_{exam_id}_{timestamp}.{format}"
        else:
            filename = f"grades_all_{timestamp}.{format}"
        
        export_path = f"exports/{filename}"
        Path("exports").mkdir(exist_ok=True)
        
        # Export grades
        storage.export_grades(export_path, exam_id)
        
        return FileResponse(
            path=export_path,
            filename=filename,
            media_type='application/json' if format == 'json' else 'text/csv'
        )
        
    except Exception as e:
        logger.error(f"Error exporting grades: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/processing/{processing_id}")
async def delete_processing_task(processing_id: str):
    """Delete a processing task"""
    if processing_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Processing task not found")
    
    del processing_tasks[processing_id]
    return {"message": "Processing task deleted successfully"}

@app.get("/processing")
async def list_processing_tasks():
    """List all processing tasks"""
    return {
        "tasks": [
            {
                "processing_id": pid,
                "status": task['status'],
                "created_at": task.get('created_at', 'unknown')
            }
            for pid, task in processing_tasks.items()
        ],
        "total_tasks": len(processing_tasks)
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

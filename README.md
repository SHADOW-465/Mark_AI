# EduGrade AI - Multi-Agentic Answer Sheet Evaluator

An automated exam grading system for handwritten answer sheets using multi-agent AI architecture.

## Features

- **Image Preprocessing Agent**: OpenCV + YOLOv8 for answer sheet detection and segmentation
- **OCR Extraction Agent**: Google Vision API + TrOCR for multi-language text extraction
- **Evaluation Agent**: Google Gemini + Perplexity Sonar API for semantic grading
- **Grade Storage Agent**: SHA-256 cryptographic hashing for tamper-proof storage
- **Teacher Dashboard**: Streamlit/Gradio interface for review and override capabilities

## Project Structure

```
Edugrade-AI/
├── agents/                 # Multi-agent system components
│   ├── image_preprocessing.py
│   ├── ocr_extraction.py
│   ├── evaluation.py
│   └── grade_storage.py
├── api/                   # FastAPI backend
│   ├── main.py
│   ├── endpoints/
│   └── models/
├── dashboard/             # Teacher dashboard
│   ├── streamlit_app.py
│   └── gradio_app.py
├── database/              # Database models and migrations
│   ├── models.py
│   └── migrations/
├── utils/                 # Utility functions
│   ├── image_utils.py
│   ├── text_utils.py
│   └── crypto_utils.py
├── tests/                 # Unit tests
├── docker/                # Docker configuration
├── requirements.txt
└── README.md
```

## Setup Instructions

### Prerequisites

- Python 3.9+
- Docker (optional)
- Google Cloud Vision API key
- Google Gemini API key
- Perplexity API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Edugrade-AI
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp env.example .env
# Edit .env with your API keys:
# - GOOGLE_GEMINI_API_KEY=your_gemini_api_key
# - GOOGLE_VISION_API_KEY=your_vision_api_key
# - PERPLEXITY_API_KEY=your_perplexity_api_key
# - SUPABASE_URL=your_supabase_url
# - SUPABASE_KEY=your_supabase_key
# - YOLO_MODEL_PATH=optional_yolo_path
# - TROCR_MODEL_NAME=microsoft/trocr-base-handwritten
# - DEVDOCK_API_KEY=optional_devdock_key (stub)
```

Windows (PowerShell) example:
```powershell
Copy-Item env.example .env
# then open .env in your editor and set values
```

5. Initialize database:
```bash
python -m database.init_db
```

6. Run the application:
```bash
# Start FastAPI backend
uvicorn api.main:app --reload

# Start teacher dashboard (in another terminal)
streamlit run dashboard/streamlit_app.py
```

### Docker Setup

```bash
# Build and run with Docker Compose
docker-compose up --build
```

## Running Locally (Dev)

1. Activate virtual environment
```bash
python -m venv venv
.\n+venv\Scripts\activate  # Windows PowerShell
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Ensure .env is configured (see above). At minimum set `SUPABASE_URL` and `SUPABASE_KEY`. If Gemini/Perplexity are not configured, the app will run with limited capabilities.

4. Start FastAPI backend
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

5. (Optional) Start the Streamlit demo dashboard
```bash
streamlit run dashboard/streamlit_app.py
```

6. Open API docs
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints (v1)

- `POST /exams/` - Create exam (answer key/rubric association optional)
- `POST /submissions/` - Upload student sheet and template; starts processing
- `GET /submissions/{id}/aligned` - Download aligned sheet image
- `GET /submissions/{id}/transforms` - Alignment transforms + accuracy
- `GET /submissions/{id}/grades` - Per-submission grading summary
- `PUT /grades/{grade_id}/override` - Teacher override score/feedback
- `GET /analytics/{exam_id}` - Aggregated analytics
- `POST /devdock/verify` - Blockchain credential verification (stub)
- `GET /health` - Status

### Sample Requests (curl)

- Health check
```bash
curl http://localhost:8000/health
```

- Create exam
```bash
curl -X POST http://localhost:8000/exams/ \
     -H "Content-Type: application/json" \
     -d '{
           "exam_id": "EXAM_001",
           "title": "Science Midterm",
           "description": "Grade 8",
           "subject": "Science",
           "total_questions": 10,
           "max_score": 100
         }'
```

- Submit answer sheet with template (alignment + pipeline)
```bash
curl -X POST http://localhost:8000/submissions/ \
     -F "sheet=@samplesheets/student_sheet.jpg" \
     -F "template=@samplesheets/template.jpg" \
     -F "question=Explain photosynthesis" \
     -F "reference_answer=Photosynthesis converts light energy..." \
     -F "student_id=STU_001" \
     -F "exam_id=EXAM_001"
```

- Fetch aligned image
```bash
curl -L http://localhost:8000/submissions/SUB_xxx/aligned --output aligned.jpg
```

- Fetch transforms (homography, inlier stats, accuracy)
```bash
curl http://localhost:8000/submissions/SUB_xxx/transforms
```

- Fetch submission grades summary
```bash
curl http://localhost:8000/submissions/SUB_xxx/grades
```

- Override a grade
```bash
curl -X PUT http://localhost:8000/grades/GRADE_xxx/override \
     -H "Content-Type: application/json" \
     -d '{
           "student_id": "STU_001",
           "exam_id": "EXAM_001",
           "question_id": "Q001",
           "new_score": 8.0,
           "new_feedback": "Accepted alternate phrasing.",
           "override_reason": "Manual review"
         }'
```

- Get analytics for an exam
```bash
curl http://localhost:8000/analytics/EXAM_001
```

- DevDock verification (stub)
```bash
curl -X POST http://localhost:8000/devdock/verify \
     -H "Content-Type: application/json" \
     -d '{ "payload": { "grade_hash": "..." } }'
```

## Configuration

Edit `config/settings.py` to configure:
- API endpoints
- Model parameters
- Database settings
- File storage paths

### Environment Variables

- GOOGLE_GEMINI_API_KEY: Gemini key (required for AI grading)
- GOOGLE_VISION_API_KEY or GOOGLE_CREDENTIALS_PATH: Vision OCR
- PERPLEXITY_API_KEY: Fact-checking
- SUPABASE_URL, SUPABASE_KEY: Database storage (required)
- YOLO_MODEL_PATH: Optional YOLOv8 weights for detection
- TROCR_MODEL_NAME: Defaults to microsoft/trocr-base-handwritten
- DEVDOCK_API_KEY: Optional (stub)
- STORAGE_DIR, UPLOAD_DIR, PROCESSED_DIR, EXPORTS_DIR: Optional paths

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agents --cov=api --cov=dashboard
```

## Housekeeping / Cleanup

- The folders `api/models` and `utils` are currently empty and not used. You can remove them safely or keep for future expansion.
- Logs, uploads, processed outputs, and exports are written to their respective directories. Clean them periodically if `CLEANUP_OLD_FILES` is disabled.

## License

MIT License

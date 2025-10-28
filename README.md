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

This project is fully containerized using Docker and Docker Compose, allowing for a consistent and isolated development environment.

#### Prerequisites

- **Docker Desktop**: Install Docker Desktop for your operating system (Windows, macOS, or Linux). You can download it from the [official Docker website](https://www.docker.com/products/docker-desktop/).

#### Initial Setup

1.  **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd Edugrade-AI
    ```

2.  **Service Account Key**:
    - Place your `serviceAccountKey.json` file in the `config/` directory.
    - **Important**: This file is included in the `.gitignore` to prevent it from being committed to your repository.

3.  **Environment Variables**:
    - Create a `.env` file by copying the example:
      ```bash
      cp env.example .env
      ```
    - Open the `.env` file and fill in your API keys and other configuration details.
    - Set the `GOOGLE_APPLICATION_CREDENTIALS` variable to point to your service account key:
      ```
      GOOGLE_APPLICATION_CREDENTIALS=/app/config/serviceAccountKey.json
      ```

#### Running the Application with Docker

1.  **Build and Run the Containers**:
    ```bash
    docker-compose up --build
    ```
    This command will:
    - Build the Docker images for the API and dashboard services.
    - Start all the services defined in the `docker-compose.yml` file.
    - Mount the necessary volumes for data persistence.

2.  **Accessing the Services**:
    - **API**: The FastAPI backend will be available at `http://localhost:8000`.
    - **Streamlit Dashboard**: The Streamlit dashboard will be running at `http://localhost:8501`.
    - **Gradio Dashboard**: The Gradio dashboard will be accessible at `http://localhost:7860`.

3.  **Stopping the Application**:
    - To stop the running containers, press `Ctrl + C` in the terminal where `docker-compose` is running.
    - To remove the containers and associated volumes, you can run:
      ```bash
      docker-compose down -v
      ```

#### Production Environment

For a production deployment, it is recommended to:

-   Use a managed database service instead of running a database in a container.
-   Store the `serviceAccountKey.json` file securely, for example, using a secret management tool like HashiCorp Vault or AWS Secrets Manager.
-   Configure the `docker-compose.yml` file with appropriate resource limits and restart policies.
-   Use a reverse proxy like Nginx or Traefik to manage incoming traffic and provide SSL termination.

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

### End-to-End Quick Start

1) Create an exam via `POST /exams/`.
2) Upload a student sheet with the matching template via `POST /submissions/`.
3) Poll `GET /submissions/{id}/grades` until status is `completed`.
4) Fetch aligned image and transforms if needed.
5) View analytics with `GET /analytics/{exam_id}`. Use `PUT /grades/{grade_id}/override` for teacher overrides.

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

### DeepSeek-OCR (vLLM) Setup

- Requires `vllm` and typically a CUDA-capable GPU for best performance.
- Environment variables:
  - `DEEPSEEK_OCR_MODEL` (default: `deepseek-ai/DeepSeek-OCR`)
- Notes:
  - Ensure compatible CUDA/PyTorch drivers are installed.
  - If running on CPU-only, performance may be slow or unsupported depending on your environment.

### Firebase Setup (Firestore)

- Install `firebase-admin` (already in `requirements.txt`).
- Create a Firebase service account and download the JSON credentials.
- Set `FIREBASE_CREDENTIALS_PATH` in `.env` to the JSON file path.
- This app writes grade documents to the `grades` collection and alignment metadata to the `alignments` collection (best-effort; Supabase remains the system of record).

### Supabase Setup (System of Record)

- Provide `SUPABASE_URL` and `SUPABASE_KEY` in `.env`.
- Ensure the following tables exist (basic columns as per `database/models.py` and storage layer): `grades`, `students`, `exams`, `processing_jobs`, `rubrics`.
- The app will read/write grades and compute analytics from Supabase.

### DevDock (Stub) Setup

- Set `DEVDOCK_API_KEY` if available.
- The current integration is a stub that records grade hash verification calls; replace `services/devdockservice.py` with your real SDK logic when available.

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

## Troubleshooting

- vLLM/DeepSeek model fails to load:
  - Check GPU availability and CUDA drivers; verify PyTorch + vllm compatibility.
  - Reduce memory use (disable large caches) or switch to a smaller model if supported.
- Firebase writes are skipped:
  - Ensure `FIREBASE_CREDENTIALS_PATH` points to a valid JSON; check IAM permissions for Firestore.
- Supabase errors:
  - Verify URL/key; confirm tables and RLS policies allow the operations.

## License

MIT License

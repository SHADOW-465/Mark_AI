# EduGrade AI: Final Project Overview – Updated for Hackathon, Development & Reference (Oct 28, 2025)

## Project Vision
EduGrade AI is a **multi-agent automated grading platform** for handwritten answer sheets with full-stack integration of computer vision, AI LLMs (Google Gemini), advanced OCR (DeepSeek-OCR, TrOCR), and blockchain credentialing (DevDock). It accelerates grading, enhances consistency, supports teacher oversight, and creates verifiable educational records.

## Problem & Solution Overview
### Key Challenges
- **Manual grading** consumes 60–70% of teachers' time and introduces error, fatigue and bias.
- **Scanned answer sheet alignment**: Student papers suffer from deskewing, perspective distortion (trapezoidal effects), scaling, translation shifts, inconsistent padding, deformations, noise artifacts, and scan variation. 
- **Auto-evaluation of structured answers**: Handwriting variability, ambiguous MCQ markings, spelling/synonym flexibility, answer positioning errors, overwriting/strikethroughs, and faint/noisy writing all remain challenging.

### Proposed Solution
Automate the entire pipeline: ingest scanned answer sheet, perform template-based geometric registration for robust alignment, crop answer regions, OCR handwritten content, evaluate using rubrics/L(Q)LM, and return interpretable results. Store grades on blockchain for tamper-proof records, and provide actionable feedback to improve student learning.

## Detailed Requirements Analysis
### 1. **Automatic Alignment & Registration**
#### Original Design
- PreprocessingAgent uses OpenCV: deskewing (±15°), denoising, binarization, contrast enhancement.

#### Required Expansion / GAP Closure
- *Template-based alignment*: Integrate geometric registration matching uploaded student sheets to the original question paper template. This addresses not just deskewing, but also perspective distortion, scaling, translation, padding/margins, and deformation.
- *Transformation parameters* and *alignment accuracy/confidence score* must be computed and exposed via the API after every alignment operation.
- *Aligned image output must be exposed via API* for audit, debugging, and analytics.
- **Input Expansion**: The original question paper template is required as an explicit API input for the alignment agent.

### 2. **Structured Answer Extraction & Evaluation**
#### Existing Codebase
- SegmentationAgent uses YOLOv8 for answer box detection and cropping.
- OCRAgent (ensemble: TrOCR + DeepSeek-OCR + Gemini Vision) for English, Hindi, Tamil handwriting, mathematical notation, and diagrams.
- GradingAgent uses Gemini LLM for rubric-based semantic scoring, handling MCQ, fill-in-the-blank, and one-word answers. Flexible enough for minor spelling/synonym issues.

#### Required Expansion / GAP Closure
- *Answer positioning errors*: Add logic to filter, flag, or realign answers outside designated boxes or overlapping with other regions.
- *Overwriting/strikethroughs*: Enhance processing to flag ambiguous/overwritten regions for manual review or auto-handling.
- *Ambiguous MCQ markings*: MCQ box classifier/logic to address ambiguous ticks, erasures, or symbols, flagging for manual override where necessary.
- *Faint/noisy writing*: Detection/enhancement for low-contrast, noise-interference regions to improve OCR reliability.
- *Explainable evaluation*: Ensure every score and correction is returned with the reasoning and reference to the marking scheme, especially in flagged cases.

### 3. **Interpretable Outputs**
#### Existing Capabilities
- GradingAgent stores scores, feedback, and reasoning in Firebase/DevDock for audit and export.

#### Required Expansion / GAP Closure
- Return *aligned student sheet*, *transformation parameters*, and *alignment accuracy score* via dedicated API endpoints.
- Per-question auto-evaluation results with source region references, handling flagged ambiguous or noisy answers.
- Real-time preview of alignment corrections for demo/debug analytics dashboard.

### 4. **Inputs & Constraints**
- *Scanned answer sheet*, *original question paper template*, and *reference answer script* must be accepted as discrete API inputs.
- Preserve *all handwriting* with high fidelity (>300 DPI, JPEG/PNG); processing should never downgrades resolution or discard student content.
- Handles multi-page sheets and damage recovery (torn/faded sheets) as bonus features.
- Operates without specialized hardware; runs on Docker/cloud, supports rapid deployment.

## Expanded Technical Solution & Architecture
- **Image Processing**: OpenCV (Preprocessing Agent), with new geometric registration for template alignment, perspective, scale, and translation correction.
- **Document Segmentation**: YOLOv8 model for answer region detection, fallback on grid segmentation.
- **OCR Extraction**: Ensemble (DeepSeek-OCR, TrOCR, Gemini Vision) for accuracy in variable scripts, faint/noisy writing, and diagrams.
- **AI Grading**: Gemini LLM, rubric-based partial marking, explainable feedback, integrated Perplexity real-time fact-checking.
- **Credential Storage**: Immutable grades on DevDock blockchain; hashes stored in Firebase. Verification APIs and badges provided.
- **Teacher Dashboard & Analytics**: Streamlit (demo), React/Next.js (production); live progress, real-time previews, question-wise analysis, ambiguity flags, error logs, override tools.

## Deliverables
- Working prototype/demo, source code, agent/module architecture documentation
- Dataset test results; technical approach summary
- Draft research paper if innovations included
- Demo video (3–5 min, bonus)

## API Endpoints (Expanded)
- POST `/exams/` Create exam (answer key, rubric, template)
- POST `/submissions/` Upload answer sheet
- GET `/submissions/{id}/aligned` Get aligned, registered sheet image
- GET `/submissions/{id}/transforms` Get transformation param & accuracy
- GET `/submissions/{id}/grades` Get grading results (per Q)
- PUT `/grades/{grade_id}/override` Teacher override score/feedback
- GET `/analytics/{exam_id}` Aggregated analytics
- POST `/devdock/verify` Blockchain credential verification
- GET `/health` Status

## Bonus Features & Roadmap
- Multi-page answer sheet support
- Damage recovery (tears/fading)
- Handwriting readability enhancement
- Automated validation flagging uncertain cases
- Explainable evaluation audit trail

## Sponsor Technology
- **Google Gemini**: Core grading, feedback, multimodal OCR
- **Perplexity**: Factual verification
- **DevDock**: Blockchain credentials
- **Firebase**: Database, authentication

## Key Takeaways for Development, Documentation, and Reference
- *Template-based geometric registration is required for alignment (not just deskewing).*
- *All outputs—aligned sheet, transformation parameters, accuracy—must be exposed via API for audit/analytics.*
- *Expanded logic needed for ambiguous cases (MCQ marking, positioning, overwrites, noise).* 
- *Preserve all handwriting with no loss during image processing.*
- *Explainability and audit trails built-in at every stage.*
- *Sponsor requirements fully integrated.*

---
For continuous updates, use this file as the living reference for requirements, development tasks, API exposure, and deliverables.

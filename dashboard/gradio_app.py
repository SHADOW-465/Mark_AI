"""
Teacher Dashboard for EduGrade AI - Gradio Version
Alternative web interface using Gradio for teachers to review and manage grades
"""

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime
import os
from typing import Dict, List, Any, Tuple
import base64
from io import BytesIO

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

class EduGradeGradioApp:
    """Gradio-based dashboard for EduGrade AI"""
    
    def __init__(self):
        self.api_base_url = API_BASE_URL
        self.current_data = {
            'students': [],
            'exams': [],
            'grades': [],
            'processing_jobs': []
        }
    
    def make_api_request(self, endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
        """Make API request to backend"""
        try:
            url = f"{self.api_base_url}{endpoint}"
            
            if method == "GET":
                response = requests.get(url)
            elif method == "POST":
                response = requests.post(url, json=data)
            elif method == "DELETE":
                response = requests.delete(url)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"API Error: {str(e)}")
            return {}
        except Exception as e:
            print(f"Error: {str(e)}")
            return {}
    
    def get_system_status(self) -> Tuple[str, str]:
        """Get system status"""
        health = self.make_api_request("/health")
        if health.get("status") == "healthy":
            return "🟢 Online", "System is running normally"
        else:
            return "🔴 Offline", "System is experiencing issues"
    
    def get_processing_jobs(self) -> str:
        """Get processing jobs status"""
        jobs = self.make_api_request("/processing")
        if jobs and jobs.get("tasks"):
            jobs_df = pd.DataFrame(jobs["tasks"])
            return jobs_df.to_string(index=False)
        else:
            return "No processing jobs found."
    
    def get_student_grades(self, student_id: str, exam_id: str) -> str:
        """Get grades for a specific student"""
        if not student_id:
            return "Please enter a student ID."
        
        grades = self.make_api_request(f"/grades/{student_id}")
        if grades and grades.get("grades"):
            grades_df = pd.DataFrame(grades["grades"])
            if exam_id:
                grades_df = grades_df[grades_df.get('exam_id', '') == exam_id]
            return grades_df.to_string(index=False)
        else:
            return "No grades found for the selected student."
    
    def get_exam_analytics(self, exam_id: str) -> Tuple[str, str]:
        """Get analytics for an exam"""
        if not exam_id:
            return "Please enter an exam ID.", ""
        
        analytics = self.make_api_request(f"/analytics/{exam_id}")
        if analytics:
            # Create summary text
            summary = f"""
Exam ID: {analytics.get('exam_id', 'N/A')}
Total Answers: {analytics.get('total_answers', 0)}
Unique Students: {analytics.get('unique_students', 0)}
Average Score: {analytics.get('average_percentage', 0):.1f}%
Min Score: {analytics.get('min_percentage', 0):.1f}%
Max Score: {analytics.get('max_percentage', 0):.1f}%
            """
            
            # Create grade distribution chart
            grade_dist = analytics.get("grade_distribution", {})
            if grade_dist:
                fig = px.bar(
                    x=list(grade_dist.keys()),
                    y=list(grade_dist.values()),
                    title="Grade Distribution",
                    labels={"x": "Grade", "y": "Count"}
                )
                chart_html = fig.to_html(include_plotlyjs='cdn')
            else:
                chart_html = "<p>No grade distribution data available.</p>"
            
            return summary.strip(), chart_html
        else:
            return "Failed to load analytics data.", ""
    
    def process_answer_sheet(self, image, question: str, reference_answer: str,
                           student_id: str, exam_id: str) -> str:
        """Process uploaded answer sheet"""
        if image is None:
            return "Please upload an image."
        
        if not all([question, student_id, exam_id]):
            return "Please fill in all required fields."
        
        try:
            # In a real implementation, this would upload the file to the API
            # For now, return a mock response
            return f"""
Answer sheet processing started!
Student ID: {student_id}
Exam ID: {exam_id}
Question: {question[:50]}...
Status: Processing in background
            """
        except Exception as e:
            return f"Error processing answer sheet: {str(e)}"
    
    def override_grade(self, student_id: str, exam_id: str, question_id: str,
                      new_score: float, new_feedback: str, override_reason: str) -> str:
        """Override a grade"""
        if not all([student_id, exam_id, question_id, new_feedback, override_reason]):
            return "Please fill in all fields."
        
        try:
            data = {
                "student_id": student_id,
                "exam_id": exam_id,
                "question_id": question_id,
                "new_score": new_score,
                "new_feedback": new_feedback,
                "override_reason": override_reason
            }
            
            result = self.make_api_request("/grades/override", "POST", data)
            
            if result:
                return "Grade overridden successfully!"
            else:
                return "Failed to override grade."
                
        except Exception as e:
            return f"Error overriding grade: {str(e)}"
    
    def verify_integrity(self) -> str:
        """Verify grade chain integrity"""
        try:
            result = self.make_api_request("/verify-integrity")
            
            if result.get("chain_integrity"):
                return "✅ Grade chain integrity verified! All records are valid."
            else:
                invalid_records = result.get("invalid_records", [])
                chain_breaks = result.get("chain_breaks", [])
                return f"""
❌ Grade chain integrity check failed!
Invalid records: {len(invalid_records)}
Chain breaks: {len(chain_breaks)}
                """
                
        except Exception as e:
            return f"Error verifying integrity: {str(e)}"
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        with gr.Blocks(title="EduGrade AI - Teacher Dashboard", theme=gr.themes.Soft()) as app:
            gr.Markdown("# 📚 EduGrade AI - Teacher Dashboard")
            gr.Markdown("Multi-Agentic Answer Sheet Evaluator System")
            
            # System Status
            with gr.Row():
                with gr.Column():
                    status_text = gr.Textbox(
                        label="System Status",
                        value="Checking...",
                        interactive=False
                    )
                    status_desc = gr.Textbox(
                        label="Status Description",
                        value="Checking...",
                        interactive=False
                    )
                    refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary")
            
            # Main tabs
            with gr.Tabs():
                # Overview Tab
                with gr.Tab("📊 Overview"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Processing Jobs")
                            jobs_text = gr.Textbox(
                                label="Current Processing Jobs",
                                lines=10,
                                interactive=False
                            )
                            refresh_jobs_btn = gr.Button("🔄 Refresh Jobs")
                        
                        with gr.Column():
                            gr.Markdown("### System Health")
                            health_text = gr.Textbox(
                                label="Health Check",
                                value="Checking...",
                                interactive=False
                            )
                
                # Grade Management Tab
                with gr.Tab("📝 Grade Management"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Search Grades")
                            student_id_input = gr.Textbox(
                                label="Student ID",
                                placeholder="Enter student ID..."
                            )
                            exam_id_input = gr.Textbox(
                                label="Exam ID (Optional)",
                                placeholder="Enter exam ID..."
                            )
                            search_btn = gr.Button("🔍 Search Grades")
                            
                            grades_text = gr.Textbox(
                                label="Student Grades",
                                lines=15,
                                interactive=False
                            )
                        
                        with gr.Column():
                            gr.Markdown("### Grade Override")
                            override_student_id = gr.Textbox(
                                label="Student ID",
                                placeholder="Enter student ID..."
                            )
                            override_exam_id = gr.Textbox(
                                label="Exam ID",
                                placeholder="Enter exam ID..."
                            )
                            override_question_id = gr.Textbox(
                                label="Question ID",
                                placeholder="Enter question ID..."
                            )
                            override_score = gr.Number(
                                label="New Score",
                                value=0.0,
                                minimum=0.0,
                                maximum=100.0
                            )
                            override_feedback = gr.Textbox(
                                label="New Feedback",
                                placeholder="Enter new feedback...",
                                lines=3
                            )
                            override_reason = gr.Textbox(
                                label="Override Reason",
                                placeholder="Enter reason for override..."
                            )
                            override_btn = gr.Button("💾 Override Grade", variant="primary")
                            
                            override_result = gr.Textbox(
                                label="Override Result",
                                interactive=False
                            )
                
                # Processing Tab
                with gr.Tab("🔄 Processing"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Upload Answer Sheet")
                            image_input = gr.Image(
                                label="Answer Sheet Image",
                                type="pil"
                            )
                            question_input = gr.Textbox(
                                label="Question",
                                placeholder="Enter the question text...",
                                lines=3
                            )
                            reference_input = gr.Textbox(
                                label="Reference Answer (Optional)",
                                placeholder="Enter reference answer...",
                                lines=3
                            )
                            process_student_id = gr.Textbox(
                                label="Student ID",
                                placeholder="Enter student ID..."
                            )
                            process_exam_id = gr.Textbox(
                                label="Exam ID",
                                placeholder="Enter exam ID..."
                            )
                            process_btn = gr.Button("🚀 Process Answer Sheet", variant="primary")
                            
                            process_result = gr.Textbox(
                                label="Processing Result",
                                lines=5,
                                interactive=False
                            )
                
                # Analytics Tab
                with gr.Tab("📈 Analytics"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Exam Analytics")
                            analytics_exam_id = gr.Textbox(
                                label="Exam ID",
                                placeholder="Enter exam ID for analytics..."
                            )
                            analytics_btn = gr.Button("📊 Generate Analytics")
                            
                            analytics_text = gr.Textbox(
                                label="Analytics Summary",
                                lines=10,
                                interactive=False
                            )
                        
                        with gr.Column():
                            gr.Markdown("### Grade Distribution Chart")
                            analytics_chart = gr.HTML(
                                value="<p>Select an exam to view analytics chart.</p>"
                            )
                
                # Settings Tab
                with gr.Tab("⚙️ Settings"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### System Settings")
                            api_url_input = gr.Textbox(
                                label="API Base URL",
                                value=self.api_base_url
                            )
                            test_connection_btn = gr.Button("🔍 Test Connection")
                            
                            gr.Markdown("### System Actions")
                            verify_btn = gr.Button("🔍 Verify Grade Integrity")
                            export_btn = gr.Button("📤 Export Grades")
                            
                            settings_result = gr.Textbox(
                                label="Settings Result",
                                lines=5,
                                interactive=False
                            )
            
            # Event handlers
            def refresh_status():
                status, desc = self.get_system_status()
                return status, desc, status
            
            def refresh_jobs():
                jobs = self.get_processing_jobs()
                return jobs
            
            def search_grades(student_id, exam_id):
                return self.get_student_grades(student_id, exam_id)
            
            def get_analytics(exam_id):
                summary, chart = self.get_exam_analytics(exam_id)
                return summary, chart
            
            def process_sheet(image, question, reference, student_id, exam_id):
                return self.process_answer_sheet(image, question, reference, student_id, exam_id)
            
            def override_grade_func(student_id, exam_id, question_id, score, feedback, reason):
                return self.override_grade(student_id, exam_id, question_id, score, feedback, reason)
            
            def verify_integrity_func():
                return self.verify_integrity()
            
            def test_connection(api_url):
                # Update API URL
                self.api_base_url = api_url
                status, desc = self.get_system_status()
                return f"API URL updated to: {api_url}\nStatus: {status}\nDescription: {desc}"
            
            # Connect events
            refresh_btn.click(
                fn=refresh_status,
                outputs=[status_text, status_desc, health_text]
            )
            
            refresh_jobs_btn.click(
                fn=refresh_jobs,
                outputs=[jobs_text]
            )
            
            search_btn.click(
                fn=search_grades,
                inputs=[student_id_input, exam_id_input],
                outputs=[grades_text]
            )
            
            analytics_btn.click(
                fn=get_analytics,
                inputs=[analytics_exam_id],
                outputs=[analytics_text, analytics_chart]
            )
            
            process_btn.click(
                fn=process_sheet,
                inputs=[image_input, question_input, reference_input, process_student_id, process_exam_id],
                outputs=[process_result]
            )
            
            override_btn.click(
                fn=override_grade_func,
                inputs=[override_student_id, override_exam_id, override_question_id, 
                       override_score, override_feedback, override_reason],
                outputs=[override_result]
            )
            
            verify_btn.click(
                fn=verify_integrity_func,
                outputs=[settings_result]
            )
            
            test_connection_btn.click(
                fn=test_connection,
                inputs=[api_url_input],
                outputs=[settings_result]
            )
            
            # Initialize with current status
            app.load(
                fn=refresh_status,
                outputs=[status_text, status_desc, health_text]
            )
        
        return app
    
    def launch(self, share: bool = False, server_name: str = "0.0.0.0", server_port: int = 7860):
        """Launch the Gradio app"""
        app = self.create_interface()
        app.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            show_error=True
        )

# Main execution
if __name__ == "__main__":
    app = EduGradeGradioApp()
    app.launch()

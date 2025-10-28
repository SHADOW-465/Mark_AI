"""
Teacher Dashboard for EduGrade AI
Streamlit-based web interface for teachers to review and manage grades
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import os
from typing import Dict, List, Any
import base64
from io import BytesIO

# Configure page
st.set_page_config(
    page_title="EduGrade AI - Teacher Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

class EduGradeDashboard:
    """Main dashboard class"""
    
    def __init__(self):
        self.api_base_url = API_BASE_URL
        self.session_state = st.session_state
        
        # Initialize session state
        if 'selected_student' not in self.session_state:
            self.session_state.selected_student = None
        if 'selected_exam' not in self.session_state:
            self.session_state.selected_exam = None
        if 'processing_jobs' not in self.session_state:
            self.session_state.processing_jobs = []
    
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
            st.error(f"API Error: {str(e)}")
            return {}
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return {}
    
    def run(self):
        """Run the main dashboard"""
        # Header
        st.title("📚 EduGrade AI - Teacher Dashboard")
        st.markdown("Multi-Agentic Answer Sheet Evaluator System")
        
        # Sidebar
        self.render_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", "📝 Grade Management", "🔄 Processing", "📈 Analytics", "⚙️ Settings"
        ])
        
        with tab1:
            self.render_overview()
        
        with tab2:
            self.render_grade_management()
        
        with tab3:
            self.render_processing()
        
        with tab4:
            self.render_analytics()
        
        with tab5:
            self.render_settings()
    
    def render_sidebar(self):
        """Render sidebar with navigation and filters"""
        st.sidebar.title("🎛️ Controls")
        
        # Health check
        health = self.make_api_request("/health")
        if health.get("status") == "healthy":
            st.sidebar.success("✅ System Online")
        else:
            st.sidebar.error("❌ System Offline")
        
        # Student selection
        st.sidebar.subheader("👨‍🎓 Student Selection")
        students = self.get_students()
        if students:
            student_options = [f"{s['student_id']} - {s['name']}" for s in students]
            selected_student = st.sidebar.selectbox(
                "Select Student",
                options=student_options,
                index=0
            )
            if selected_student:
                self.session_state.selected_student = selected_student.split(" - ")[0]
        
        # Exam selection
        st.sidebar.subheader("📋 Exam Selection")
        exams = self.get_exams()
        if exams:
            exam_options = [f"{e['exam_id']} - {e['title']}" for e in exams]
            selected_exam = st.sidebar.selectbox(
                "Select Exam",
                options=exam_options,
                index=0
            )
            if selected_exam:
                self.session_state.selected_exam = selected_exam.split(" - ")[0]
        
        # Quick actions
        st.sidebar.subheader("⚡ Quick Actions")
        if st.sidebar.button("🔄 Refresh Data"):
            st.rerun()
        
        if st.sidebar.button("📊 Generate Report"):
            self.generate_report()
        
        if st.sidebar.button("🔍 Verify Integrity"):
            self.verify_integrity()
    
    def render_overview(self):
        """Render overview dashboard"""
        st.header("📊 System Overview")
        
        # System status
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            health = self.make_api_request("/health")
            st.metric("System Status", "Online" if health.get("status") == "healthy" else "Offline")
        
        with col2:
            processing_jobs = self.make_api_request("/processing")
            total_jobs = processing_jobs.get("total_tasks", 0)
            st.metric("Processing Jobs", total_jobs)
        
        with col3:
            # Get total students (mock data for now)
            st.metric("Total Students", "150")
        
        with col4:
            # Get total exams (mock data for now)
            st.metric("Total Exams", "25")
        
        # Recent activity
        st.subheader("📈 Recent Activity")
        
        # Processing jobs timeline
        if processing_jobs.get("tasks"):
            jobs_df = pd.DataFrame(processing_jobs["tasks"])
            if not jobs_df.empty:
                fig = px.timeline(
                    jobs_df,
                    x_start="created_at",
                    x_end="completed_at",
                    y="processing_id",
                    color="status",
                    title="Processing Jobs Timeline"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Grade distribution
        st.subheader("📊 Grade Distribution")
        if self.session_state.selected_exam:
            analytics = self.make_api_request(f"/analytics/{self.session_state.selected_exam}")
            if analytics:
                grade_dist = analytics.get("grade_distribution", {})
                if grade_dist:
                    fig = px.pie(
                        values=list(grade_dist.values()),
                        names=list(grade_dist.keys()),
                        title="Grade Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    def render_grade_management(self):
        """Render grade management interface"""
        st.header("📝 Grade Management")
        
        # Grade search and filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            student_id = st.text_input("Student ID", value=self.session_state.selected_student or "")
        
        with col2:
            exam_id = st.text_input("Exam ID", value=self.session_state.selected_exam or "")
        
        with col3:
            if st.button("🔍 Search Grades"):
                self.search_grades(student_id, exam_id)
        
        # Grade table
        if self.session_state.selected_student:
            grades = self.make_api_request(f"/grades/{self.session_state.selected_student}")
            if grades and grades.get("grades"):
                self.display_grades_table(grades["grades"])
            else:
                st.info("No grades found for the selected student.")
        else:
            st.info("Please select a student to view grades.")
    
    def display_grades_table(self, grades: List[Dict]):
        """Display grades in a table with override functionality"""
        st.subheader("📋 Student Grades")
        
        # Create DataFrame
        df = pd.DataFrame(grades)
        
        # Display table
        st.dataframe(df, use_container_width=True)
        
        # Grade override section
        st.subheader("✏️ Grade Override")
        
        with st.form("grade_override_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                question_id = st.selectbox(
                    "Question ID",
                    options=df["question_id"].tolist()
                )
                new_score = st.number_input(
                    "New Score",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0
                )
            
            with col2:
                new_feedback = st.text_area("New Feedback", height=100)
                override_reason = st.text_input("Override Reason")
            
            if st.form_submit_button("💾 Override Grade"):
                if question_id and new_score is not None and new_feedback and override_reason:
                    self.override_grade(
                        self.session_state.selected_student,
                        self.session_state.selected_exam,
                        question_id,
                        new_score,
                        new_feedback,
                        override_reason
                    )
                else:
                    st.error("Please fill in all fields.")
    
    def render_processing(self):
        """Render processing interface"""
        st.header("🔄 Answer Sheet Processing")
        
        # File upload
        st.subheader("📤 Upload Answer Sheet")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a scanned answer sheet image"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            st.image(uploaded_file, caption="Uploaded Answer Sheet", use_column_width=True)
            
            # Processing form
            with st.form("processing_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    question = st.text_area(
                        "Question",
                        placeholder="Enter the question text...",
                        height=100
                    )
                    reference_answer = st.text_area(
                        "Reference Answer (Optional)",
                        placeholder="Enter the reference answer...",
                        height=100
                    )
                
                with col2:
                    student_id = st.text_input("Student ID")
                    exam_id = st.text_input("Exam ID")
                    rubric_path = st.text_input("Rubric Path (Optional)")
                
                if st.form_submit_button("🚀 Process Answer Sheet"):
                    if question and student_id and exam_id:
                        self.process_answer_sheet(
                            uploaded_file, question, reference_answer,
                            student_id, exam_id, rubric_path
                        )
                    else:
                        st.error("Please fill in all required fields.")
        
        # Processing jobs status
        st.subheader("📋 Processing Jobs")
        self.display_processing_jobs()
    
    def display_processing_jobs(self):
        """Display current processing jobs"""
        jobs = self.make_api_request("/processing")
        
        if jobs and jobs.get("tasks"):
            jobs_df = pd.DataFrame(jobs["tasks"])
            
            # Status counts
            status_counts = jobs_df["status"].value_counts()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Pending", status_counts.get("pending", 0))
            with col2:
                st.metric("Processing", status_counts.get("processing", 0))
            with col3:
                st.metric("Completed", status_counts.get("completed", 0))
            with col4:
                st.metric("Failed", status_counts.get("failed", 0))
            
            # Jobs table
            st.dataframe(jobs_df, use_container_width=True)
            
            # Individual job details
            if st.button("🔄 Refresh Jobs"):
                st.rerun()
        else:
            st.info("No processing jobs found.")
    
    def render_analytics(self):
        """Render analytics dashboard"""
        st.header("📈 Analytics Dashboard")
        
        # Exam selection for analytics
        exams = self.get_exams()
        if exams:
            exam_options = [f"{e['exam_id']} - {e['title']}" for e in exams]
            selected_exam = st.selectbox(
                "Select Exam for Analytics",
                options=exam_options,
                key="analytics_exam"
            )
            
            if selected_exam:
                exam_id = selected_exam.split(" - ")[0]
                self.display_exam_analytics(exam_id)
        else:
            st.info("No exams available for analytics.")
    
    def display_exam_analytics(self, exam_id: str):
        """Display detailed analytics for an exam"""
        analytics = self.make_api_request(f"/analytics/{exam_id}")
        
        if analytics:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Answers", analytics.get("total_answers", 0))
            with col2:
                st.metric("Unique Students", analytics.get("unique_students", 0))
            with col3:
                st.metric("Average Score", f"{analytics.get('average_percentage', 0):.1f}%")
            with col4:
                st.metric("Score Range", f"{analytics.get('min_percentage', 0):.1f}% - {analytics.get('max_percentage', 0):.1f}%")
            
            # Grade distribution chart
            grade_dist = analytics.get("grade_distribution", {})
            if grade_dist:
                fig = px.bar(
                    x=list(grade_dist.keys()),
                    y=list(grade_dist.values()),
                    title="Grade Distribution",
                    labels={"x": "Grade", "y": "Count"}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Performance trends (mock data)
            st.subheader("📊 Performance Trends")
            
            # Generate mock trend data
            dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="D")
            scores = [70 + (i % 20) for i in range(len(dates))]
            
            trend_df = pd.DataFrame({
                "Date": dates,
                "Average Score": scores
            })
            
            fig = px.line(trend_df, x="Date", y="Average Score", title="Average Score Over Time")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Failed to load analytics data.")
    
    def render_settings(self):
        """Render settings interface"""
        st.header("⚙️ Settings")
        
        # API Configuration
        st.subheader("🔌 API Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            api_url = st.text_input("API Base URL", value=self.api_base_url)
            if st.button("💾 Save API Settings"):
                self.api_base_url = api_url
                st.success("API settings saved!")
        
        with col2:
            if st.button("🔍 Test API Connection"):
                health = self.make_api_request("/health")
                if health.get("status") == "healthy":
                    st.success("✅ API connection successful!")
                else:
                    st.error("❌ API connection failed!")
        
        # System Information
        st.subheader("ℹ️ System Information")
        
        system_info = {
            "API Base URL": self.api_base_url,
            "Dashboard Version": "1.0.0",
            "Last Updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        for key, value in system_info.items():
            st.text(f"{key}: {value}")
    
    def get_students(self) -> List[Dict]:
        """Get list of students (mock data for now)"""
        # In a real implementation, this would call the API
        return [
            {"student_id": "STU001", "name": "John Doe"},
            {"student_id": "STU002", "name": "Jane Smith"},
            {"student_id": "STU003", "name": "Bob Johnson"}
        ]
    
    def get_exams(self) -> List[Dict]:
        """Get list of exams (mock data for now)"""
        # In a real implementation, this would call the API
        return [
            {"exam_id": "EXAM001", "title": "Biology Midterm"},
            {"exam_id": "EXAM002", "title": "Mathematics Final"},
            {"exam_id": "EXAM003", "title": "Physics Quiz"}
        ]
    
    def search_grades(self, student_id: str, exam_id: str):
        """Search for grades"""
        if student_id:
            self.session_state.selected_student = student_id
        if exam_id:
            self.session_state.selected_exam = exam_id
        st.rerun()
    
    def process_answer_sheet(self, file, question: str, reference_answer: str, 
                           student_id: str, exam_id: str, rubric_path: str):
        """Process uploaded answer sheet"""
        try:
            # In a real implementation, this would upload the file to the API
            st.success("Answer sheet processing started!")
            st.info("Processing is running in the background. Check the Processing tab for status.")
            
            # Mock processing result
            with st.spinner("Processing..."):
                import time
                time.sleep(2)
                st.success("Processing completed!")
                
        except Exception as e:
            st.error(f"Error processing answer sheet: {str(e)}")
    
    def override_grade(self, student_id: str, exam_id: str, question_id: str,
                      new_score: float, new_feedback: str, override_reason: str):
        """Override a grade"""
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
                st.success("Grade overridden successfully!")
                st.rerun()
            else:
                st.error("Failed to override grade.")
                
        except Exception as e:
            st.error(f"Error overriding grade: {str(e)}")
    
    def generate_report(self):
        """Generate a comprehensive report"""
        st.info("Generating report...")
        # In a real implementation, this would generate and download a report
        st.success("Report generated successfully!")
    
    def verify_integrity(self):
        """Verify grade chain integrity"""
        try:
            result = self.make_api_request("/verify-integrity")
            
            if result.get("chain_integrity"):
                st.success("✅ Grade chain integrity verified!")
            else:
                st.error("❌ Grade chain integrity check failed!")
                st.write("Invalid records:", result.get("invalid_records", []))
                st.write("Chain breaks:", result.get("chain_breaks", []))
                
        except Exception as e:
            st.error(f"Error verifying integrity: {str(e)}")

# Main execution
if __name__ == "__main__":
    dashboard = EduGradeDashboard()
    dashboard.run()

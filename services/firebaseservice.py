from typing import Dict, Any, Optional
import os
import logging

try:
	import firebase_admin
	from firebase_admin import credentials, firestore
	FIREBASE_AVAILABLE = True
except Exception as e:
	FIREBASE_AVAILABLE = False
	firebase_admin = None
	credentials = None
	firestore = None

logger = logging.getLogger(__name__)

class FirebaseService:
	def __init__(self, credentials_path: Optional[str] = None):
		self.enabled = False
		self.db = None
		if not FIREBASE_AVAILABLE:
			logger.warning("firebase-admin not installed; Firebase disabled")
			return
		creds_path = credentials_path or os.getenv('FIREBASE_CREDENTIALS_PATH')
		if not creds_path or not os.path.exists(creds_path):
			logger.warning("Firebase credentials not provided; Firebase disabled")
			return
		try:
			if not firebase_admin._apps:
				cred = credentials.Certificate(creds_path)
				firebase_admin.initialize_app(cred)
			self.db = firestore.client()
			self.enabled = True
			logger.info("Firebase initialized")
		except Exception as e:
			logger.warning(f"Failed to init Firebase: {e}")
			self.enabled = False

	def store_grade(self, grade: Dict[str, Any]) -> Optional[str]:
		if not self.enabled:
			return None
		try:
			ref = self.db.collection('grades').document()
			ref.set(grade)
			return ref.id
		except Exception as e:
			logger.warning(f"Firebase store_grade failed: {e}")
			return None

	def store_alignment(self, submission_id: str, alignment: Dict[str, Any]) -> Optional[str]:
		if not self.enabled:
			return None

	def store_exam(self, exam: Dict[str, Any]) -> Optional[str]:
		if not self.enabled:
			return None
		try:
			ref = self.db.collection('exams').document(exam.get('exam_id'))
			ref.set(exam, merge=True)
			return ref.id
		except Exception as e:
			logger.warning(f"Firebase store_exam failed: {e}")
			return None

	def get_grades_by_student(self, student_id: str, exam_id: Optional[str] = None) -> list[Dict[str, Any]]:
		if not self.enabled:
			return []
		try:
			query = self.db.collection('grades').where('student_id', '==', student_id)
			if exam_id:
				query = query.where('exam_id', '==', exam_id)
			docs = query.stream()
			return [d.to_dict() for d in docs]
		except Exception as e:
			logger.warning(f"Firebase get_grades_by_student failed: {e}")
			return []

	def get_grades_by_exam(self, exam_id: str) -> list[Dict[str, Any]]:
		if not self.enabled:
			return []
		try:
			docs = self.db.collection('grades').where('exam_id', '==', exam_id).stream()
			return [d.to_dict() for d in docs]
		except Exception as e:
			logger.warning(f"Firebase get_grades_by_exam failed: {e}")
			return []

	def list_exam_ids(self) -> list[str]:
		if not self.enabled:
			return []
		try:
			docs = self.db.collection('grades').stream()
			exam_ids = set()
			for d in docs:
				data = d.to_dict()
				if 'exam_id' in data:
					exam_ids.add(data['exam_id'])
			return list(exam_ids)
		except Exception as e:
			logger.warning(f"Firebase list_exam_ids failed: {e}")
			return []
		try:
			ref = self.db.collection('alignments').document(submission_id)
			ref.set(alignment, merge=True)
			return ref.id
		except Exception as e:
			logger.warning(f"Firebase store_alignment failed: {e}")
			return None

"""
Pytest configuration and fixtures for EduGrade AI tests
"""

import pytest
import tempfile
import os
import shutil
from pathlib import Path

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)

@pytest.fixture
def temp_db():
    """Create a temporary database file for tests"""
    temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_file.close()
    yield temp_file.name
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)

@pytest.fixture
def sample_image():
    """Create a sample image for testing"""
    import numpy as np
    import cv2
    
    # Create a simple test image
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.rectangle(image, (10, 10), (50, 50), (0, 0, 0), -1)
    
    return image

@pytest.fixture
def sample_patches():
    """Create sample image patches for testing"""
    import numpy as np
    
    patches = [
        np.ones((50, 50, 3), dtype=np.uint8) * 255,
        np.ones((60, 60, 3), dtype=np.uint8) * 200,
        np.ones((40, 40, 3), dtype=np.uint8) * 150
    ]
    
    return patches

@pytest.fixture
def sample_question():
    """Sample question for testing"""
    return "What is photosynthesis?"

@pytest.fixture
def sample_answer():
    """Sample student answer for testing"""
    return "Photosynthesis is the process by which plants convert light energy into chemical energy."

@pytest.fixture
def sample_reference():
    """Sample reference answer for testing"""
    return "Photosynthesis is the process by which plants use sunlight to synthesize foods with the help of chlorophyll."

@pytest.fixture
def sample_rubric():
    """Sample rubric for testing"""
    return {
        "criteria": [
            {
                "name": "Accuracy",
                "weight": 0.4,
                "max_points": 40,
                "description": "Correctness of the answer",
                "keywords": ["correct", "accurate", "right"]
            },
            {
                "name": "Completeness",
                "weight": 0.3,
                "max_points": 30,
                "description": "Thoroughness of the response",
                "keywords": ["complete", "thorough", "comprehensive"]
            }
        ],
        "total_points": 100,
        "grading_level": "moderate",
        "partial_marking": True,
        "feedback_length": "medium"
    }

@pytest.fixture
def mock_gemini_client():
    """Mock Google Gemini client for testing"""
    from unittest.mock import Mock
    
    mock_model = Mock()
    mock_response = Mock()
    mock_response.text = '{"scores": {"Accuracy": 35, "Completeness": 25}, "assessment": "Good answer", "strengths": ["Clear"], "weaknesses": ["Could be more detailed"], "suggestions": ["Add examples"], "confidence": 0.8}'
    mock_model.generate_content.return_value = mock_response
    
    return mock_model

@pytest.fixture
def mock_perplexity_client():
    """Mock Perplexity client for testing"""
    from unittest.mock import Mock
    
    mock_client = Mock()
    mock_client.query.return_value = "Fact-checking response"
    
    return mock_client

@pytest.fixture
def mock_google_vision_client():
    """Mock Google Vision client for testing"""
    from unittest.mock import Mock
    
    mock_client = Mock()
    mock_response = Mock()
    mock_text = Mock()
    mock_text.description = "Extracted text from Google Vision"
    mock_response.text_annotations = [mock_text]
    mock_client.text_detection.return_value = mock_response
    
    return mock_client

@pytest.fixture
def mock_trocr_components():
    """Mock TrOCR components for testing"""
    from unittest.mock import Mock
    
    mock_processor = Mock()
    mock_model = Mock()
    mock_processor.return_value = Mock(pixel_values=Mock())
    mock_model.generate.return_value = Mock()
    mock_processor.batch_decode.return_value = ["Extracted text from TrOCR"]
    
    return mock_processor, mock_model

@pytest.fixture
def mock_easyocr_reader():
    """Mock EasyOCR reader for testing"""
    from unittest.mock import Mock
    
    mock_reader = Mock()
    mock_reader.readtext.return_value = [
        ([[0, 0], [100, 0], [100, 50], [0, 50]], "Extracted text", 0.9)
    ]
    
    return mock_reader

@pytest.fixture
def mock_yolo_model():
    """Mock YOLO model for testing"""
    from unittest.mock import Mock
    
    mock_model = Mock()
    mock_result = Mock()
    mock_result.boxes = Mock()
    mock_result.boxes.xyxy = [[10, 10, 50, 50]]
    mock_result.boxes.conf = [0.8]
    mock_model.return_value = [mock_result]
    
    return mock_model

# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Add unit marker to all tests in test_* files
        if "test_" in item.nodeid and "conftest" not in item.nodeid:
            item.add_marker(pytest.mark.unit)
        
        # Mark slow tests
        if "slow" in item.keywords:
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.keywords:
            item.add_marker(pytest.mark.integration)

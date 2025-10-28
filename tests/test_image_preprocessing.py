"""
Unit tests for Image Preprocessing Agent
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.image_preprocessing import ImagePreprocessingAgent

class TestImagePreprocessingAgent:
    """Test cases for ImagePreprocessingAgent"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.agent = ImagePreprocessingAgent()
        self.test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    def test_initialization(self):
        """Test agent initialization"""
        assert self.agent.confidence_threshold == 0.5
        assert self.agent.yolo_model is None  # No model path provided
    
    def test_initialization_with_yolo_model(self):
        """Test initialization with YOLO model path"""
        with patch('os.path.exists', return_value=True):
            with patch('ultralytics.YOLO') as mock_yolo:
                agent = ImagePreprocessingAgent(yolo_model_path="test_model.pt")
                mock_yolo.assert_called_once_with("test_model.pt")
    
    def test_enhance_image(self):
        """Test image enhancement"""
        enhanced = self.agent._enhance_image(self.test_image)
        
        assert enhanced is not None
        assert enhanced.shape[:2] == self.test_image.shape[:2]  # Same dimensions
        assert len(enhanced.shape) == 2  # Grayscale output
    
    def test_correct_rotation(self):
        """Test rotation correction"""
        # Test with no rotation needed
        corrected = self.agent._correct_rotation(self.test_image)
        assert corrected is not None
        assert corrected.shape == self.test_image.shape
    
    def test_detect_with_opencv(self):
        """Test OpenCV-based detection"""
        # Create a test image with some contours
        test_img = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(test_img, (10, 10), (50, 50), 255, -1)
        
        boxes = self.agent._detect_with_opencv(test_img)
        
        assert isinstance(boxes, list)
        # Should detect the rectangle
        assert len(boxes) > 0
        assert all('bbox' in box for box in boxes)
        assert all('confidence' in box for box in boxes)
        assert all('class' in box for box in boxes)
    
    def test_detect_with_yolo(self):
        """Test YOLO-based detection"""
        # Mock YOLO model
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.xyxy = np.array([[10, 10, 50, 50]])
        mock_result.boxes.conf = np.array([0.8])
        mock_model.return_value = [mock_result]
        
        self.agent.yolo_model = mock_model
        
        boxes = self.agent._detect_with_yolo(self.test_image)
        
        assert isinstance(boxes, list)
        assert len(boxes) == 1
        assert boxes[0]['confidence'] == 0.8
        assert boxes[0]['class'] == 'answer_box'
    
    def test_extract_answer_patches(self):
        """Test answer patch extraction"""
        boxes = [
            {'bbox': [10, 10, 50, 50], 'confidence': 0.8, 'class': 'answer_box'},
            {'bbox': [60, 60, 90, 90], 'confidence': 0.9, 'class': 'answer_box'}
        ]
        
        patches = self.agent._extract_answer_patches(self.test_image, boxes)
        
        assert isinstance(patches, list)
        assert len(patches) == 2
        assert all(isinstance(patch, np.ndarray) for patch in patches)
        assert all(patch.size > 0 for patch in patches)
    
    def test_preprocess_image_success(self):
        """Test successful image preprocessing"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, self.test_image)
            
            try:
                result = self.agent.preprocess_image(tmp_file.name)
                
                assert 'preprocessing_success' in result
                assert result['preprocessing_success'] is True
                assert 'answer_boxes' in result
                assert 'answer_patches' in result
                assert 'metadata' in result
            finally:
                os.unlink(tmp_file.name)
    
    def test_preprocess_image_failure(self):
        """Test image preprocessing failure"""
        result = self.agent.preprocess_image("nonexistent_file.jpg")
        
        assert 'preprocessing_success' in result
        assert result['preprocessing_success'] is False
        assert 'error' in result
    
    def test_save_processed_data(self):
        """Test saving processed data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_data = {
                'processed_image': self.test_image,
                'answer_patches': [self.test_image[:50, :50]],
                'metadata': {'test': 'data'}
            }
            
            saved_files = self.agent.save_processed_data(processed_data, temp_dir)
            
            assert 'processed_image' in saved_files
            assert 'answer_patch_1' in saved_files
            assert 'metadata' in saved_files
            
            # Check if files were actually created
            assert os.path.exists(saved_files['processed_image'])
            assert os.path.exists(saved_files['answer_patch_1'])
            assert os.path.exists(saved_files['metadata'])
    
    def test_detect_answer_boxes_fallback(self):
        """Test fallback from YOLO to OpenCV"""
        # Mock YOLO to raise an exception
        mock_model = Mock()
        mock_model.side_effect = Exception("YOLO error")
        self.agent.yolo_model = mock_model
        
        # Should fallback to OpenCV
        with patch.object(self.agent, '_detect_with_opencv') as mock_opencv:
            mock_opencv.return_value = []
            self.agent._detect_answer_boxes(self.test_image)
            mock_opencv.assert_called_once_with(self.test_image)
    
    def test_confidence_threshold_filtering(self):
        """Test confidence threshold filtering"""
        # Mock YOLO with different confidence levels
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.xyxy = np.array([[10, 10, 50, 50], [60, 60, 90, 90]])
        mock_result.boxes.conf = np.array([0.3, 0.8])  # One below, one above threshold
        mock_model.return_value = [mock_result]
        
        self.agent.yolo_model = mock_model
        self.agent.confidence_threshold = 0.5
        
        boxes = self.agent._detect_with_yolo(self.test_image)
        
        # Should only return boxes above threshold
        assert len(boxes) == 1
        assert boxes[0]['confidence'] == 0.8

if __name__ == "__main__":
    pytest.main([__file__])

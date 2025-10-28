"""
Unit tests for OCR Extraction Agent
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import tempfile
import os

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.ocr_extraction import OCRExtractionAgent

class TestOCRExtractionAgent:
    """Test cases for OCRExtractionAgent"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.agent = OCRExtractionAgent()
        self.test_image = Image.new('RGB', (100, 100), color='white')
        self.test_patches = [np.ones((50, 50, 3), dtype=np.uint8) * 255]
    
    def test_initialization(self):
        """Test agent initialization"""
        assert self.agent.languages == ['en', 'hi', 'ta']
        assert self.agent.google_client is None  # No credentials provided
        assert self.agent.trocr_processor is None  # Mocked in tests
        assert self.agent.easyocr_reader is None  # Mocked in tests
    
    def test_initialization_with_credentials(self):
        """Test initialization with Google credentials"""
        with patch('os.path.exists', return_value=True):
            with patch('google.oauth2.service_account.Credentials.from_service_account_file') as mock_creds:
                with patch('google.cloud.vision.ImageAnnotatorClient') as mock_client:
                    agent = OCRExtractionAgent(google_credentials_path="test_creds.json")
                    mock_creds.assert_called_once_with("test_creds.json")
                    mock_client.assert_called_once()
    
    def test_detect_language_english(self):
        """Test English language detection"""
        text = "This is an English text with many words."
        language = self.agent._detect_language(text)
        assert language == 'en'
    
    def test_detect_language_hindi(self):
        """Test Hindi language detection"""
        text = "यह हिंदी में लिखा गया पाठ है।"
        language = self.agent._detect_language(text)
        assert language == 'hi'
    
    def test_detect_language_tamil(self):
        """Test Tamil language detection"""
        text = "இது தமிழில் எழுதப்பட்ட உரை."
        language = self.agent._detect_language(text)
        assert language == 'ta'
    
    def test_detect_language_mixed(self):
        """Test mixed language detection"""
        text = "This is mixed text with English and हिंदी words."
        language = self.agent._detect_language(text)
        assert language == 'mixed'
    
    def test_detect_language_empty(self):
        """Test empty text language detection"""
        text = ""
        language = self.agent._detect_language(text)
        assert language == 'unknown'
    
    def test_extract_with_google_vision(self):
        """Test Google Vision API extraction"""
        # Mock Google Vision client
        mock_client = Mock()
        mock_response = Mock()
        mock_text = Mock()
        mock_text.description = "Extracted text from Google Vision"
        mock_response.text_annotations = [mock_text]
        mock_client.text_detection.return_value = mock_response
        
        self.agent.google_client = mock_client
        
        result = self.agent._extract_with_google_vision(self.test_image)
        
        assert result['text'] == "Extracted text from Google Vision"
        assert result['confidence'] == 0.9
        assert result['method'] == 'google_vision'
        assert result['language'] == 'en'
    
    def test_extract_with_google_vision_no_text(self):
        """Test Google Vision API with no text found"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text_annotations = []
        mock_client.text_detection.return_value = mock_response
        
        self.agent.google_client = mock_client
        
        result = self.agent._extract_with_google_vision(self.test_image)
        
        assert result['text'] == ''
        assert result['confidence'] == 0.0
        assert result['method'] == 'google_vision'
    
    def test_extract_with_trocr(self):
        """Test TrOCR extraction"""
        # Mock TrOCR components
        mock_processor = Mock()
        mock_model = Mock()
        mock_processor.return_value = Mock(pixel_values=Mock())
        mock_model.generate.return_value = Mock()
        mock_processor.batch_decode.return_value = ["Extracted text from TrOCR"]
        
        self.agent.trocr_processor = mock_processor
        self.agent.trocr_model = mock_model
        
        result = self.agent._extract_with_trocr(self.test_image)
        
        assert result['text'] == "Extracted text from TrOCR"
        assert result['confidence'] == 0.8
        assert result['method'] == 'trocr'
        assert result['language'] == 'en'
    
    def test_extract_with_easyocr(self):
        """Test EasyOCR extraction"""
        # Mock EasyOCR reader
        mock_reader = Mock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 50], [0, 50]], "Extracted text", 0.9)
        ]
        
        self.agent.easyocr_reader = mock_reader
        
        result = self.agent._extract_with_easyocr(self.test_image)
        
        assert result['text'] == "Extracted text"
        assert result['confidence'] == 0.9
        assert result['method'] == 'easyocr'
        assert result['language'] == 'en'
    
    def test_extract_with_easyocr_no_text(self):
        """Test EasyOCR with no text found"""
        mock_reader = Mock()
        mock_reader.readtext.return_value = []
        
        self.agent.easyocr_reader = mock_reader
        
        result = self.agent._extract_with_easyocr(self.test_image)
        
        assert result['text'] == ''
        assert result['confidence'] == 0.0
        assert result['method'] == 'easyocr'
    
    def test_extract_text_ensemble(self):
        """Test ensemble text extraction"""
        # Mock all OCR methods
        with patch.object(self.agent, '_extract_with_google_vision') as mock_google:
            with patch.object(self.agent, '_extract_with_trocr') as mock_trocr:
                with patch.object(self.agent, '_extract_with_easyocr') as mock_easyocr:
                    mock_google.return_value = {'text': 'Google text', 'confidence': 0.9, 'method': 'google_vision'}
                    mock_trocr.return_value = {'text': 'TrOCR text', 'confidence': 0.8, 'method': 'trocr'}
                    mock_easyocr.return_value = {'text': 'EasyOCR text', 'confidence': 0.7, 'method': 'easyocr'}
                    
                    # Mock clients to enable all methods
                    self.agent.google_client = Mock()
                    self.agent.trocr_processor = Mock()
                    self.agent.trocr_model = Mock()
                    self.agent.easyocr_reader = Mock()
                    
                    results = self.agent._extract_text_ensemble(self.test_image)
                    
                    assert 'google_vision' in results
                    assert 'trocr' in results
                    assert 'easyocr' in results
    
    def test_combine_extraction_results_single(self):
        """Test combining single extraction result"""
        results = {
            'google_vision': {'text': 'Test text', 'confidence': 0.9, 'method': 'google_vision', 'language': 'en'}
        }
        
        combined = self.agent._combine_extraction_results(results)
        
        assert combined['text'] == 'Test text'
        assert combined['confidence'] == 0.9
        assert combined['method'] == 'google_vision'
    
    def test_combine_extraction_results_multiple(self):
        """Test combining multiple extraction results"""
        results = {
            'google_vision': {'text': 'Google text', 'confidence': 0.9, 'method': 'google_vision', 'language': 'en'},
            'trocr': {'text': 'TrOCR text', 'confidence': 0.8, 'method': 'trocr', 'language': 'en'},
            'easyocr': {'text': 'EasyOCR text', 'confidence': 0.7, 'method': 'easyocr', 'language': 'en'}
        }
        
        combined = self.agent._combine_extraction_results(results)
        
        # Should use the best result (Google Vision)
        assert combined['text'] == 'Google text'
        assert combined['confidence'] == 0.9
        assert 'google_vision' in combined['method']
    
    def test_merge_texts(self):
        """Test text merging functionality"""
        texts = ['Short text', 'This is a longer text with more words']
        merged = self.agent._merge_texts(texts)
        
        assert isinstance(merged, str)
        assert len(merged) > 0
    
    def test_extract_text_from_patches(self):
        """Test extracting text from multiple patches"""
        # Mock the ensemble extraction
        with patch.object(self.agent, '_extract_text_ensemble') as mock_ensemble:
            mock_ensemble.return_value = {
                'google_vision': {'text': 'Test text', 'confidence': 0.9, 'method': 'google_vision', 'language': 'en'}
            }
            
            results = self.agent.extract_text_from_patches(self.test_patches)
            
            assert len(results) == 1
            assert results[0]['extracted_text'] == 'Test text'
            assert results[0]['confidence'] == 0.9
            assert results[0]['patch_index'] == 0
    
    def test_extract_equations_and_diagrams(self):
        """Test equation and diagram extraction"""
        result = self.agent.extract_equations_and_diagrams(self.test_image)
        
        assert 'equations' in result
        assert 'diagrams' in result
        assert 'has_math_content' in result
        assert 'has_diagram_content' in result
        assert isinstance(result['equations'], list)
        assert isinstance(result['diagrams'], list)
        assert isinstance(result['has_math_content'], bool)
        assert isinstance(result['has_diagram_content'], bool)
    
    def test_error_handling_google_vision(self):
        """Test error handling in Google Vision extraction"""
        mock_client = Mock()
        mock_client.text_detection.side_effect = Exception("API Error")
        self.agent.google_client = mock_client
        
        result = self.agent._extract_with_google_vision(self.test_image)
        
        assert 'error' in result
        assert result['text'] == ''
        assert result['confidence'] == 0.0
    
    def test_error_handling_trocr(self):
        """Test error handling in TrOCR extraction"""
        mock_processor = Mock()
        mock_processor.side_effect = Exception("TrOCR Error")
        self.agent.trocr_processor = mock_processor
        
        result = self.agent._extract_with_trocr(self.test_image)
        
        assert 'error' in result
        assert result['text'] == ''
        assert result['confidence'] == 0.0
    
    def test_error_handling_easyocr(self):
        """Test error handling in EasyOCR extraction"""
        mock_reader = Mock()
        mock_reader.readtext.side_effect = Exception("EasyOCR Error")
        self.agent.easyocr_reader = mock_reader
        
        result = self.agent._extract_with_easyocr(self.test_image)
        
        assert 'error' in result
        assert result['text'] == ''
        assert result['confidence'] == 0.0

if __name__ == "__main__":
    pytest.main([__file__])

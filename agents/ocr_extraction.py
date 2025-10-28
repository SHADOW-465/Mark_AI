"""
OCR Extraction Agent for EduGrade AI
Handles text extraction from answer patches using Google Vision API and TrOCR
Supports Hindi, Tamil, and English handwriting
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from PIL import Image
import io
import base64
import json

# Google Vision API
try:
    from google.cloud import vision
    from google.oauth2 import service_account
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False
    logging.warning("Google Vision API not available. Install google-cloud-vision")

# TrOCR
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False
    logging.warning("TrOCR not available. Install transformers and torch")

# EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.warning("EasyOCR not available. Install easyocr")

logger = logging.getLogger(__name__)

class OCRExtractionAgent:
    """Agent responsible for extracting text from answer patches using multiple OCR methods"""
    
    def __init__(self, 
                 google_credentials_path: Optional[str] = None,
                 trocr_model_name: str = "microsoft/trocr-base-handwritten",
                 languages: List[str] = ['en', 'hi', 'ta']):
        """
        Initialize the OCR Extraction Agent
        
        Args:
            google_credentials_path: Path to Google Cloud credentials JSON file
            trocr_model_name: TrOCR model name
            languages: List of languages to support
        """
        self.languages = languages
        self.google_client = None
        self.trocr_processor = None
        self.trocr_model = None
        self.easyocr_reader = None
        self.deepseek_available = False
        self.gemini_vision_available = False
        
        # Initialize Google Vision API
        if GOOGLE_VISION_AVAILABLE and google_credentials_path:
            try:
                credentials = service_account.Credentials.from_service_account_file(
                    google_credentials_path
                )
                self.google_client = vision.ImageAnnotatorClient(credentials=credentials)
                logger.info("Google Vision API client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Google Vision API: {e}")
        
        # Initialize TrOCR
        if TROCR_AVAILABLE:
            try:
                self.trocr_processor = TrOCRProcessor.from_pretrained(trocr_model_name)
                self.trocr_model = VisionEncoderDecoderModel.from_pretrained(trocr_model_name)
                logger.info(f"TrOCR model loaded: {trocr_model_name}")
            except Exception as e:
                logger.warning(f"Failed to load TrOCR model: {e}")
        
        # Initialize EasyOCR
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(languages, gpu=False)
                logger.info(f"EasyOCR initialized with languages: {languages}")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")
        
        # Initialize DeepSeek-OCR via vLLM if available
        try:
            from vllm import LLM
            from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
            self._deepseek_logits_processors = [NGramPerReqLogitsProcessor]
            # Lazy model load; defer heavy init until first call
            self._deepseek_llm = None
            self._deepseek_model_name = os.getenv('DEEPSEEK_OCR_MODEL', 'deepseek-ai/DeepSeek-OCR')
            self.deepseek_available = True
            logger.info("DeepSeek-OCR available via vLLM")
        except Exception as e:
            self.deepseek_available = False
            self._deepseek_llm = None
            logger.warning(f"DeepSeek-OCR not available: {e}")

        # Initialize DeepSeek-OCR transformers fallback (no GPU/vLLM)
        self.deepseek_transformers_available = False
        self._deepseek_tok = None
        self._deepseek_tf_model = None
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            self.deepseek_transformers_available = True
            logger.info("DeepSeek-OCR transformers fallback available")
        except Exception as e:
            logger.warning(f"Transformers fallback not available: {e}")
        
        # Initialize Gemini Vision stub (placeholder)
        self.gemini_vision_available = True
    
    def extract_text_from_patches(self, patches: List[np.ndarray]) -> List[Dict]:
        """
        Extract text from multiple answer patches using ensemble approach
        
        Args:
            patches: List of image patches (numpy arrays)
            
        Returns:
            List of dictionaries containing extracted text and metadata
        """
        results = []
        
        for i, patch in enumerate(patches):
            logger.info(f"Processing answer patch {i+1}/{len(patches)}")
            
            # Convert numpy array to PIL Image
            if len(patch.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(patch)
            
            # Pre-enhance faint/noisy patches
            enhanced = self._enhance_for_ocr(pil_image)
            
            # Extract text using multiple methods
            extraction_results = self._extract_text_ensemble(enhanced)
            
            # Combine results
            combined_result = self._combine_extraction_results(extraction_results)
            
            results.append({
                'patch_index': i,
                'extracted_text': combined_result['text'],
                'confidence': combined_result['confidence'],
                'language': combined_result['language'],
                'method_used': combined_result['method'],
                'raw_results': extraction_results,
                'metadata': {
                    'patch_shape': patch.shape,
                    'processing_success': True
                }
            })
        
        return results
    
    def _extract_text_ensemble(self, image: Image.Image) -> Dict:
        """Extract text using multiple OCR methods and combine results"""
        results = {}
        
        # Google Vision API
        if self.google_client:
            try:
                google_result = self._extract_with_google_vision(image)
                results['google_vision'] = google_result
            except Exception as e:
                logger.warning(f"Google Vision extraction failed: {e}")
                results['google_vision'] = {'text': '', 'confidence': 0.0, 'error': str(e)}
        
        # TrOCR
        if self.trocr_processor and self.trocr_model:
            try:
                trocr_result = self._extract_with_trocr(image)
                results['trocr'] = trocr_result
            except Exception as e:
                logger.warning(f"TrOCR extraction failed: {e}")
                results['trocr'] = {'text': '', 'confidence': 0.0, 'error': str(e)}
        
        # EasyOCR
        if self.easyocr_reader:
            try:
                easyocr_result = self._extract_with_easyocr(image)
                results['easyocr'] = easyocr_result
            except Exception as e:
                logger.warning(f"EasyOCR extraction failed: {e}")
                results['easyocr'] = {'text': '', 'confidence': 0.0, 'error': str(e)}
        
        # DeepSeek-OCR (vLLM or transformers fallback)
        if self.deepseek_available or self.deepseek_transformers_available:
            try:
                deepseek_result = self._extract_with_deepseek(image)
                results['deepseek'] = deepseek_result
            except Exception as e:
                logger.warning(f"DeepSeek extraction failed: {e}")
                results['deepseek'] = {'text': '', 'confidence': 0.0, 'error': str(e)}
        
        # Gemini Vision (stub)
        if self.gemini_vision_available:
            try:
                gemini_vis_result = self._extract_with_gemini_vision(image)
                results['gemini_vision'] = gemini_vis_result
            except Exception as e:
                logger.warning(f"Gemini Vision extraction failed: {e}")
                results['gemini_vision'] = {'text': '', 'confidence': 0.0, 'error': str(e)}
        
        return results
    
    def _extract_with_google_vision(self, image: Image.Image) -> Dict:
        """Extract text using Google Vision API"""
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Create image object
        vision_image = vision.Image(content=img_byte_arr)
        
        # Perform text detection
        response = self.google_client.text_detection(image=vision_image)
        texts = response.text_annotations
        
        if texts:
            # Get the first (full) text annotation
            full_text = texts[0].description
            confidence = 0.9  # Google Vision doesn't provide confidence for full text
            
            # Detect language
            language = self._detect_language(full_text)
            
            return {
                'text': full_text.strip(),
                'confidence': confidence,
                'language': language,
                'method': 'google_vision'
            }
        else:
            return {
                'text': '',
                'confidence': 0.0,
                'language': 'unknown',
                'method': 'google_vision'
            }
    
    def _extract_with_trocr(self, image: Image.Image) -> Dict:
        """Extract text using TrOCR"""
        # Preprocess image
        pixel_values = self.trocr_processor(image, return_tensors="pt").pixel_values
        
        # Generate text
        with torch.no_grad():
            generated_ids = self.trocr_model.generate(pixel_values)
            generated_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Estimate confidence (TrOCR doesn't provide confidence scores)
        confidence = 0.8 if len(generated_text.strip()) > 0 else 0.0
        
        # Detect language
        language = self._detect_language(generated_text)
        
        return {
            'text': generated_text.strip(),
            'confidence': confidence,
            'language': language,
            'method': 'trocr'
        }
    
    def _extract_with_easyocr(self, image: Image.Image) -> Dict:
        """Extract text using EasyOCR"""
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Perform OCR
        results = self.easyocr_reader.readtext(img_array)
        
        if results:
            # Combine all text
            full_text = ' '.join([result[1] for result in results])
            
            # Calculate average confidence
            confidences = [result[2] for result in results]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Detect language
            language = self._detect_language(full_text)
            
            return {
                'text': full_text.strip(),
                'confidence': avg_confidence,
                'language': language,
                'method': 'easyocr'
            }
        else:
            return {
                'text': '',
                'confidence': 0.0,
                'language': 'unknown',
                'method': 'easyocr'
            }
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on character sets"""
        if not text.strip():
            return 'unknown'
        
        # Check for Hindi characters (Devanagari script)
        hindi_chars = sum(1 for char in text if '\u0900' <= char <= '\u097F')
        
        # Check for Tamil characters
        tamil_chars = sum(1 for char in text if '\u0B80' <= char <= '\u0BFF')
        
        # Check for English characters
        english_chars = sum(1 for char in text if char.isalpha() and ord(char) < 128)
        
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars == 0:
            return 'unknown'
        
        hindi_ratio = hindi_chars / total_chars
        tamil_ratio = tamil_chars / total_chars
        english_ratio = english_chars / total_chars
        
        if hindi_ratio > 0.3:
            return 'hi'
        elif tamil_ratio > 0.3:
            return 'ta'
        elif english_ratio > 0.5:
            return 'en'
        else:
            return 'mixed'
    
    def _combine_extraction_results(self, results: Dict) -> Dict:
        """Combine results from multiple OCR methods"""
        valid_results = []
        
        for method, result in results.items():
            if 'error' not in result and result['text'].strip():
                valid_results.append(result)
        
        if not valid_results:
            return {
                'text': '',
                'confidence': 0.0,
                'language': 'unknown',
                'method': 'none'
            }
        
        # If only one valid result, return it
        if len(valid_results) == 1:
            return valid_results[0]
        
        # Combine multiple results
        # Priority: Gemini Vision > Google Vision > DeepSeek > TrOCR > EasyOCR
        method_priority = {'gemini_vision': 5, 'google_vision': 4, 'deepseek': 3, 'trocr': 2, 'easyocr': 1}
        
        # Sort by priority and confidence
        valid_results.sort(key=lambda x: (
            method_priority.get(x['method'], 0),
            x['confidence']
        ), reverse=True)
        
        # Use the best result
        best_result = valid_results[0]
        
        # If we have multiple results, try to combine them
        if len(valid_results) > 1:
            combined_text = self._merge_texts([r['text'] for r in valid_results])
            avg_confidence = sum(r['confidence'] for r in valid_results) / len(valid_results)
            
            return {
                'text': combined_text,
                'confidence': avg_confidence,
                'language': best_result['language'],
                'method': f"ensemble_{best_result['method']}"
            }
        
        return best_result

    def _enhance_for_ocr(self, image: Image.Image) -> Image.Image:
        """Enhance faint/noisy writing: contrast, denoise, adaptive threshold."""
        try:
            import cv2
            import numpy as np
            arr = np.array(image)
            if arr.ndim == 3:
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            else:
                gray = arr
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            # CLAHE for contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(gray)
            thr = cv2.adaptiveThreshold(cl, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            return Image.fromarray(thr)
        except Exception:
            return image

    def _extract_with_deepseek(self, image: Image.Image) -> Dict:
        """DeepSeek-OCR extraction: prefer vLLM, fallback to transformers if vLLM/GPU unavailable."""
        # Prefer vLLM if available
        if self.deepseek_available:
            try:
                from vllm import LLM, SamplingParams
                if self._deepseek_llm is None:
                    self._deepseek_llm = LLM(
                        model=self._deepseek_model_name,
                        enable_prefix_caching=False,
                        mm_processor_cache_gb=0,
                        logits_processors=self._deepseek_logits_processors
                    )
                prompt = "<image>\nFree OCR."
                pil_image = image.convert("RGB")
                model_input = [{
                    'prompt': prompt,
                    'multi_modal_data': {'image': pil_image}
                }]
                sampling_param = SamplingParams(
                    temperature=0.0,
                    max_tokens=8192,
                    extra_args=dict(
                        ngram_size=30,
                        window_size=90,
                        whitelist_token_ids={128821, 128822},
                    ),
                    skip_special_tokens=False,
                )
                outputs = self._deepseek_llm.generate(model_input, sampling_param)
                text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ''
                language = self._detect_language(text)
                confidence = 0.85 if text.strip() else 0.0
                return {
                    'text': text.strip(),
                    'confidence': confidence,
                    'language': language,
                    'method': 'deepseek'
                }
            except Exception as e:
                logger.warning(f"DeepSeek vLLM path failed, trying transformers fallback: {e}")

        # Transformers fallback
        if not self.deepseek_transformers_available:
            raise RuntimeError("DeepSeek transformers fallback not available")

        import torch
        from transformers import AutoModel, AutoTokenizer
        if self._deepseek_tok is None or self._deepseek_tf_model is None:
            model_name = os.getenv('DEEPSEEK_OCR_MODEL', 'deepseek-ai/DeepSeek-OCR')
            self._deepseek_tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            use_cuda = torch.cuda.is_available()
            attn_impl = 'flash_attention_2' if use_cuda else None
            if attn_impl:
                self._deepseek_tf_model = AutoModel.from_pretrained(
                    model_name,
                    _attn_implementation=attn_impl,
                    trust_remote_code=True,
                    use_safetensors=True
                )
            else:
                self._deepseek_tf_model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    use_safetensors=True
                )
            self._deepseek_tf_model = self._deepseek_tf_model.eval()
            if use_cuda:
                self._deepseek_tf_model = self._deepseek_tf_model.cuda().to(torch.bfloat16)

        # Save PIL image to a temporary file because infer expects a path
        import tempfile
        import uuid
        tmp_dir = tempfile.gettempdir()
        tmp_path = os.path.join(tmp_dir, f"deepseek_{uuid.uuid4().hex}.jpg")
        image.convert("RGB").save(tmp_path)

        prompt = os.getenv('DEEPSEEK_PROMPT', '<image>\nFree OCR.')
        try:
            res = self._deepseek_tf_model.infer(
                self._deepseek_tok,
                prompt=prompt,
                image_file=tmp_path,
                output_path=tmp_dir,
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=False,
                test_compress=True
            )
            # infer may save outputs; here we rely on returned result text if available
            text = ''
            if isinstance(res, dict) and 'text' in res:
                text = res['text']
        except Exception as e:
            logger.warning(f"DeepSeek transformers infer failed: {e}")
            text = ''
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        language = self._detect_language(text)
        confidence = 0.8 if text and text.strip() else 0.0
        return {
            'text': (text or '').strip(),
            'confidence': confidence,
            'language': language,
            'method': 'deepseek-transformers'
        }

    def _extract_with_gemini_vision(self, image: Image.Image) -> Dict:
        """Gemini Vision placeholder for diagrams/math-rich content."""
        # Placeholder: detect if image is sparse text -> low confidence
        return {
            'text': '',
            'confidence': 0.0,
            'language': 'unknown',
            'method': 'gemini_vision'
        }
    
    def _merge_texts(self, texts: List[str]) -> str:
        """Merge multiple text extractions intelligently"""
        if not texts:
            return ''
        
        if len(texts) == 1:
            return texts[0]
        
        # Simple merging strategy: use the longest text that's not too different
        texts = [t.strip() for t in texts if t.strip()]
        
        if not texts:
            return ''
        
        # Sort by length
        texts.sort(key=len, reverse=True)
        
        # Use the longest text as base
        base_text = texts[0]
        
        # Check if other texts are significantly different
        for text in texts[1:]:
            if len(text) > len(base_text) * 0.8:  # Similar length
                # Check for common words
                base_words = set(base_text.lower().split())
                text_words = set(text.lower().split())
                common_words = base_words.intersection(text_words)
                
                if len(common_words) > len(base_words) * 0.5:  # Similar content
                    # Merge by taking longer words
                    base_words_list = base_text.split()
                    text_words_list = text.split()
                    
                    merged_words = []
                    for i in range(max(len(base_words_list), len(text_words_list))):
                        if i < len(base_words_list) and i < len(text_words_list):
                            # Take the longer word
                            if len(text_words_list[i]) > len(base_words_list[i]):
                                merged_words.append(text_words_list[i])
                            else:
                                merged_words.append(base_words_list[i])
                        elif i < len(base_words_list):
                            merged_words.append(base_words_list[i])
                        else:
                            merged_words.append(text_words_list[i])
                    
                    base_text = ' '.join(merged_words)
        
        return base_text
    
    def extract_equations_and_diagrams(self, image: Image.Image) -> Dict:
        """Extract mathematical equations and diagrams from image"""
        # This is a placeholder for advanced equation/diagram extraction
        # In a full implementation, you would use specialized libraries like:
        # - MathPix API for equations
        # - Custom CNN models for diagram recognition
        # - LaTeX rendering for equation conversion
        
        return {
            'equations': [],
            'diagrams': [],
            'has_math_content': False,
            'has_diagram_content': False
        }

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize agent
    agent = OCRExtractionAgent()
    
    # Test with sample image
    # sample_image = Image.open("sample_answer.jpg")
    # result = agent.extract_text_from_patches([np.array(sample_image)])
    # print(f"Extracted text: {result[0]['extracted_text']}")

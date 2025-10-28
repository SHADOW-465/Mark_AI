"""
Image Preprocessing Agent for EduGrade AI
Handles image alignment, rotation correction, and answer sheet detection using OpenCV and YOLOv8
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import logging
from ultralytics import YOLO
from PIL import Image
import os

logger = logging.getLogger(__name__)

class ImagePreprocessingAgent:
    """Agent responsible for preprocessing answer sheet images"""
    
    def __init__(self, yolo_model_path: str = None, confidence_threshold: float = 0.5):
        """
        Initialize the Image Preprocessing Agent
        
        Args:
            yolo_model_path: Path to YOLOv8 model for answer sheet detection
            confidence_threshold: Minimum confidence for detection
        """
        self.confidence_threshold = confidence_threshold
        self.yolo_model = None
        
        if yolo_model_path and os.path.exists(yolo_model_path):
            try:
                self.yolo_model = YOLO(yolo_model_path)
                logger.info(f"YOLO model loaded from {yolo_model_path}")
            except Exception as e:
                logger.warning(f"Failed to load YOLO model: {e}")
                logger.info("Using OpenCV-based detection as fallback")
        else:
            logger.info("No YOLO model provided, using OpenCV-based detection")
    
    def preprocess_image(self, image_path: str, template_path: Optional[str] = None) -> Dict:
        """
        Main preprocessing pipeline for answer sheet images
        
        Args:
            image_path: Path to the input image
            template_path: Optional path to the original question paper template for geometric registration
            
        Returns:
            Dictionary containing processed image data and metadata
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            original_image = image.copy()
            
            # Step 1: Image enhancement and noise reduction
            enhanced_image = self._enhance_image(image)
            
            # Step 2: Rotation correction
            corrected_image = self._correct_rotation(enhanced_image)
            
            alignment_info = None
            aligned_image = corrected_image
            
            # Step 3: Optional template-based geometric registration
            if template_path and os.path.exists(template_path):
                try:
                    template_img = cv2.imread(template_path)
                    if template_img is None:
                        raise ValueError(f"Could not load template from {template_path}")
                    aligned_image, alignment_info = self._align_to_template(corrected_image, template_img)
                except Exception as e:
                    logger.warning(f"Template-based alignment failed: {e}")
                    alignment_info = {
                        'status': 'failed',
                        'error': str(e)
                    }
            
            # Step 4: Answer sheet detection and segmentation
            answer_boxes = self._detect_answer_boxes(aligned_image)
            
            # Step 5: Extract individual answer regions
            answer_patches = self._extract_answer_patches(aligned_image, answer_boxes)
            
            return {
                'original_image': original_image,
                'processed_image': aligned_image,
                'aligned_image': aligned_image,
                'answer_boxes': answer_boxes,
                'answer_patches': answer_patches,
                'metadata': {
                    'image_path': image_path,
                    'original_shape': original_image.shape,
                    'processed_shape': aligned_image.shape,
                    'num_answers': len(answer_patches),
                    'preprocessing_success': True,
                    'alignment': alignment_info
                }
            }
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {e}")
            return {
                'error': str(e),
                'preprocessing_success': False
            }
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance image quality for better processing"""
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up the image
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def _correct_rotation(self, image: np.ndarray) -> np.ndarray:
        """Correct image rotation using contour detection"""
        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # Find the largest contour (likely the answer sheet)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        
        # Correct angle if it's significantly off
        if abs(angle) > 1:
            # Rotate the image
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
        
        return image
    
    def _detect_answer_boxes(self, image: np.ndarray) -> List[Dict]:
        """Detect answer boxes using YOLO or OpenCV-based methods"""
        if self.yolo_model:
            return self._detect_with_yolo(image)
        else:
            return self._detect_with_opencv(image)
    
    def _detect_with_yolo(self, image: np.ndarray) -> List[Dict]:
        """Detect answer boxes using YOLOv8"""
        try:
            results = self.yolo_model(image, conf=self.confidence_threshold)
            
            answer_boxes = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        answer_boxes.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class': 'answer_box'
                        })
            
            return answer_boxes
            
        except Exception as e:
            logger.warning(f"YOLO detection failed: {e}, falling back to OpenCV")
            return self._detect_with_opencv(image)
    
    def _detect_with_opencv(self, image: np.ndarray) -> List[Dict]:
        """Detect answer boxes using OpenCV contour detection"""
        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        answer_boxes = []
        min_area = 1000  # Minimum area for answer boxes
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (answer boxes are typically rectangular)
                aspect_ratio = w / h
                if 0.5 < aspect_ratio < 3.0:
                    answer_boxes.append({
                        'bbox': [x, y, x + w, y + h],
                        'confidence': 0.8,  # Default confidence for OpenCV detection
                        'class': 'answer_box'
                    })
        
        # Sort by position (top to bottom, left to right)
        answer_boxes.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))
        
        return answer_boxes
    
    def _extract_answer_patches(self, image: np.ndarray, answer_boxes: List[Dict]) -> List[np.ndarray]:
        """Extract individual answer patches from detected boxes"""
        patches = []
        
        for i, box in enumerate(answer_boxes):
            x1, y1, x2, y2 = box['bbox']
            
            # Add padding around the box
            padding = 10
            h, w = image.shape[:2]
            
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # Extract patch
            patch = image[y1:y2, x1:x2]
            
            if patch.size > 0:
                patches.append(patch)
                logger.debug(f"Extracted answer patch {i+1}: {patch.shape}")
        
        return patches
    
    def _align_to_template(self, image: np.ndarray, template: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Align the input image to the template using feature-based homography.
        Returns aligned image and alignment metadata with transformation parameters and accuracy.
        """
        # Ensure grayscale for feature detection
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
        
        # Detect and compute ORB features
        orb = cv2.ORB_create(5000)
        kp1, des1 = orb.detectAndCompute(img_gray, None)
        kp2, des2 = orb.detectAndCompute(tpl_gray, None)
        if des1 is None or des2 is None:
            raise ValueError("Insufficient features for alignment")
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if len(good) < 10:
            raise ValueError("Not enough good matches for reliable homography")
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        # Compute homography with RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            raise ValueError("Homography computation failed")
        inliers = int(mask.sum()) if mask is not None else 0
        total = int(len(mask)) if mask is not None else len(good)
        inlier_ratio = inliers / max(total, 1)
        
        # Warp image to template space
        height, width = (template.shape[0], template.shape[1])
        aligned = cv2.warpPerspective(image, H, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        # Compute simple alignment accuracy metric via SSIM over downscaled grayscale
        try:
            from skimage.metrics import structural_similarity as ssim
            img_small = cv2.resize(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY) if len(aligned.shape) == 3 else aligned, (min(800, width), min(800, height)))
            tpl_small = cv2.resize(tpl_gray, (img_small.shape[1], img_small.shape[0]))
            accuracy = float(ssim(img_small, tpl_small))
        except Exception:
            # Fallback: normalized inlier ratio as proxy
            accuracy = float(min(max(inlier_ratio, 0.0), 1.0))
        
        alignment_info = {
            'status': 'success',
            'transformation': {
                'type': 'homography',
                'matrix': H.tolist()
            },
            'matches': {
                'total': total,
                'inliers': inliers,
                'inlier_ratio': inlier_ratio
            },
            'accuracy': accuracy,
            'template_shape': template.shape[:2]
        }
        
        return aligned, alignment_info
    
    def save_processed_data(self, processed_data: Dict, output_dir: str) -> Dict:
        """Save processed image data to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        try:
            # Save processed image
            processed_image_path = output_path / "processed_image.jpg"
            cv2.imwrite(str(processed_image_path), processed_data['processed_image'])
            saved_files['processed_image'] = str(processed_image_path)
            
            # Save answer patches
            patches_dir = output_path / "answer_patches"
            patches_dir.mkdir(exist_ok=True)
            
            for i, patch in enumerate(processed_data['answer_patches']):
                patch_path = patches_dir / f"answer_{i+1:03d}.jpg"
                cv2.imwrite(str(patch_path), patch)
                saved_files[f'answer_patch_{i+1}'] = str(patch_path)
            
            # Save metadata
            metadata_path = output_path / "metadata.json"
            import json
            with open(metadata_path, 'w') as f:
                json.dump(processed_data['metadata'], f, indent=2)
            saved_files['metadata'] = str(metadata_path)
            
            logger.info(f"Processed data saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            saved_files['error'] = str(e)
        
        return saved_files

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize agent
    agent = ImagePreprocessingAgent()
    
    # Test with sample image (replace with actual image path)
    # result = agent.preprocess_image("sample_answer_sheet.jpg")
    # print(f"Processing result: {result['metadata']}")

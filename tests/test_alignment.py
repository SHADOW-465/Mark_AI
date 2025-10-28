import numpy as np
import cv2
from agents.image_preprocessing import ImagePreprocessingAgent

def test_alignment_runs_without_template():
    agent = ImagePreprocessingAgent()
    # Create a synthetic image
    img = np.full((200, 300), 255, dtype=np.uint8)
    cv2.putText(img, 'TEST', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,), 2)
    path = 'tmp_test.jpg'
    cv2.imwrite(path, img)
    out = agent.preprocess_image(path)
    assert out.get('metadata', {}).get('preprocessing_success') is True



import cv2
import numpy as np

class PointOperation:
    """Point operations work on individual pixels"""
    
    def brightness_adjustment(self, image, brightness_value=0):
        """Adjust brightness by adding/subtracting a constant value"""
        result = cv2.convertScaleAbs(image, alpha=1.0, beta=brightness_value)
        return result
    
    def contrast_adjustment(self, image, contrast_value=1.0):
        """Adjust contrast by scaling pixel values"""
        result = cv2.convertScaleAbs(image, alpha=contrast_value, beta=0)
        return result
    
    def complement(self, image):
        """Calculate image negative (complement)"""
        if len(image.shape) == 3:
            return 255 - image
        else:
            return 255 - image
    
    def power_law_transformation(self, image, gamma=1.0):
        """Apply power law transformation (gamma correction)"""
        normalized = image.astype(np.float32) / 255.0
        corrected = np.power(normalized, gamma)
        result = (corrected * 255).astype(np.uint8)
        return result
    
    def log_transformation(self, image, c=1):
        """Apply logarithmic transformation"""
        log_image = c * np.log(1 + image.astype(np.float32))
        log_image = np.clip(log_image, 0, 255)
        return log_image.astype(np.uint8)

import cv2
import numpy as np

class ColorOperation:
    def convert_to_grayscale(self, image):
        """Convert color image to grayscale"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def enhance_contrast(self, image, alpha=1.5):
        """Enhance contrast of color image"""
        result = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        return result
    
    def color_channel_split(self, image, channel='R'):
        """Extract specific color channel"""
        if len(image.shape) != 3:
            return image
        
        # OpenCV uses BGR format
        if channel == 'B':
            return image[:, :, 0]
        elif channel == 'G':
            return image[:, :, 1]
        elif channel == 'R':
            return image[:, :, 2]
        else:
            return image
    
    def hsv_color_space(self, image):
        """Convert to HSV color space"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return image
    
    def color_balance(self, image, red_gain=1.0, green_gain=1.0, blue_gain=1.0):
        """Adjust color balance by scaling individual channels"""
        if len(image.shape) != 3:
            return image
        
        result = image.copy().astype(np.float32)
        result[:, :, 0] *= blue_gain   # Blue channel
        result[:, :, 1] *= green_gain  # Green channel
        result[:, :, 2] *= red_gain    # Red channel
        
        # Clip values to valid range
        result = np.clip(result, 0, 255)
        return result.astype(np.uint8)

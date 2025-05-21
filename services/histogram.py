import cv2
import numpy as np
import matplotlib.pyplot as plt

class Histogram:
    """Histogram analysis and equalization operations"""
    
    def show_histogram(self, image):
        """Calculate and return histogram data"""
        if len(image.shape) == 3:
            # Color image - calculate histogram for each channel
            colors = ('b', 'g', 'r')
            histograms = {}
            for i, color in enumerate(colors):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                histograms[color] = hist.flatten()
            return histograms
        else:
            # Grayscale image
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            return {'gray': hist.flatten()}
    
    def histogram_equalization(self, image):
        """Apply histogram equalization"""
        if len(image.shape) == 3:
            # Convert to YUV color space for better results
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            return result
        else:
            return cv2.equalizeHist(image)
    
    def adaptive_histogram_equalization(self, image, clip_limit=2.0):
        """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)"""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        
        if len(image.shape) == 3:
            # Convert to LAB color space for better results
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            return result
        else:
            return clahe.apply(image)
import cv2
import numpy as np
from scipy import ndimage

class NeighborhoodProcessing:
    """Neighborhood-based filters and operations"""
    
    def mean_filter(self, image, kernel_size=5):
        """Apply mean (averaging) filter"""
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        return cv2.filter2D(image, -1, kernel)
    
    def median_filter(self, image, kernel_size=5):
        """Apply median filter for noise reduction"""
        return cv2.medianBlur(image, kernel_size)
    
    def gaussian_blur(self, image, kernel_size=5, sigma=1.0):
        """Apply Gaussian blur filter"""
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def sharpening_filter(self, image, strength=1.0):
        """Apply sharpening filter"""
        # Create sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        kernel = kernel * strength
        kernel[1, 1] = 8 * strength + 1  # Adjust center value
        
        # Apply filter
        sharpened = cv2.filter2D(image, -1, kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def unsharp_masking(self, image, radius=1, amount=1.0):
        """Apply unsharp masking for edge enhancement"""
        # Create blurred version
        if radius % 2 == 0:
            radius += 1
        blurred = cv2.GaussianBlur(image, (radius, radius), 0)
        
        # Create unsharp mask
        if len(image.shape) == 3:
            mask = image.astype(np.float32) - blurred.astype(np.float32)
            result = image.astype(np.float32) + amount * mask
        else:
            mask = image.astype(np.float32) - blurred.astype(np.float32)
            result = image.astype(np.float32) + amount * mask
        
        # Clip and convert back to uint8
        result = np.clip(result, 0, 255)
        return result.astype(np.uint8)
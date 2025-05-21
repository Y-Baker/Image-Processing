import cv2
import numpy as np
from scipy import signal, ndimage

class EdgeDetection:
    """Edge detection algorithms"""
    
    def sobel(self, image):
        """Apply Sobel edge detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = np.clip(magnitude, 0, 255)
        return magnitude.astype(np.uint8)
    
    def prewitt(self, image):
        """Apply Prewitt edge detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Prewitt kernels
        kernel_x = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1],
                            [0, 0, 0],
                            [1, 1, 1]])
        
        # Apply kernels
        grad_x = cv2.filter2D(gray.astype(np.float32), -1, kernel_x)
        grad_y = cv2.filter2D(gray.astype(np.float32), -1, kernel_y)
        
        # Calculate magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = np.clip(magnitude, 0, 255)
        return magnitude.astype(np.uint8)
    
    def canny(self, image, low_threshold=50, high_threshold=150):
        """Apply Canny edge detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return cv2.Canny(gray, low_threshold, high_threshold)
    
    def laplacian(self, image, kernel_size=3):
        """Apply Laplacian edge detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur first to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply Laplacian
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)
        laplacian = np.absolute(laplacian)
        laplacian = np.clip(laplacian, 0, 255)
        return laplacian.astype(np.uint8)
    
    def zero_crossing(self, image, threshold=0.1):
        """Detect zero crossings in Laplacian of Gaussian"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        
        # Apply Laplacian
        laplacian = cv2.Laplacian(blurred.astype(np.float32), cv2.CV_32F)
        
        # Find zero crossings
        LoG = laplacian
        edges = np.zeros(LoG.shape, dtype=np.uint8)
        
        # Check for zero crossings in 4-connectivity
        for i in range(1, LoG.shape[0] - 1):
            for j in range(1, LoG.shape[1] - 1):
                if (LoG[i, j] * LoG[i + 1, j] < 0 or
                    LoG[i, j] * LoG[i - 1, j] < 0 or
                    LoG[i, j] * LoG[i, j + 1] < 0 or
                    LoG[i, j] * LoG[i, j - 1] < 0):
                    if abs(LoG[i, j]) > threshold:
                        edges[i, j] = 255
        
        return edges
import cv2
import numpy as np

class Morphology:
    """Mathematical morphology operations"""
    
    def _get_kernel(self, kernel_size, shape='ellipse'):
        """Get morphological kernel"""
        if shape == 'ellipse':
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        elif shape == 'rect':
            return cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        elif shape == 'cross':
            return cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        else:
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    def erosion(self, image, kernel_size=5, iterations=1):
        """Apply morphological erosion"""
        kernel = self._get_kernel(kernel_size)
        return cv2.erode(image, kernel, iterations=iterations)
    
    def dilation(self, image, kernel_size=5, iterations=1):
        """Apply morphological dilation"""
        kernel = self._get_kernel(kernel_size)
        return cv2.dilate(image, kernel, iterations=iterations)
    
    def opening(self, image, kernel_size=5):
        """Apply morphological opening (erosion followed by dilation)"""
        kernel = self._get_kernel(kernel_size)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    def closing(self, image, kernel_size=5):
        """Apply morphological closing (dilation followed by erosion)"""
        kernel = self._get_kernel(kernel_size)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    def gradient(self, image, kernel_size=5):
        """Apply morphological gradient (difference between dilation and erosion)"""
        kernel = self._get_kernel(kernel_size)
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    
    def top_hat(self, image, kernel_size=5):
        """Apply top hat transformation (difference between input and opening)"""
        kernel = self._get_kernel(kernel_size)
        return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    
    def black_hat(self, image, kernel_size=5):
        """Apply black hat transformation (difference between closing and input)"""
        kernel = self._get_kernel(kernel_size)
        return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
import cv2
import numpy as np

class Segmentation:
    """Image segmentation techniques"""

    def global_thresholding(self, image, threshold=127):
        """Segment image using a global threshold value"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, result = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return result

    def otsus_method(self, image):
        """Segment image using Otsu's method (automatic thresholding)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return result

    def adaptive_thresholding(self, image, block_size=11, C=2):
        """Segment image using adaptive thresholding"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        result = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, C
        )
        return result

    def k_means_clustering(self, image, k=4):
        """Segment image using k-means clustering"""
        Z = image.reshape((-1, 3))
        Z = np.float32(Z)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)
        segmented = centers[labels.flatten()]
        result = segmented.reshape(image.shape)
        return result
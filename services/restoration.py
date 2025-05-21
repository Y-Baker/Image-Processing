
import cv2
import numpy as np
from scipy import ndimage

class Restoration:
    def __init__(self):
        """Initialize Restoration with image restoration methods."""
        pass
    
    def add_salt_pepper_noise(self, image, amount=0.05):
        """
        Add salt and pepper noise to the image.
        
        Args:
            image: Input image (grayscale or color)
            amount: Proportion of image pixels to be noisy (0.0 to 1.0)
            
        Returns:
            Image with salt and pepper noise added
        """
        noisy_image = image.copy()
        row, col = image.shape[:2]
        num_salt = np.ceil(amount * image.size * 0.5).astype(int)
        num_pepper = np.ceil(amount * image.size * 0.5).astype(int)

        # Add salt (white) noise
        coords = [np.random.randint(0, i, num_salt) for i in (row, col)]
        if len(image.shape) == 2:  # Grayscale
            noisy_image[coords[0], coords[1]] = 255
        else:  # Color
            noisy_image[coords[0], coords[1]] = [255, 255, 255]

        # Add pepper (black) noise
        coords = [np.random.randint(0, i, num_pepper) for i in (row, col)]
        if len(image.shape) == 2:
            noisy_image[coords[0], coords[1]] = 0
        else:
            noisy_image[coords[0], coords[1]] = [0, 0, 0]

        return noisy_image

    
    def remove_sp_noise_average(self, image):
        """
        Remove salt and pepper noise using average (mean) filtering.
        
        This method uses a simple average filter to reduce salt and pepper noise.
        May blur edges but effective for general noise reduction.
        
        Args:
            image: Input image with salt and pepper noise
            
        Returns:
            Image with noise reduced using average filtering
        """
        kernel = np.ones((3, 3), np.float32) / 9
        
        if len(image.shape) == 3:
            filtered = cv2.filter2D(image, -1, kernel)
        else:
            filtered = cv2.filter2D(image, -1, kernel)
        
        return filtered
    
    def remove_sp_noise_median(self, image):
        """
        Remove salt and pepper noise using median filtering.
        
        Median filter is very effective against salt and pepper noise
        as it replaces each pixel with the median of its neighborhood.
        
        Args:
            image: Input image with salt and pepper noise
            
        Returns:
            Image with noise reduced using median filtering
        """
        if len(image.shape) == 3:
            filtered = np.zeros_like(image)
            for i in range(3):
                filtered[:, :, i] = cv2.medianBlur(image[:, :, i], 5)
        else:
            filtered = cv2.medianBlur(image, 5)
        
        return filtered
    
    def remove_sp_noise_outlier(self, image):
        """
        Remove salt and pepper noise using outlier detection and replacement.
        
        This method detects outliers (pixels significantly different from neighbors)
        and replaces them with the average of non-outlier neighbors.
        
        Args:
            image: Input image with salt and pepper noise
            
        Returns:
            Image with noise reduced using outlier detection
        """
        def remove_outliers_single_channel(channel):
            channel_float = channel.astype(np.float32)
            result = channel_float.copy()
            
            kernel = np.ones((3, 3), np.uint8)
            
            h, w = channel.shape
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    neighborhood = channel_float[i-1:i+2, j-1:j+2]
                    center_pixel = neighborhood[1, 1]
                    
                    neighbors = np.concatenate([
                        neighborhood[:1, :].flatten(),
                        neighborhood[1:2, :1].flatten(),
                        neighborhood[1:2, 2:].flatten(),
                        neighborhood[2:, :].flatten()
                    ])
                    
                    neighbor_mean = np.mean(neighbors)
                    neighbor_std = np.std(neighbors)
                    
                    if abs(center_pixel - neighbor_mean) > 2 * neighbor_std:
                        result[i, j] = np.median(neighbors)
            
            return result.astype(np.uint8)
        
        if len(image.shape) == 3:
            filtered = np.zeros_like(image)
            for i in range(3):
                filtered[:, :, i] = remove_outliers_single_channel(image[:, :, i])
        else:
            filtered = remove_outliers_single_channel(image)
        
        return filtered
    
    def add_gaussian_noise(self, image, mean=0, std=0.1):
        """
        Add Gaussian noise to the image.
        
        Gaussian noise is additive noise with a normal distribution,
        commonly found in digital images due to sensor limitations.
        
        Args:
            image: Input image (grayscale or color)
            mean: Mean of the Gaussian noise (default: 0)
            std: Standard deviation of the Gaussian noise (default: 0.1)
            
        Returns:
            Image with Gaussian noise added
        """
        image_float = image.astype(np.float32) / 255.0
        
        noise = np.random.normal(mean, std, image.shape)
        
        noisy_image = image_float + noise
        
        noisy_image = np.clip(noisy_image, 0, 1)
        noisy_image = (noisy_image * 255).astype(np.uint8)
        
        return noisy_image
    
    def remove_gaussian_noise_average(self, image):
        """
        Remove Gaussian noise using average (mean) filtering.
        
        Average filtering is effective for reducing Gaussian noise
        by smoothing the image, though it may also blur edges.
        
        Args:
            image: Input image with Gaussian noise
            
        Returns:
            Image with Gaussian noise reduced using average filtering
        """
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        
        if len(image.shape) == 3:
            filtered = cv2.filter2D(image, -1, kernel)
        else:
            filtered = cv2.filter2D(image, -1, kernel)
        
        return filtered
    
    def inverse_filtering(self, image, regularization=0.01):
        """
        Apply inverse filtering for image restoration.
        
        Inverse filtering attempts to reverse the effects of a known degradation.
        Regularization is added to prevent noise amplification.
        
        Args:
            image: Input degraded image
            regularization: Regularization parameter (0.001-0.1)
            
        Returns:
            Restored image using inverse filtering
        """
        
        gaussian_blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
        
        if len(image.shape) == 3:
            restored = image + regularization * (image - gaussian_blurred)
        else:
            restored = image + regularization * (image - gaussian_blurred)
        
        return np.clip(restored, 0, 255).astype(np.uint8)
    
    def motion_blur(self, image, length=15, angle=45):
        """
        Restore image affected by motion blur.
        
        This method attempts to reverse motion blur using deconvolution techniques.
        
        Args:
            image: Input motion-blurred image
            length: Length of motion blur (1-50)
            angle: Angle of motion blur in degrees (0-180)
            
        Returns:
            Restored image with motion blur reduced
        """
        def get_motion_blur_kernel(length, angle):
            kernel = np.zeros((length, length))
            kernel[int((length-1)/2), :] = np.ones(length)
            kernel = kernel / length
            
            center = (length // 2, length // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            kernel = cv2.warpAffine(kernel, rotation_matrix, (length, length))
            
            return kernel
        
        kernel = get_motion_blur_kernel(length, angle)
        
        restored = image.copy().astype(np.float32)
        
        for _ in range(10):
            convolved = cv2.filter2D(restored, -1, kernel)
            
            convolved[convolved == 0] = 1e-10
            
            ratio = image.astype(np.float32) / convolved
            
            flipped_kernel = np.flip(kernel)
            correction = cv2.filter2D(ratio, -1, flipped_kernel)
            
            restored = restored * correction
        
        return np.clip(restored, 0, 255).astype(np.uint8)
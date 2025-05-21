import cv2
import numpy as np
from scipy import ndimage

class AdvancedFilter:
    def bilateral_filter(self, image, d=9, sigma_color=75, sigma_space=75):
        d = max(5, min(15, d))
        if d % 2 == 0:
            d += 1
        if len(image.shape) == 3:
            filtered = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        else:
            filtered = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        return filtered
    
    def non_local_means(self, image, h=10, template_window=7, search_window=21):
        if template_window % 2 == 0:
            template_window += 1
        if search_window % 2 == 0:
            search_window += 1

        h = max(3, min(15, h))
        template_window = max(7, min(21, template_window))
        search_window = max(21, min(35, search_window))
        
        if len(image.shape) == 3:
    
            denoised = cv2.fastNlMeansDenoisingColored(
                image, None, h, h, template_window, search_window
            )
        else:
            denoised = cv2.fastNlMeansDenoising(
                image, None, h, template_window, search_window
            )       
        return denoised
    
    def guided_filter(self, image, radius=8, eps=0.1):
        def guided_filter_single_channel(I, p, radius, eps):
            """Apply guided filter to single channel."""
            mean_I = cv2.boxFilter(I, cv2.CV_64F, (radius, radius))
            mean_p = cv2.boxFilter(p, cv2.CV_64F, (radius, radius))
            mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (radius, radius))
            cov_Ip = mean_Ip - mean_I * mean_p
            
            mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (radius, radius))
            var_I = mean_II - mean_I * mean_I
            
            a = cov_Ip / (var_I + eps)
            b = mean_p - a * mean_I
            
            mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
            mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))
            
            q = mean_a * I + mean_b
            return q
        
        if radius % 2 == 0:
            radius += 1
    
        image_float = image.astype(np.float64) / 255.0
        
        if len(image.shape) == 3:
    
            result = np.zeros_like(image_float)
            
    
            guide = cv2.cvtColor(image_float, cv2.COLOR_BGR2GRAY)
            
            for i in range(3):
                result[:, :, i] = guided_filter_single_channel(
                    guide, image_float[:, :, i], radius, eps
                )
        else:
            result = guided_filter_single_channel(
                image_float, image_float, radius, eps
            )
        
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return result
    
    def anisotropic_diffusion(self, image, iterations=20, kappa=50):
        def anisotropic_diffusion_single_channel(img, iterations, kappa):
            """Apply anisotropic diffusion to single channel."""
            img = img.astype(np.float64)

            img_out = img.copy()
            
            def g1(gradient_magnitude):
                return np.exp(-(gradient_magnitude / kappa) ** 2) 
            for _ in range(iterations):
                grad_n = np.roll(img_out, -1, axis=0) - img_out
                grad_s = np.roll(img_out, 1, axis=0) - img_out 
                grad_e = np.roll(img_out, -1, axis=1) - img_out
                grad_w = np.roll(img_out, 1, axis=1) - img_out 
                
                c_n = g1(np.abs(grad_n))
                c_s = g1(np.abs(grad_s))
                c_e = g1(np.abs(grad_e))
                c_w = g1(np.abs(grad_w))
                
                img_out += 0.25 * (
                    c_n * grad_n + c_s * grad_s + c_e * grad_e + c_w * grad_w
                )
            
            return img_out
        

        iterations = max(1, min(50, iterations))
        kappa = max(10, min(100, kappa))
        
        if len(image.shape) == 3:
    
            result = np.zeros_like(image, dtype=np.float64)
            
            for i in range(3):
                result[:, :, i] = anisotropic_diffusion_single_channel(
                    image[:, :, i], iterations, kappa
                )
        else:
    
            result = anisotropic_diffusion_single_channel(image, iterations, kappa)
        

        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

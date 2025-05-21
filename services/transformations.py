import numpy as np
import cv2
from typing import List, Tuple, Dict, Union, Optional, Any

class Transformation:
    TRANSFORMATIONS = {
        "Rotate": {"params": ["angle"], "range": [-180, 180]},
        "Resize": {"params": ["width", "height"], "range": [[50, 2000], [50, 2000]]},
        "Affine Transform": {"params": ["src_points", "dst_points"]},
        "Perspective Transform": {"params": ["src_points", "dst_points"]},
        "Translation": {"params": ["tx", "ty"], "range": [[-100, 100], [-100, 100]]},
        "Scaling": {"params": ["sx", "sy"], "range": [[0.1, 5.0], [0.1, 5.0]]}
    }

    def validate_parameters(self, transform_type: str, params: Dict[str, Any]) -> bool:
        if transform_type not in self.TRANSFORMATIONS:
            raise ValueError(f"Transformation '{transform_type}' is not supported.")
            
        transform_info = self.TRANSFORMATIONS[transform_type]
        required_params = transform_info["params"]

        for param in required_params:
            if param not in params:
                raise ValueError(f"Parameter '{param}' is required for '{transform_type}'.")

        if "range" in transform_info:
            for i, param in enumerate(required_params):
                if isinstance(transform_info["range"][0], list):  # Multiple ranges
                    if params[param] < transform_info["range"][i][0] or params[param] > transform_info["range"][i][1]:
                        return False
                else:  # Single range
                    if params[param] < transform_info["range"][0] or params[param] > transform_info["range"][1]:
                        return False 
        return True

    def rotate(self, image: np.ndarray, angle: float, center=None) -> np.ndarray:
        if not self.validate_parameters("Rotate", {"angle": angle}):
            raise ValueError(f"Angle {angle} is outside the allowed range [-180, 180].")
            
        if center is None:
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

        return rotated_image
    
    def resize(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        if not self.validate_parameters("Resize", {"width": width, "height": height}):
            raise ValueError(f"Width {width} or height {height} is outside the allowed range [50, 2000].")
            
        resized_image = cv2.resize(image, (width, height))
        
        return resized_image
    
    def affine_transform(self, image: np.ndarray, src_points: List[Tuple[float, float]], 
                         dst_points: List[Tuple[float, float]]) -> np.ndarray:
        if len(src_points) != 3 or len(dst_points) != 3:
            raise ValueError("Affine transformation requires exactly 3 source and 3 destination points.")
            
        src_points_array = np.float32(src_points)
        dst_points_array = np.float32(dst_points)
        
        transform_matrix = cv2.getAffineTransform(src_points_array, dst_points_array)
        transformed_image = cv2.warpAffine(image, transform_matrix, (image.shape[1], image.shape[0]))
        
        return transformed_image
    
    def perspective_transform(self, image: np.ndarray, src_points: List[Tuple[float, float]], 
                              dst_points: List[Tuple[float, float]]) -> np.ndarray:
        if len(src_points) != 4 or len(dst_points) != 4:
            raise ValueError("Perspective transformation requires exactly 4 source and 4 destination points.")
            
        src_points_array = np.float32(src_points)
        dst_points_array = np.float32(dst_points)
        
        transform_matrix = cv2.getPerspectiveTransform(src_points_array, dst_points_array)
        transformed_image = cv2.warpPerspective(image, transform_matrix, (image.shape[1], image.shape[0]))
        
        return transformed_image
    
    def translate(self, image: np.ndarray, tx: float, ty: float) -> np.ndarray:
        if not self.validate_parameters("Translation", {"tx": tx, "ty": ty}):
            raise ValueError(f"Translation values tx={tx}, ty={ty} are outside the allowed range [-100, 100].")
            
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
        
        return translated_image
    
    def scaling(self, image: np.ndarray, sx: float, sy: float) -> np.ndarray:
        if not self.validate_parameters("Scaling", {"sx": sx, "sy": sy}):
            raise ValueError(f"Scaling factors sx={sx}, sy={sy} are outside the allowed range [0.1, 5.0].")
            
        height, width = image.shape[:2]
        scaled_image = cv2.resize(image, None, fx=sx, fy=sy)
        
        return scaled_image
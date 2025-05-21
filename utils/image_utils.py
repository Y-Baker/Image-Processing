import cv2
import numpy as np
from pathlib import Path

def load_image(path: str, as_gray: bool = False) -> np.ndarray: # not used
    """
    Load an image from a given file path.

    :param path: Path to the image file
    :param as_gray: Load in grayscale if True
    :return: Image as a NumPy array
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    
    flag = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR
    image = cv2.imread(path, flag)

    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    
    return image

def save_image(image: np.ndarray, path: str) -> None: # not used
    """
    Save an image to disk.

    :param image: NumPy image array
    :param path: Path to save the image
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(path, image):
        raise IOError(f"Failed to save image to {path}")

def convert_rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert PIL image to BGR numpy array"""
    from PIL import Image
    if isinstance(image, Image.Image):
        image = np.array(image)
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR to RGB format"""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def resize_image(image: np.ndarray, width: int = None, height: int = None) -> np.ndarray: # not used
    """
    Resize image to given width or height while maintaining aspect ratio.

    :param image: Input image
    :param width: New width
    :param height: New height
    :return: Resized image
    """
    h, w = image.shape[:2]

    if width is None and height is None:
        return image

    if width is not None:
        ratio = width / w
        dim = (width, int(h * ratio))
    else:
        ratio = height / h
        dim = (int(w * ratio), height)

    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

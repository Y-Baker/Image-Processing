from services.point_operation import PointOperation
from services.color_operation import ColorOperation
from services.histogram import Histogram
from services.neighborhood import NeighborhoodProcessing
from services.restoration import Restoration
from services.segmentation import Segmentation
from services.edge_detection import EdgeDetection
from services.morphology import Morphology
from services.advanced_filters import AdvancedFilter
from services.transformations import Transformation

class FilterManager:
    def __init__(self):
        self.point_ops = PointOperation()
        self.color_ops = ColorOperation()
        self.histogram = Histogram()
        self.neighborhood = NeighborhoodProcessing()
        self.restoration = Restoration()
        self.segmentation = Segmentation()
        self.edge_detection = EdgeDetection()
        self.morphology = Morphology()
        self.advanced = AdvancedFilter()
        self.transform = Transformation()

    def get_all_services(self):
        return {
            "Point Operation": {
                "Brightness adjustment": {"params": ["brightness_value"], "range": [-100, 100]},
                "Contrast adjustment": {"params": ["contrast_value"], "range": [0.5, 3.0]},
                "Complement": {"params": []},
                "Power law transformation": {"params": ["gamma"], "range": [0.1, 3.0]},
                "Log transformation": {"params": ["c"], "range": [1, 255]}
            },
            "Color Image Operation": {
                "Convert to Grayscale": {"params": []},
                "Enhance Contrast": {"params": ["alpha"], "range": [1.0, 3.0]},
                "Color Channel Split": {"params": ["channel"], "options": ["R", "G", "B"]},
                "HSV Color Space": {"params": []},
                "Color Balance": {"params": ["red_gain", "green_gain", "blue_gain"], "range": [0.5, 2.0]}
            },
            "Image Histogram": {
                "Show Histogram": {"params": []},
                "Histogram Equalization": {"params": []},
                "Adaptive Histogram Equalization": {"params": ["clip_limit"], "range": [1.0, 4.0]}
            },
            "Neighborhood Processing": {
                "Mean Filter": {"params": ["kernel_size"], "range": [3, 15]},
                "Median Filter": {"params": ["kernel_size"], "range": [3, 15], "step": {"kernel_size": 2}},
                "Gaussian Blur": {"params": ["kernel_size", "sigma"], "range": [[3, 15], [0.5, 5.0]], "step": {"kernel_size": 2, "sigma": 0.1}},
                "Sharpening Filter": {"params": ["strength"], "range": [0.1, 2.0]},
                "Unsharp Masking": {"params": ["radius", "amount"], "range": [[1, 5], [0.5, 2.0]]}
            },
            "Image Restoration": {
                "Add Salt Pepper Noise": {"params": ["amount"], "range": [0.01, 0.2], "step": {"amount": 0.01}},
                "Remove SP Noise (Average)": {"params": []},
                "Remove SP Noise (Median)": {"params": []},
                "Remove SP Noise (Outlier)": {"params": []},
                "Add Gaussian Noise": {"params": ["mean", "std"], "range": [[-0.1, 0.1], [0.01, 0.3]], "step": {"mean": 0.01, "std": 0.01}},
                "Remove Gaussian Noise (Average)": {"params": []},
                "Inverse Filtering": {"params": ["regularization"], "range": [0.01, 1.0]},
                "Motion Blur": {"params": ["length", "angle"], "range": [[1, 20], [0, 180]], "step": {"length": 1, "angle": 5}}
            },
            "Image Segmentation": {
                "Global Thresholding": {"params": ["threshold"], "range": [0, 255], "step": {"threshold": 5}},
                "Otsu's Method": {"params": []},
                "Adaptive Thresholding": {"params": ["block_size", "C"], "range": [[3, 15], [2, 10]], "step": {"block_size": 2, "C": 2}},
                "K-Means Clustering": {"params": ["k"], "range": [2, 8]},
            },
            "Edge Detection": {
                "Sobel": {"params": []},
                "Prewitt": {"params": []},
                "Canny": {"params": ["low_threshold", "high_threshold"], "range": [[50, 150], [100, 200]]},
                "Laplacian": {"params": ["kernel_size"], "range": [1, 7], "step": {"kernel_size" : 2}},
                "Zero Crossing": {"params": ["threshold"], "range": [0.1, 1.0]}
            },
            "Mathematical Morphology": {
                "Erosion": {"params": ["kernel_size", "iterations"], "range": [[3, 15], [1, 5]]},
                "Dilation": {"params": ["kernel_size", "iterations"], "range": [[3, 15], [1, 5]]},
                "Opening": {"params": ["kernel_size"], "range": [3, 15]},
                "Closing": {"params": ["kernel_size"], "range": [3, 15]},
                "Gradient": {"params": ["kernel_size"], "range": [3, 15]},
                "Top Hat": {"params": ["kernel_size"], "range": [3, 15]},
                "Black Hat": {"params": ["kernel_size"], "range": [3, 15]}
            },
            "Advanced Filters": {
                "Bilateral Filter": {"params": ["d", "sigma_color", "sigma_space"], "range": [[5, 15], [10, 100], [10, 100]]},
                "Non-Local Means": {"params": ["h", "template_window", "search_window"], "range": [[3, 15], [7, 21], [21, 35]]},
                "Guided Filter": {"params": ["radius", "eps"], "range": [[1, 8], [0.01, 1.0]]},
                "Anisotropic Diffusion": {"params": ["iterations", "kappa"], "range": [[1, 50], [10, 100]]}
            },
            "Transformations": {
                "Rotate": {"params": ["angle"], "range": [-180, 180], "step": {"angle": 5}},
                "Resize": {"params": ["width", "height"], "range": [[50, 2000], [50, 2000]]},
                "Affine Transform": {"params": ["src_points", "dst_points"]},
                "Perspective Transform": {"params": ["src_points", "dst_points"]},
                "Translation": {"params": ["tx", "ty"], "range": [[-100, 100], [-100, 100]]},
                "Scaling": {"params": ["sx", "sy"], "range": [[0.1, 5.0], [0.1, 5.0]]}
            }
        }

    def process_image(self, image, service_category, operation, params=None):
        """
        Process image using specified service and operation
        """
        service_map = {
            "Point Operation": self.point_ops,
            "Color Image Operation": self.color_ops,
            "Image Histogram": self.histogram,
            "Neighborhood Processing": self.neighborhood,
            "Image Restoration": self.restoration,
            "Image Segmentation": self.segmentation,
            "Edge Detection": self.edge_detection,
            "Mathematical Morphology": self.morphology,
            "Advanced Filters": self.advanced,
            "Transformations": self.transform
        }
        
        service = service_map.get(service_category)
        if not service:
            raise ValueError(f"Unknown service category: {service_category}")
        
        # Convert operation name to method name
        method_name = operation.lower().replace(' ', '_').replace("'", "").replace("(", "").replace(")", "").replace("-", "_")
        method = getattr(service, method_name, None)
        
        if not method:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Call the method with parameters
        if params:
            return method(image, **params)
        else:
            return method(image)

    def get_operation_info(self, service_category, operation):
        """
        Get parameter information for a specific operation
        """
        services = self.get_all_services()
        return services.get(service_category, {}).get(operation, {})

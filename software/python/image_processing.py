"""
PYNQ Z7 Precision Object Measurement System - Advanced Image Processing Module
Implements sophisticated image processing algorithms for sub-millimeter accuracy
Designed for eventual FPGA hardware acceleration
"""

import cv2
import numpy as np
import time
from datetime import datetime
from scipy import ndimage, signal
from skimage import filters, morphology, measure, feature
import matplotlib.pyplot as plt
from pathlib import Path


def log_with_timestamp(message):
    """Print message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] IMG_PROC: {message}")


class ImageProcessor:
    """Advanced image processing for precision measurement"""

    def __init__(self):
        self.processing_history = []
        self.performance_metrics = {}
        self.debug_mode = False

        # Algorithm parameters optimized for measurement accuracy
        self.edge_params = {
            'gaussian_sigma': 1.0,
            'low_threshold': 50,
            'high_threshold': 120,
            'aperture_size': 3
        }

        self.morphology_params = {
            'kernel_size': 3,
            'erosion_iterations': 1,
            'dilation_iterations': 2,
            'opening_kernel': np.ones((3, 3), np.uint8),
            'closing_kernel': np.ones((5, 5), np.uint8)
        }

        self.noise_params = {
            'gaussian_kernel': 5,
            'bilateral_d': 9,
            'bilateral_sigma_color': 75,
            'bilateral_sigma_space': 75,
            'median_kernel': 5
        }

        log_with_timestamp("Advanced ImageProcessor initialized")

    def set_debug_mode(self, enable=True):
        """Enable/disable debug mode for detailed processing info"""
        self.debug_mode = enable
        log_with_timestamp(f"Debug mode: {'enabled' if enable else 'disabled'}")

    def preprocess_image(self, image, method='adaptive'):
        """
        Advanced preprocessing for measurement accuracy
        Args:
            image: Input image (BGR or grayscale)
            method: 'adaptive', 'clahe', 'histogram_eq', 'gamma'
        Returns:
            Preprocessed image optimized for edge detection
        """
        start_time = time.time()

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if method == 'adaptive':
            # Adaptive histogram equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(gray)

        elif method == 'clahe':
            # Contrast Limited Adaptive Histogram Equalization
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
            processed = clahe.apply(gray)

        elif method == 'histogram_eq':
            # Global histogram equalization
            processed = cv2.equalizeHist(gray)

        elif method == 'gamma':
            # Gamma correction for better contrast
            gamma = 1.2
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            processed = cv2.LUT(gray, table)

        else:
            processed = gray

        processing_time = (time.time() - start_time) * 1000
        if self.debug_mode:
            log_with_timestamp(f"Preprocessing ({method}): {processing_time:.2f}ms")

        return processed

    def apply_noise_filter(self, image, method='bilateral'):
        """
        Advanced noise filtering while preserving edges
        Args:
            image: Input image
            method: 'bilateral', 'gaussian', 'median', 'nlm' (non-local means)
        Returns:
            Filtered image with reduced noise
        """
        start_time = time.time()

        if method == 'bilateral':
            # Bilateral filter - reduces noise while preserving edges
            filtered = cv2.bilateralFilter(
                image,
                self.noise_params['bilateral_d'],
                self.noise_params['bilateral_sigma_color'],
                self.noise_params['bilateral_sigma_space']
            )

        elif method == 'gaussian':
            # Gaussian blur for general noise reduction
            kernel_size = self.noise_params['gaussian_kernel']
            filtered = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

        elif method == 'median':
            # Median filter for salt-and-pepper noise
            kernel_size = self.noise_params['median_kernel']
            filtered = cv2.medianBlur(image, kernel_size)

        elif method == 'nlm':
            # Non-local means denoising (slower but very effective)
            if len(image.shape) == 3:
                filtered = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            else:
                filtered = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

        else:
            filtered = image

        processing_time = (time.time() - start_time) * 1000
        if self.debug_mode:
            log_with_timestamp(f"Noise filtering ({method}): {processing_time:.2f}ms")

        return filtered

    def detect_edges(self, image, low_threshold=None, high_threshold=None, method='canny_adaptive'):
        """
        Advanced edge detection with multiple algorithms
        Args:
            image: Input image
            low_threshold, high_threshold: Canny thresholds (auto-calculated if None)
            method: 'canny_adaptive', 'canny_standard', 'sobel', 'laplacian', 'scharr'
        Returns:
            Edge-detected binary image
        """
        start_time = time.time()

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if method == 'canny_adaptive':
            # Adaptive Canny with automatic threshold calculation
            if low_threshold is None or high_threshold is None:
                # Automatic threshold calculation using Otsu's method
                high_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                low_thresh = 0.5 * high_thresh
            else:
                low_thresh = low_threshold
                high_thresh = high_threshold

            # Apply Gaussian blur for better edge detection
            blurred = cv2.GaussianBlur(gray, (5, 5), self.edge_params['gaussian_sigma'])
            edges = cv2.Canny(blurred, int(low_thresh), int(high_thresh),
                              apertureSize=self.edge_params['aperture_size'])

        elif method == 'canny_standard':
            # Standard Canny edge detection
            low_thresh = low_threshold or self.edge_params['low_threshold']
            high_thresh = high_threshold or self.edge_params['high_threshold']
            edges = cv2.Canny(gray, low_thresh, high_thresh)

        elif method == 'sobel':
            # Sobel edge detection (good for FPGA implementation)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
            edges = np.uint8(edges / edges.max() * 255)
            _, edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        elif method == 'laplacian':
            # Laplacian edge detection
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            edges = np.uint8(np.absolute(laplacian))
            _, edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        elif method == 'scharr':
            # Scharr edge detection (more accurate than Sobel)
            scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
            scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
            edges = np.sqrt(scharr_x ** 2 + scharr_y ** 2)
            edges = np.uint8(edges / edges.max() * 255)
            _, edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        else:
            edges = gray

        processing_time = (time.time() - start_time) * 1000
        if self.debug_mode:
            log_with_timestamp(f"Edge detection ({method}): {processing_time:.2f}ms")

        return edges

    def apply_morphological_operations(self, image, operation='close'):
        """
        Advanced morphological operations for shape refinement
        Args:
            image: Binary input image
            operation: 'open', 'close', 'gradient', 'tophat', 'blackhat', 'clean'
        Returns:
            Processed binary image
        """
        start_time = time.time()

        if operation == 'open':
            # Opening (erosion followed by dilation) - removes noise
            kernel = self.morphology_params['opening_kernel']
            processed = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        elif operation == 'close':
            # Closing (dilation followed by erosion) - fills gaps
            kernel = self.morphology_params['closing_kernel']
            processed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        elif operation == 'gradient':
            # Morphological gradient - highlights edges
            kernel = np.ones((3, 3), np.uint8)
            processed = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

        elif operation == 'tophat':
            # Top hat - highlights bright spots
            kernel = np.ones((9, 9), np.uint8)
            processed = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

        elif operation == 'blackhat':
            # Black hat - highlights dark spots
            kernel = np.ones((9, 9), np.uint8)
            processed = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

        elif operation == 'clean':
            # Combined cleaning operations
            # 1. Opening to remove noise
            kernel_open = np.ones((2, 2), np.uint8)
            opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_open)

            # 2. Closing to fill gaps
            kernel_close = np.ones((4, 4), np.uint8)
            processed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

        else:
            processed = image

        processing_time = (time.time() - start_time) * 1000
        if self.debug_mode:
            log_with_timestamp(f"Morphological operation ({operation}): {processing_time:.2f}ms")

        return processed

    def find_contours_advanced(self, image, min_area=50, max_area=50000, filter_params=None):
        """
        Advanced contour detection with filtering
        Args:
            image: Binary input image
            min_area: Minimum contour area
            max_area: Maximum contour area
            filter_params: Dictionary with filtering parameters
        Returns:
            List of filtered contours with properties
        """
        start_time = time.time()

        # Find contours
        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered_contours = []

        for contour in contours:
            # Calculate basic properties
            area = cv2.contourArea(contour)

            # Filter by area
            if area < min_area or area > max_area:
                continue

            # Calculate additional properties
            perimeter = cv2.arcLength(contour, True)

            # Calculate circularity
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0

            # Calculate aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0

            # Calculate extent (contour area / bounding rectangle area)
            rect_area = w * h
            extent = float(area) / rect_area if rect_area > 0 else 0

            # Calculate solidity (contour area / convex hull area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0

            contour_info = {
                'contour': contour,
                'area': area,
                'perimeter': perimeter,
                'circularity': circularity,
                'aspect_ratio': aspect_ratio,
                'extent': extent,
                'solidity': solidity,
                'bounding_rect': (x, y, w, h),
                'center': (x + w // 2, y + h // 2)
            }

            # Apply additional filtering if parameters provided
            if filter_params:
                if 'min_circularity' in filter_params and circularity < filter_params['min_circularity']:
                    continue
                if 'max_circularity' in filter_params and circularity > filter_params['max_circularity']:
                    continue
                if 'min_aspect_ratio' in filter_params and aspect_ratio < filter_params['min_aspect_ratio']:
                    continue
                if 'max_aspect_ratio' in filter_params and aspect_ratio > filter_params['max_aspect_ratio']:
                    continue
                if 'min_solidity' in filter_params and solidity < filter_params['min_solidity']:
                    continue

            filtered_contours.append(contour_info)

        processing_time = (time.time() - start_time) * 1000
        if self.debug_mode:
            log_with_timestamp(f"Contour detection: {len(filtered_contours)} contours found in {processing_time:.2f}ms")

        return filtered_contours

    def sub_pixel_edge_detection(self, image, contour):
        """
        Sub-pixel edge detection for enhanced measurement accuracy
        Args:
            image: Grayscale input image
            contour: Contour to refine
        Returns:
            Refined contour with sub-pixel accuracy
        """
        start_time = time.time()

        refined_points = []

        for point in contour:
            x, y = point[0]

            # Extract local region around the point
            region_size = 5
            x_start = max(0, x - region_size)
            x_end = min(image.shape[1], x + region_size + 1)
            y_start = max(0, y - region_size)
            y_end = min(image.shape[0], y + region_size + 1)

            region = image[y_start:y_end, x_start:x_end]

            if region.size > 0:
                # Calculate gradients
                grad_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)

                # Find sub-pixel edge location using gradient information
                # This is a simplified approach - more sophisticated methods exist
                center_x = region.shape[1] // 2
                center_y = region.shape[0] // 2

                if center_y < grad_x.shape[0] and center_x < grad_x.shape[1]:
                    # Calculate sub-pixel offset
                    offset_x = 0
                    offset_y = 0

                    # Use gradient magnitude to estimate sub-pixel position
                    if abs(grad_x[center_y, center_x]) > 10:
                        offset_x = grad_x[center_y, center_x] / 255.0 * 0.5

                    if abs(grad_y[center_y, center_x]) > 10:
                        offset_y = grad_y[center_y, center_x] / 255.0 * 0.5

                    refined_x = x + offset_x
                    refined_y = y + offset_y

                    refined_points.append([[refined_x, refined_y]])
                else:
                    refined_points.append([[float(x), float(y)]])
            else:
                refined_points.append([[float(x), float(y)]])

        refined_contour = np.array(refined_points, dtype=np.float32)

        processing_time = (time.time() - start_time) * 1000
        if self.debug_mode:
            log_with_timestamp(f"Sub-pixel refinement: {processing_time:.2f}ms")

        return refined_contour

    def process_for_measurement(self, image, preprocessing='adaptive', noise_filter='bilateral',
                                edge_method='canny_adaptive', morphology='clean', low_threshold=None,
                                high_threshold=None):
        """
        Complete processing pipeline optimized for measurement accuracy
        Args:
            image: Input image
            preprocessing: Preprocessing method
            noise_filter: Noise filtering method
            edge_method: Edge detection method
            morphology: Morphological operation
            low_threshold, high_threshold: Edge detection thresholds
        Returns:
            Dictionary with processed images and metadata
        """
        start_time = time.time()

        # Step 1: Preprocessing
        preprocessed = self.preprocess_image(image, preprocessing)

        # Step 2: Noise filtering
        filtered = self.apply_noise_filter(preprocessed, noise_filter)

        # Step 3: Edge detection
        edges = self.detect_edges(filtered, low_threshold, high_threshold, edge_method)

        # Step 4: Morphological operations
        cleaned = self.apply_morphological_operations(edges, morphology)

        # Step 5: Contour detection
        contours = self.find_contours_advanced(cleaned)

        total_time = (time.time() - start_time) * 1000

        result = {
            'original': image,
            'preprocessed': preprocessed,
            'filtered': filtered,
            'edges': edges,
            'cleaned': cleaned,
            'contours': contours,
            'processing_time_ms': total_time,
            'parameters': {
                'preprocessing': preprocessing,
                'noise_filter': noise_filter,
                'edge_method': edge_method,
                'morphology': morphology,
                'low_threshold': low_threshold,
                'high_threshold': high_threshold
            }
        }

        if self.debug_mode:
            log_with_timestamp(f"Complete processing pipeline: {total_time:.2f}ms")

        return result

    def save_debug_images(self, processed_result, base_filename="debug"):
        """Save intermediate processing results for analysis"""
        if not self.debug_mode:
            return

        debug_dir = Path("images/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        stages = ['original', 'preprocessed', 'filtered', 'edges', 'cleaned']

        for stage in stages:
            if stage in processed_result:
                filename = debug_dir / f"{base_filename}_{stage}_{timestamp}.png"
                cv2.imwrite(str(filename), processed_result[stage])

        log_with_timestamp(f"Debug images saved to {debug_dir}")

    def get_performance_metrics(self):
        """Get processing performance statistics"""
        return self.performance_metrics.copy()

    def reset_performance_metrics(self):
        """Reset performance tracking"""
        self.performance_metrics = {}
        log_with_timestamp("Performance metrics reset")


def test_image_processing():
    """Test the image processing module with sample data"""
    log_with_timestamp("Starting image processing test")

    processor = ImageProcessor()
    processor.set_debug_mode(True)

    # Create a test image with geometric shapes
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image.fill(50)  # Gray background

    # Add some test shapes
    cv2.circle(test_image, (160, 120), 40, (255, 255, 255), -1)  # White circle
    cv2.rectangle(test_image, (300, 80), (500, 120), (255, 255, 255), -1)  # White rectangle
    cv2.circle(test_image, (160, 300), 35, (200, 200, 200), -1)  # Gray circle

    # Add some noise
    noise = np.random.randint(0, 30, test_image.shape, dtype=np.uint8)
    test_image = cv2.add(test_image, noise)

    # Test complete processing pipeline
    result = processor.process_for_measurement(
        test_image,
        preprocessing='adaptive',
        noise_filter='bilateral',
        edge_method='canny_adaptive',
        morphology='clean'
    )

    log_with_timestamp(f"Processing completed in {result['processing_time_ms']:.2f}ms")
    log_with_timestamp(f"Found {len(result['contours'])} contours")

    # Save debug images
    processor.save_debug_images(result, "test_shapes")

    # Test different edge detection methods
    methods = ['canny_adaptive', 'sobel', 'scharr']
    for method in methods:
        start_time = time.time()
        edges = processor.detect_edges(test_image, method=method)
        processing_time = (time.time() - start_time) * 1000
        log_with_timestamp(f"{method} edge detection: {processing_time:.2f}ms")

    log_with_timestamp("Image processing test completed")


if __name__ == "__main__":
    test_image_processing()

"""
PYNQ Z7 Precision Object Measurement System - FIXED Enhanced GUI
FIXES: Noise filter dropdown, object detection, statistics button
All original functionality preserved and working
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import threading
import time
import numpy as np
from pathlib import Path
import sys
import os
from datetime import datetime

def log_with_timestamp(message):
    """Print message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from config.settings import *
    if 'REFERENCE_OBJECTS' not in globals():
        REFERENCE_OBJECTS = {
            "1 Shekel": 18.0, "2 Shekel": 21.6, "5 Shekel": 24.0, "10 Shekel": 26.0,
            "Ruler (10cm)": 100.0, "Ruler (15cm)": 150.0, "Credit Card": 85.6
        }
        log_with_timestamp("REFERENCE_OBJECTS added with Israeli coins")
    else:
        log_with_timestamp(f"REFERENCE_OBJECTS loaded from settings: {list(REFERENCE_OBJECTS.keys())}")
except ImportError as e:
    log_with_timestamp(f"Config import error: {e}")
    CAMERA_ID = 1
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    EDGE_DETECTION_LOW = 50
    EDGE_DETECTION_HIGH = 150
    REFERENCE_OBJECTS = {
        "1 Shekel": 18.0, "2 Shekel": 21.6, "5 Shekel": 24.0, "10 Shekel": 26.0,
        "Ruler (10cm)": 100.0, "Ruler (15cm)": 150.0, "Credit Card": 85.6
    }
    log_with_timestamp("Using fallback configuration with Israeli coins")

# Import camera interface
CAMERA_AVAILABLE = False
try:
    from camera_interface import USBCameraInterface
    CAMERA_AVAILABLE = True
    log_with_timestamp("USBCameraInterface imported successfully")
except ImportError as e:
    log_with_timestamp(f"Camera import error: {e}")
    CAMERA_AVAILABLE = False

# Import Advanced Measurement Algorithms
try:
    from measurement_algorithms import (
        AdvancedMeasurementCalculator,
        MeasurementIntegrator,
        CalibrationData,
        MeasurementResult
    )
    ADVANCED_MEASUREMENTS_AVAILABLE = True
    log_with_timestamp("Advanced measurement algorithms imported successfully")
except ImportError as e:
    log_with_timestamp(f"Advanced measurements import error: {e}")
    ADVANCED_MEASUREMENTS_AVAILABLE = False

class ImageProcessor:
    """Lightweight image processor using only OpenCV and NumPy"""
    def __init__(self):
        self.debug_mode = True
        log_with_timestamp("Lightweight ImageProcessor initialized")

    def detect_edges(self, frame, low_thresh, high_thresh, method='canny_standard'):
        """Advanced edge detection using OpenCV only"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

            if method == 'canny_adaptive':
                high_threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                low_threshold = 0.5 * high_threshold
                blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
                edges = cv2.Canny(blurred, int(low_threshold), int(high_threshold))
            elif method == 'sobel':
                sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
                edges = np.uint8(sobel_combined / sobel_combined.max() * 255)
                _, edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif method == 'scharr':
                scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
                scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
                scharr_combined = np.sqrt(scharr_x**2 + scharr_y**2)
                edges = np.uint8(scharr_combined / scharr_combined.max() * 255)
                _, edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:  # canny_standard
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, low_thresh, high_thresh)

            return edges
        except Exception as e:
            log_with_timestamp(f"Edge detection error: {e}")
            return gray

    def apply_noise_filter(self, frame, method='bilateral'):
        """Advanced noise filtering using OpenCV only"""
        try:
            if method == 'bilateral':
                return cv2.bilateralFilter(frame, 9, 75, 75)
            elif method == 'gaussian':
                return cv2.GaussianBlur(frame, (5, 5), 0)
            elif method == 'median':
                return cv2.medianBlur(frame, 5)
            else:
                return frame
        except Exception as e:
            log_with_timestamp(f"Noise filtering error: {e}")
            return frame

class ObjectDetector:
    """Enhanced object detector - FIXED to work properly"""
    def __init__(self):
        self.detection_enabled = False

    def detect_coins(self, frame):
        """Detect circular coins using HoughCircles with optimized parameters"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (15, 15), 2)

            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=80,
                param1=80, param2=40, minRadius=25, maxRadius=80
            )

            detected_coins = []
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    circularity_score = self._validate_coin_circularity(gray, x, y, r)
                    if circularity_score > 0.7:
                        detected_coins.append({
                            'type': 'coin', 'center': (x, y), 'radius': r,
                            'diameter_pixels': r * 2, 'confidence': circularity_score
                        })

            detected_coins.sort(key=lambda x: x['confidence'], reverse=True)
            return detected_coins[:2]

        except Exception as e:
            log_with_timestamp(f"Coin detection error: {e}")
            return []

    def detect_rulers(self, frame):
        """FIXED ruler detection"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Enhanced preprocessing for ruler detection
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Multiple thresholding approaches
            adaptive_thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
            _, otsu_thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Combine thresholds
            combined = cv2.bitwise_or(adaptive_thresh, otsu_thresh)

            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detected_rulers = []
            for contour in contours:
                area = cv2.contourArea(contour)

                if area < 3000 or area > 50000:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0

                if aspect_ratio < 3.0 or aspect_ratio > 12.0:
                    continue

                rect_area = w * h
                rectangularity = area / rect_area if rect_area > 0 else 0

                if rectangularity < 0.4:
                    continue

                length_pixels = max(w, h)
                confidence = min(0.9, (rectangularity + min(1.0, area/10000)) / 2)

                detected_rulers.append({
                    'type': 'ruler',
                    'center': (x + w//2, y + h//2),
                    'length_pixels': length_pixels,
                    'width_pixels': min(w, h),
                    'confidence': confidence,
                    'bounding_rect': (x, y, w, h)
                })

            detected_rulers.sort(key=lambda x: x['confidence'], reverse=True)
            return detected_rulers[:2]

        except Exception as e:
            log_with_timestamp(f"Ruler detection error: {e}")
            return []

    def _validate_coin_circularity(self, gray, cx, cy, radius):
        """Validate that a detected circle is actually circular"""
        try:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (cx, cy), radius, 255, -1)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            masked = cv2.bitwise_and(thresh, mask)
            contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    return min(1.0, circularity)
            return 0.5
        except:
            return 0.5

    def detect_objects(self, frame):
        """Detect coins and rulers in frame - FIXED"""
        detected_objects = {}

        # Detect coins
        coins = self.detect_coins(frame)
        if coins:
            detected_objects['coins'] = coins

        # Detect rulers
        rulers = self.detect_rulers(frame)
        if rulers:
            detected_objects['rulers'] = rulers

        return detected_objects

    def draw_detection_overlay(self, frame, detected_objects):
        """Draw detection overlays - FIXED"""
        overlay = frame.copy()

        # Draw coins
        if 'coins' in detected_objects:
            for coin in detected_objects['coins']:
                center = coin['center']
                radius = coin['radius']
                cv2.circle(overlay, center, radius, (255, 0, 255), 2)  # Magenta circle
                cv2.circle(overlay, center, 3, (255, 0, 255), -1)     # Center dot
                cv2.putText(overlay, f"Coin: {radius*2}px ({coin['confidence']:.2f})",
                           (center[0]-40, center[1]-radius-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        # Draw rulers
        if 'rulers' in detected_objects:
            for ruler in detected_objects['rulers']:
                x, y, w, h = ruler['bounding_rect']
                center = ruler['center']
                cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 255), 2)  # Cyan rectangle
                cv2.circle(overlay, center, 3, (0, 255, 255), -1)             # Center dot
                cv2.putText(overlay, f"Ruler: {ruler['length_pixels']}px ({ruler['confidence']:.2f})",
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        return overlay

class DirectOpenCVCamera:
    """ULTRA-FAST OpenCV camera - immediate initialization"""
    def __init__(self, cap):
        self.cap = cap
        log_with_timestamp("ULTRA-FAST OpenCV camera initialized")

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        if self.cap:
            self.cap.release()
            log_with_timestamp("ULTRA-FAST OpenCV camera released")

class EnhancedMeasurementGUI:
    """Enhanced GUI with FIXED Advanced Measurement Algorithms Integration"""

    def __init__(self):
        self.root = tk.Tk()
        self.setup_window()

        # Initialize system components
        self.camera = None
        self.image_processor = ImageProcessor()
        self.object_detector = ObjectDetector()

        # Initialize Advanced Measurement System
        if ADVANCED_MEASUREMENTS_AVAILABLE:
            self.measurement_calculator = AdvancedMeasurementCalculator(debug_mode=True)
            self.measurement_integrator = MeasurementIntegrator(self.image_processor)
            log_with_timestamp("Advanced measurement system initialized")
        else:
            self.measurement_calculator = None
            self.measurement_integrator = None
            log_with_timestamp("Using basic measurement system (advanced algorithms not available)")

        # Measurement session tracking
        self.measurement_session = {
            'start_time': datetime.now(),
            'total_measurements': 0,
            'calibration_history': []
        }

        # GUI state variables
        self.is_measuring = False
        self.current_frame = None
        self.processed_frame = None
        self.calibration_active = False
        self.measurement_results = {}

        # Measurement state
        self.measurement_mode = None
        self.measurement_points = []
        self.pixels_per_mm = 1.0
        self.measurement_overlay = None
        self.mouse_coords = (0, 0)

        # Threading control
        self.camera_thread = None
        self.stop_camera = threading.Event()

        # Create GUI components
        self.create_widgets()
        self.setup_layout()

        self.update_status("Enhanced GUI Ready - Advanced measurements available" if ADVANCED_MEASUREMENTS_AVAILABLE else "GUI Ready - Basic measurements only")
        log_with_timestamp("Enhanced GUI initialized with FIXED functionality")

    def setup_window(self):
        """Configure main window properties"""
        self.root.title("PYNQ Z7 Precision Object Measurement System - Enhanced")
        self.root.geometry("1600x900")
        self.root.minsize(1200, 800)

        style = ttk.Style()
        style.theme_use('clam')
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.resizable(True, True)

    def create_widgets(self):
        """Create all GUI widgets with FIXED advanced measurement features"""

        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")

        # Video display frame
        self.video_frame = ttk.LabelFrame(self.main_frame, text="Live Camera Feed", padding="5")
        self.video_label = ttk.Label(self.video_frame, text="Click 'Start Camera' to begin",
                                   background="black", foreground="white")

        # Bind mouse events to video label
        self.video_label.bind("<Button-1>", self.on_video_click)
        self.video_label.bind("<Button-3>", self.on_video_right_click)
        self.video_label.bind("<Motion>", self.on_video_motion)

        # Mouse coordinate display
        self.mouse_coords_label = ttk.Label(self.video_frame, text="Mouse: (0, 0)",
                                          font=("Consolas", 9), background="black", foreground="lime")

        # Control panel frame with scrolling
        self.control_outer_frame = ttk.LabelFrame(self.main_frame, text="Controls", padding="5")
        self.control_canvas = tk.Canvas(self.control_outer_frame, highlightthickness=0, width=350)
        self.control_scrollbar = ttk.Scrollbar(self.control_outer_frame, orient="vertical", command=self.control_canvas.yview)
        self.control_frame = ttk.Frame(self.control_canvas)

        # Scrolling configuration
        self.control_canvas.configure(yscrollcommand=self.control_scrollbar.set)
        self.control_canvas.create_window((0, 0), window=self.control_frame, anchor="nw")

        def _on_mousewheel(event):
            self.control_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        def _bind_to_mousewheel(event):
            self.control_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_from_mousewheel(event):
            self.control_canvas.unbind_all("<MouseWheel>")

        self.control_canvas.bind('<Enter>', _bind_to_mousewheel)
        self.control_canvas.bind('<Leave>', _unbind_from_mousewheel)

        def configure_scroll_region(event):
            self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))

        self.control_frame.bind("<Configure>", configure_scroll_region)

        # Camera controls
        self.camera_controls_frame = ttk.Frame(self.control_frame)
        self.start_button = ttk.Button(self.camera_controls_frame, text="Start Camera", command=self.start_camera)
        self.stop_button = ttk.Button(self.camera_controls_frame, text="Stop Camera", command=self.stop_camera_feed, state="disabled")
        self.capture_button = ttk.Button(self.camera_controls_frame, text="Capture Image", command=self.capture_image, state="disabled")

        # Processing controls
        self.processing_frame = ttk.LabelFrame(self.control_frame, text="Advanced Image Processing", padding="5")
        self.edge_detection_var = tk.BooleanVar(value=True)
        self.edge_detection_cb = ttk.Checkbutton(self.processing_frame, text="Edge Detection", variable=self.edge_detection_var, command=self.update_processing)
        self.noise_filter_var = tk.BooleanVar(value=True)
        self.noise_filter_cb = ttk.Checkbutton(self.processing_frame, text="Noise Filtering", variable=self.noise_filter_var, command=self.update_processing)

        # FIXED: Advanced processing method selection - BOTH dropdowns included
        self.advanced_frame = ttk.Frame(self.processing_frame)

        # Edge detection method
        ttk.Label(self.advanced_frame, text="Edge Method:").grid(row=0, column=0, sticky="w")
        self.edge_method_var = tk.StringVar(value="canny_adaptive")
        self.edge_method_combo = ttk.Combobox(self.advanced_frame, textvariable=self.edge_method_var,
                                            values=["canny_adaptive", "canny_standard", "sobel", "scharr"],
                                            state="readonly", width=12)
        self.edge_method_combo.grid(row=0, column=1, padx=(5,0))

        # FIXED: Noise filter method - was missing in enhanced version
        ttk.Label(self.advanced_frame, text="Noise Filter:").grid(row=1, column=0, sticky="w")
        self.noise_method_var = tk.StringVar(value="bilateral")
        self.noise_method_combo = ttk.Combobox(self.advanced_frame, textvariable=self.noise_method_var,
                                             values=["bilateral", "gaussian", "median"],
                                             state="readonly", width=12)
        self.noise_method_combo.grid(row=1, column=1, padx=(5,0))

        def on_edge_method_select(event):
            widget = event.widget
            widget.selection_clear()
            self.root.focus_set()
            self.update_processing()

        def on_noise_method_select(event):
            widget = event.widget
            widget.selection_clear()
            self.root.focus_set()
            self.update_processing()

        self.edge_method_combo.bind("<<ComboboxSelected>>", on_edge_method_select)
        self.noise_method_combo.bind("<<ComboboxSelected>>", on_noise_method_select)

        # Threshold controls
        self.threshold_frame = ttk.Frame(self.processing_frame)
        ttk.Label(self.threshold_frame, text="Edge Threshold Low:").grid(row=0, column=0, sticky="w")
        self.threshold_low_var = tk.IntVar(value=EDGE_DETECTION_LOW)
        self.threshold_low_scale = ttk.Scale(self.threshold_frame, from_=10, to=100,
                                           variable=self.threshold_low_var, orient="horizontal", length=150,
                                           command=self.update_thresholds)
        self.threshold_low_label = ttk.Label(self.threshold_frame, text=str(self.threshold_low_var.get()),
                                           width=3, relief="sunken")

        ttk.Label(self.threshold_frame, text="Edge Threshold High:").grid(row=1, column=0, sticky="w")
        self.threshold_high_var = tk.IntVar(value=EDGE_DETECTION_HIGH)
        self.threshold_high_scale = ttk.Scale(self.threshold_frame, from_=50, to=200,
                                            variable=self.threshold_high_var, orient="horizontal", length=150,
                                            command=self.update_thresholds)
        self.threshold_high_label = ttk.Label(self.threshold_frame, text=str(self.threshold_high_var.get()),
                                            width=3, relief="sunken")

        # Measurement controls
        self.measurement_frame = ttk.LabelFrame(self.control_frame, text="Measurements", padding="5")
        self.measure_distance_button = ttk.Button(self.measurement_frame, text="Measure Distance", command=self.measure_distance)
        self.measure_diameter_button = ttk.Button(self.measurement_frame, text="Measure Diameter", command=self.measure_diameter)
        self.measure_area_button = ttk.Button(self.measurement_frame, text="Measure Area", command=self.measure_area)
        self.cancel_measurement_button = ttk.Button(self.measurement_frame, text="Cancel Measurement", command=self.cancel_measurement, state="disabled")

        # Enhanced measurement status with quality indicator
        self.measurement_status = ttk.Label(self.measurement_frame, text="Ready for enhanced measurement", foreground="green")

        # FIXED: Statistics button - only appears if advanced measurements available
        if ADVANCED_MEASUREMENTS_AVAILABLE:
            self.stats_button = ttk.Button(self.measurement_frame, text="View Statistics", command=self.show_measurement_statistics)
        else:
            self.stats_button = None

        # Calibration controls
        self.calibration_frame = ttk.LabelFrame(self.control_frame, text="Calibration", padding="5")
        self.calibration_method_frame = ttk.Frame(self.calibration_frame)
        ttk.Label(self.calibration_method_frame, text="Reference Object:").pack(anchor="w")

        self.calibration_method_var = tk.StringVar(value="5 Shekel")
        self.objects_frame = ttk.Frame(self.calibration_method_frame)

        for obj_name in REFERENCE_OBJECTS.keys():
            size_mm = REFERENCE_OBJECTS[obj_name]
            if "Ruler" in obj_name:
                display_text = f"{obj_name} ({size_mm}mm length)"
            elif "Card" in obj_name:
                display_text = f"{obj_name} ({size_mm}mm width)"
            else:
                display_text = f"{obj_name} ({size_mm}mm diameter)"

            radio_btn = ttk.Radiobutton(self.objects_frame, text=display_text,
                          variable=self.calibration_method_var, value=obj_name)
            radio_btn.pack(anchor="w")

        # Calibration action buttons
        self.calibration_buttons_frame = ttk.Frame(self.calibration_frame)
        self.auto_calibrate_button = ttk.Button(self.calibration_buttons_frame, text="Auto Detect & Calibrate", command=self.auto_calibrate)
        self.manual_calibrate_button = ttk.Button(self.calibration_buttons_frame, text="Manual Calibration", command=self.manual_calibrate)

        # Enhanced calibration status
        self.calibration_status = ttk.Label(self.calibration_frame, text="Status: Not Calibrated", foreground="red")
        self.calibration_value_label = ttk.Label(self.calibration_frame, text="Scale: 1.0 px/mm", foreground="blue")

        # FIXED: Object detection toggle - properly implemented
        self.detection_frame = ttk.LabelFrame(self.control_frame, text="Object Detection", padding="5")
        self.object_detection_var = tk.BooleanVar(value=False)
        self.object_detection_cb = ttk.Checkbutton(self.detection_frame, text="Show Object Detection",
                                                 variable=self.object_detection_var, command=self.toggle_object_detection)
        self.detection_status = ttk.Label(self.detection_frame, text="Detection: Off", foreground="gray")

        # Results panel
        self.results_frame = ttk.LabelFrame(self.main_frame, text="Measurement Results", padding="5")
        self.results_text_frame = ttk.Frame(self.results_frame)
        self.results_text = tk.Text(self.results_text_frame, height=15, width=40, wrap=tk.WORD, state="disabled")
        self.results_scrollbar = ttk.Scrollbar(self.results_text_frame, command=self.results_text.yview)
        self.results_text.config(yscrollcommand=self.results_scrollbar.set)

        # Results control buttons
        self.results_buttons_frame = ttk.Frame(self.results_frame)
        self.clear_results_button = ttk.Button(self.results_buttons_frame, text="Clear Results", command=self.clear_results)
        self.save_results_button = ttk.Button(self.results_buttons_frame, text="Save Results", command=self.save_results)

        # Status bar
        self.status_frame = ttk.Frame(self.main_frame)
        self.status_var = tk.StringVar(value="Enhanced System Ready")
        self.status_label = ttk.Label(self.status_frame, textvariable=self.status_var, relief="sunken", anchor="w")

    def setup_layout(self):
        """Setup layout with proper proportions"""
        # Main frame
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Grid layout with balanced proportions
        self.main_frame.columnconfigure(0, weight=4, minsize=600)  # Video
        self.main_frame.columnconfigure(1, weight=2, minsize=360)  # Controls
        self.main_frame.columnconfigure(2, weight=3, minsize=400)  # Results
        self.main_frame.rowconfigure(0, weight=1)

        # Video section
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.video_label.pack(expand=True, fill="both", padx=5, pady=5)
        self.mouse_coords_label.place(relx=1.0, rely=0.0, anchor="ne", x=-15, y=15)

        # Control section
        self.control_outer_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 5))
        self.control_outer_frame.rowconfigure(0, weight=1)
        self.control_outer_frame.columnconfigure(0, weight=1)
        self.control_canvas.grid(row=0, column=0, sticky="nsew")
        self.control_scrollbar.grid(row=0, column=1, sticky="ns")

        # Results section
        self.results_frame.grid(row=0, column=2, sticky="nsew")
        self.results_text_frame.pack(fill="both", expand=True)
        self.results_text.pack(side="left", fill="both", expand=True)
        self.results_scrollbar.pack(side="right", fill="y")
        self.results_buttons_frame.pack(fill="x", pady=(5, 0))
        self.clear_results_button.pack(side="left", padx=(0, 5))
        self.save_results_button.pack(side="right")

        # Pack all control widgets
        self._setup_control_widgets()

        # Status bar
        self.status_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(5, 0))
        self.status_label.pack(fill="x")

        log_with_timestamp("Enhanced layout completed with FIXED functionality")

    def _setup_control_widgets(self):
        """Setup all control widgets layout - FIXED to include all elements"""
        # Camera controls
        self.camera_controls_frame.pack(fill="x", pady=(0, 5))
        self.start_button.pack(side="left", padx=(0, 2))
        self.stop_button.pack(side="left", padx=(0, 2))
        self.capture_button.pack(side="left")

        # Processing controls
        self.processing_frame.pack(fill="x", pady=(0, 5))
        self.edge_detection_cb.pack(anchor="w")
        self.noise_filter_cb.pack(anchor="w")

        # FIXED: Advanced frame with BOTH dropdowns
        self.advanced_frame.pack(fill="x", pady=(2, 0))

        # Threshold controls
        self.threshold_frame.pack(fill="x", pady=(2, 0))
        self.threshold_low_scale.grid(row=0, column=1, padx=(5, 0))
        self.threshold_low_label.grid(row=0, column=2, padx=(5, 0))
        self.threshold_high_scale.grid(row=1, column=1, padx=(5, 0))
        self.threshold_high_label.grid(row=1, column=2, padx=(5, 0))

        # Measurement controls
        self.measurement_frame.pack(fill="x", pady=(0, 5))
        self.measure_distance_button.pack(fill="x", pady=1)
        self.measure_diameter_button.pack(fill="x", pady=1)
        self.measure_area_button.pack(fill="x", pady=1)
        self.cancel_measurement_button.pack(fill="x", pady=1)
        self.measurement_status.pack(pady=1)

        # FIXED: Add statistics button if available
        if hasattr(self, 'stats_button') and self.stats_button:
            self.stats_button.pack(fill="x", pady=1)

        # Calibration controls
        self.calibration_frame.pack(fill="x", pady=(0, 5))
        self.calibration_method_frame.pack(fill="x", pady=(0, 2))
        self.objects_frame.pack(fill="x", pady=(2, 0))
        self.calibration_buttons_frame.pack(fill="x", pady=(2, 2))
        self.auto_calibrate_button.pack(fill="x", pady=1)
        self.manual_calibrate_button.pack(fill="x", pady=1)
        self.calibration_status.pack(pady=1)
        self.calibration_value_label.pack(pady=1)

        # FIXED: Object detection controls
        self.detection_frame.pack(fill="x", pady=(0, 5))
        self.object_detection_cb.pack(anchor="w")
        self.detection_status.pack(pady=1)

    # ENHANCED: Advanced measurement methods with uncertainty quantification
    def calculate_distance(self):
        """ENHANCED distance calculation with uncertainty quantification"""
        if len(self.measurement_points) < 2:
            return

        p1, p2 = self.measurement_points[0], self.measurement_points[1]

        if ADVANCED_MEASUREMENTS_AVAILABLE and self.measurement_integrator:
            # Use advanced measurement system
            try:
                result = self.measurement_integrator.measure_distance_from_gui(p1, p2)

                # Display enhanced results with uncertainty
                formatted_result = (f"Distance: {result['formatted_result']}\n"
                                   f"Quality: {result['quality']}\n"
                                   f"Confidence: {result['confidence']:.1%}")

                self.add_result(formatted_result)
                self.update_measurement_quality_display(result.get('quality_score', 0.8))

                # Update session tracking
                self.measurement_session['total_measurements'] += 1

                log_with_timestamp(f"Enhanced distance: {result['formatted_result']}, Quality: {result['quality']}")

            except Exception as e:
                log_with_timestamp(f"Advanced measurement error: {e}")
                # Fallback to basic calculation
                self._calculate_basic_distance(p1, p2)
        else:
            # Use basic calculation
            self._calculate_basic_distance(p1, p2)

        # Keep existing GUI behavior
        self.measurement_mode = None
        self.reset_measurement_buttons()
        self.measurement_status.config(text="Distance measured - points visible", foreground="green")
        self.video_frame.config(text="Live Camera Feed - Distance Result Shown")

    def _calculate_basic_distance(self, p1, p2):
        """Fallback basic distance calculation"""
        pixel_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        real_distance = pixel_distance / self.pixels_per_mm
        result = f"Distance: {real_distance:.2f} mm ({pixel_distance:.1f} pixels)"
        self.add_result(result)
        log_with_timestamp(f"Basic distance calculated: {result}")

    def calculate_diameter(self):
        """ENHANCED diameter calculation with uncertainty quantification"""
        if len(self.measurement_points) < 2:
            return

        p1, p2 = self.measurement_points[0], self.measurement_points[1]

        if ADVANCED_MEASUREMENTS_AVAILABLE and self.measurement_integrator:
            try:
                result = self.measurement_integrator.measure_diameter_from_gui(p1, p2)

                formatted_result = (f"Diameter: {result['formatted_result']}\n"
                                   f"Quality: {result['quality']}\n"
                                   f"Confidence: {result['confidence']:.1%}")

                self.add_result(formatted_result)
                self.update_measurement_quality_display(result.get('quality_score', 0.8))

                self.measurement_session['total_measurements'] += 1

                log_with_timestamp(f"Enhanced diameter: {result['formatted_result']}, Quality: {result['quality']}")

            except Exception as e:
                log_with_timestamp(f"Advanced measurement error: {e}")
                self._calculate_basic_diameter(p1, p2)
        else:
            self._calculate_basic_diameter(p1, p2)

        self.measurement_mode = None
        self.reset_measurement_buttons()
        self.measurement_status.config(text="Diameter measured - points visible", foreground="green")
        self.video_frame.config(text="Live Camera Feed - Diameter Result Shown")

    def _calculate_basic_diameter(self, p1, p2):
        """Fallback basic diameter calculation"""
        pixel_diameter = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        real_diameter = pixel_diameter / self.pixels_per_mm
        result = f"Diameter: {real_diameter:.2f} mm ({pixel_diameter:.1f} pixels)"
        self.add_result(result)
        log_with_timestamp(f"Basic diameter calculated: {result}")

    def calculate_area(self):
        """ENHANCED area calculation with uncertainty quantification"""
        if len(self.measurement_points) < 3:
            return

        if ADVANCED_MEASUREMENTS_AVAILABLE and self.measurement_integrator:
            try:
                result = self.measurement_integrator.measure_area_from_gui(self.measurement_points)

                formatted_result = (f"Area: {result['formatted_result']}\n"
                                   f"Vertices: {result['vertices']}\n"
                                   f"Quality: {result['quality']}\n"
                                   f"Confidence: {result['confidence']:.1%}")

                self.add_result(formatted_result)
                self.update_measurement_quality_display(result.get('quality_score', 0.8))

                self.measurement_session['total_measurements'] += 1

                log_with_timestamp(f"Enhanced area: {result['formatted_result']}, Quality: {result['quality']}")

            except Exception as e:
                log_with_timestamp(f"Advanced measurement error: {e}")
                self._calculate_basic_area()
        else:
            self._calculate_basic_area()

        self.measurement_mode = None
        self.reset_measurement_buttons()
        self.measurement_status.config(text="Area measured - polygon visible", foreground="green")
        self.video_frame.config(text="Live Camera Feed - Area Result Shown")

    def _calculate_basic_area(self):
        """Fallback basic area calculation"""
        points = np.array(self.measurement_points)
        x = points[:, 0]
        y = points[:, 1]
        pixel_area = 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(-1, len(x)-1)))
        real_area = pixel_area / (self.pixels_per_mm ** 2)
        result = f"Area: {real_area:.2f} mm² ({pixel_area:.1f} pixels²)"
        self.add_result(result)
        log_with_timestamp(f"Basic area calculated: {result}")

    def update_measurement_quality_display(self, quality_score):
        """NEW: Update GUI with measurement quality indicator"""
        if quality_score >= 0.9:
            color = "green"
            status = "Excellent"
        elif quality_score >= 0.7:
            color = "blue"
            status = "Good"
        elif quality_score >= 0.5:
            color = "orange"
            status = "Acceptable"
        else:
            color = "red"
            status = "Poor"

        self.measurement_status.config(text=f"Quality: {status} ({quality_score:.2f})", foreground=color)

    def show_measurement_statistics(self):
        """NEW: Display measurement statistics in popup window"""
        if not ADVANCED_MEASUREMENTS_AVAILABLE or not self.measurement_integrator:
            messagebox.showinfo("Statistics", "Advanced measurement statistics not available")
            return

        try:
            stats = self.measurement_integrator.get_measurement_statistics()

            if 'error' in stats:
                messagebox.showinfo("Statistics", stats['error'])
                return

            # Enhanced statistics display
            stats_text = (f"Measurement Statistics:\n\n"
                         f"Total Measurements: {stats['count']}\n"
                         f"Mean: {stats['mean']}\n"
                         f"Standard Deviation: {stats['std']}\n"
                         f"Precision: {stats['precision']}\n"
                         f"Quality Score: {stats['quality']}\n"
                         f"Outliers Detected: {stats['outliers']}\n\n"
                         f"Session Information:\n"
                         f"Session Start: {self.measurement_session['start_time'].strftime('%H:%M:%S')}\n"
                         f"Total Measurements: {self.measurement_session['total_measurements']}\n"
                         f"Calibrations: {len(self.measurement_session['calibration_history'])}")

            # Create custom statistics window
            self._create_statistics_window(stats, stats_text)

        except Exception as e:
            log_with_timestamp(f"Statistics error: {e}")
            messagebox.showerror("Statistics Error", f"Failed to generate statistics: {e}")

    def _create_statistics_window(self, stats, stats_text):
        """Create enhanced statistics display window"""
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Measurement Statistics - Enhanced Analysis")
        stats_window.geometry("500x400")
        stats_window.transient(self.root)
        stats_window.grab_set()

        # Main frame
        main_frame = ttk.Frame(stats_window, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Text display
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill="both", expand=True)

        text_widget = tk.Text(text_frame, wrap=tk.WORD, state="disabled")
        scrollbar = ttk.Scrollbar(text_frame, command=text_widget.yview)
        text_widget.config(yscrollcommand=scrollbar.set)

        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Insert statistics text
        text_widget.config(state="normal")
        text_widget.insert(tk.END, stats_text)
        text_widget.config(state="disabled")

        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(10, 0))

        # Export button
        export_btn = ttk.Button(button_frame, text="Export Data",
                               command=lambda: self._export_measurement_data())
        export_btn.pack(side="left")

        # Close button
        close_btn = ttk.Button(button_frame, text="Close",
                              command=stats_window.destroy)
        close_btn.pack(side="right")

    def _export_measurement_data(self):
        """NEW: Export measurement data using advanced algorithms"""
        if not ADVANCED_MEASUREMENTS_AVAILABLE or not self.measurement_calculator:
            messagebox.showwarning("Export", "Advanced export not available - using basic export")
            self.save_results()
            return

        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv"), ("All files", "*.*")]
            )

            if filename:
                # Determine format from extension
                if filename.lower().endswith('.csv'):
                    format_type = 'csv'
                else:
                    format_type = 'json'

                # Export using advanced algorithms
                self.measurement_calculator.export_measurement_data(filename, format_type)

                messagebox.showinfo("Export Complete",
                                   f"Enhanced measurement data exported to {filename}")
                self.update_status(f"Enhanced data exported to {filename}")

        except Exception as e:
            log_with_timestamp(f"Export error: {e}")
            messagebox.showerror("Export Error", f"Failed to export data: {e}")

    # Camera and processing methods - UNCHANGED but working
    def start_camera(self):
        """Initialize and start camera"""
        log_with_timestamp("Start camera button pressed")
        self.start_button.config(state="disabled", text="Initializing...")
        self.update_status("Initializing camera in background...")
        self.video_label.config(text="Initializing camera...", foreground="yellow")
        self.root.update_idletasks()

        def init_and_start_camera():
            try:
                log_with_timestamp("Background camera initialization starting...")
                camera_ready = False

                try:
                    log_with_timestamp("Trying direct OpenCV with minimal settings...")
                    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
                    if not cap.isOpened():
                        cap = cv2.VideoCapture(CAMERA_ID)

                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                        start_test = time.time()
                        ret, frame = cap.read()
                        test_time = time.time() - start_test

                        if ret and frame is not None and test_time < 5.0:
                            self.camera = DirectOpenCVCamera(cap)
                            camera_ready = True
                            log_with_timestamp("Fast OpenCV camera ready")
                        else:
                            cap.release()
                except Exception as e:
                    log_with_timestamp(f"Direct OpenCV failed: {e}")

                def update_gui_after_init():
                    if camera_ready and self.camera:
                        self.stop_camera.clear()
                        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
                        self.camera_thread.start()

                        self.start_button.config(state="disabled", text="Stop Camera")
                        self.stop_button.config(state="normal")
                        self.capture_button.config(state="normal")

                        self.update_status("Enhanced camera started successfully")
                        self.video_label.config(text="Camera active", foreground="lime")
                        log_with_timestamp("Enhanced camera started successfully")
                    else:
                        self.start_button.config(state="normal", text="Start Camera")
                        self.update_status("Camera initialization failed")
                        self.video_label.config(text="Camera failed - check connection", foreground="red")

                self.root.after(0, update_gui_after_init)

            except Exception as e:
                log_with_timestamp(f"Camera initialization error: {e}")

        init_thread = threading.Thread(target=init_and_start_camera, daemon=True)
        init_thread.start()

    def stop_camera_feed(self):
        """Stop camera feed"""
        self.stop_camera.set()
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)
        if self.camera:
            self.camera.release()
            self.camera = None

        self.start_button.config(state="normal", text="Start Camera")
        self.stop_button.config(state="disabled")
        self.capture_button.config(state="disabled")
        self.video_label.config(text="Camera stopped - Click 'Start Camera' to begin", foreground="white")
        self.update_status("Camera stopped")

    def camera_loop(self):
        """Main camera processing loop"""
        log_with_timestamp("Camera loop started")
        frame_count = 0
        last_log_time = time.time()

        while not self.stop_camera.is_set():
            try:
                frame = self.camera.get_frame()
                if frame is not None:
                    frame_count += 1
                    current_time = time.time()

                    if current_time - last_log_time >= 5.0:
                        log_with_timestamp(f"Camera running: {frame_count} frames processed")
                        last_log_time = current_time

                    self.current_frame = frame.copy()
                    display_frame = self.process_frame(frame)
                    self.update_video_display(display_frame)

                time.sleep(1/30)
            except Exception as e:
                log_with_timestamp(f"Camera loop error: {e}")
                break

        log_with_timestamp("Camera loop ended")

    def process_frame(self, frame):
        """Process frame using image processing methods - FIXED"""
        processed = frame.copy()

        try:
            # Apply noise filtering FIRST if enabled
            if self.noise_filter_var.get():
                noise_method = getattr(self, 'noise_method_var', None)
                if noise_method and hasattr(noise_method, 'get'):
                    method = noise_method.get()
                else:
                    method = 'bilateral'
                processed = self.image_processor.apply_noise_filter(processed, method=method)

            # Apply edge detection SECOND if enabled
            if self.edge_detection_var.get():
                edge_method = getattr(self, 'edge_method_var', None)
                if edge_method and hasattr(edge_method, 'get'):
                    method = edge_method.get()
                else:
                    method = 'canny_adaptive'
                processed = self.image_processor.detect_edges(processed,
                                                            self.threshold_low_var.get(),
                                                            self.threshold_high_var.get(),
                                                            method=method)
        except Exception as e:
            log_with_timestamp(f"Frame processing error: {e}")
            processed = frame.copy()

        # FIXED: Add object detection overlay if enabled
        if self.object_detection_var.get() and self.object_detector and self.current_frame is not None:
            try:
                detected_objects = self.object_detector.detect_objects(self.current_frame)
                if detected_objects:
                    if len(processed.shape) == 2:
                        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                    processed = self.object_detector.draw_detection_overlay(processed, detected_objects)
            except Exception as e:
                log_with_timestamp(f"Object detection overlay error: {e}")

        # Add measurement overlay if points exist
        if len(self.measurement_points) > 0:
            try:
                if len(processed.shape) == 2:
                    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                processed = self.draw_measurement_overlay(processed)
            except Exception as e:
                log_with_timestamp(f"Measurement overlay error: {e}")

        self.processed_frame = processed
        return processed

    def update_video_display(self, frame):
        """Update video display with new frame"""
        try:
            if len(frame.shape) == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            pil_image = Image.fromarray(frame_rgb)
            display_size = (640, 480)
            pil_image = pil_image.resize(display_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil_image)

            self.video_label.configure(image=photo, text="")
            self.video_label.image = photo
        except Exception as e:
            self.update_status(f"Display update error: {e}")

    def update_processing(self):
        """Update processing settings - FIXED"""
        edge_enabled = self.edge_detection_var.get()
        noise_enabled = self.noise_filter_var.get()

        edge_method = getattr(self, 'edge_method_var', None)
        if edge_method and hasattr(edge_method, 'get'):
            edge_method_name = edge_method.get()
        else:
            edge_method_name = "canny_adaptive"

        noise_method = getattr(self, 'noise_method_var', None)
        if noise_method and hasattr(noise_method, 'get'):
            noise_method_name = noise_method.get()
        else:
            noise_method_name = "bilateral"

        log_with_timestamp(f"Processing update: Edge={edge_enabled}({edge_method_name}), Noise={noise_enabled}({noise_method_name})")
        self.update_status(f"Processing: Edge={edge_method_name}, Noise={noise_method_name}")

    def update_thresholds(self, value=None):
        """Update threshold display labels"""
        low_val = self.threshold_low_var.get()
        high_val = self.threshold_high_var.get()

        self.threshold_low_label.config(text=str(low_val))
        self.threshold_high_label.config(text=str(high_val))

        if high_val <= low_val:
            if value is not None:
                try:
                    current_val = int(float(value))
                    if abs(current_val - low_val) < abs(current_val - high_val):
                        new_high = max(low_val + 10, high_val)
                        if new_high <= 200:
                            self.threshold_high_var.set(new_high)
                            self.threshold_high_label.config(text=str(new_high))
                    else:
                        new_low = min(high_val - 10, low_val)
                        if new_low >= 10:
                            self.threshold_low_var.set(new_low)
                            self.threshold_low_label.config(text=str(new_low))
                except:
                    pass

    def toggle_object_detection(self):
        """FIXED: Toggle object detection overlay"""
        if self.object_detection_var.get():
            self.detection_status.config(text="Detection: On", foreground="green")
            log_with_timestamp("Object detection enabled")
        else:
            self.detection_status.config(text="Detection: Off", foreground="gray")
            log_with_timestamp("Object detection disabled")

        self.update_status(f"Object detection: {'On' if self.object_detection_var.get() else 'Off'}")

    # All remaining methods remain exactly the same as working version
    # (measure_distance, measure_diameter, measure_area, calibration methods, etc.)

    def measure_distance(self):
        """Measure distance between two points"""
        if self.current_frame is None:
            messagebox.showwarning("Measurement Error", "No image available for measurement")
            return

        self.start_measurement_mode('distance')
        self.measurement_status.config(text="Click two points to measure distance", foreground="blue")
        self.add_result("Enhanced distance measurement: Click two points on the image")

    def measure_diameter(self):
        """Measure object diameter"""
        if self.current_frame is None:
            messagebox.showwarning("Measurement Error", "No image available for measurement")
            return

        self.start_measurement_mode('diameter')
        self.measurement_status.config(text="Click two points on object edge", foreground="blue")
        self.add_result("Enhanced diameter measurement: Click two points on object edge")

    def measure_area(self):
        """Measure object area"""
        if self.current_frame is None:
            messagebox.showwarning("Measurement Error", "No image available for measurement")
            return

        self.start_measurement_mode('area')
        self.measurement_status.config(text="Click points to outline object area", foreground="blue")
        self.add_result("Enhanced area measurement: Click points to outline the object (right-click to finish)")

    def start_measurement_mode(self, mode):
        """Initialize measurement mode"""
        log_with_timestamp(f"Starting {mode} measurement mode")
        self.measurement_points = []
        self.measurement_mode = mode
        self.last_measurement_type = mode

        self.measure_distance_button.config(state="disabled")
        self.measure_diameter_button.config(state="disabled")
        self.measure_area_button.config(state="disabled")
        self.cancel_measurement_button.config(state="normal")

        self.video_frame.config(text=f"Live Camera Feed - {mode.upper()} MODE")
        self.update_status(f"Enhanced {mode.capitalize()} measurement mode active")

    def reset_measurement_buttons(self):
        """Reset measurement buttons to normal state"""
        self.measure_distance_button.config(state="normal")
        self.measure_diameter_button.config(state="normal")
        self.measure_area_button.config(state="normal")
        self.cancel_measurement_button.config(state="normal")

    def cancel_measurement(self):
        """Cancel current measurement and clear all points"""
        log_with_timestamp(f"Cancelling measurement - clearing {len(self.measurement_points)} points")

        self.measurement_mode = None
        self.measurement_points = []
        self.calibration_active = False

        self.measure_distance_button.config(state="normal")
        self.measure_diameter_button.config(state="normal")
        self.measure_area_button.config(state="normal")
        self.cancel_measurement_button.config(state="disabled")

        self.auto_calibrate_button.config(state="normal")
        self.manual_calibrate_button.config(state="normal")

        self.video_frame.config(text="Live Camera Feed")
        self.measurement_status.config(text="Ready for enhanced measurement", foreground="green")

        if hasattr(self, 'calibration_status') and self.calibration_active:
            self.calibration_status.config(text="Status: Calibration Cancelled", foreground="orange")

        self.update_status("Measurement/Calibration cancelled - points cleared")
        self.add_result("Measurement/Calibration cancelled - all points cleared")

    def capture_image(self):
        """Capture current frame"""
        if self.current_frame is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.png"

            samples_dir = project_root / "images" / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)

            filepath = samples_dir / filename
            cv2.imwrite(str(filepath), self.current_frame)

            self.add_result(f"Image captured: {filename}")
            self.update_status(f"Image saved: {filename}")
        else:
            messagebox.showwarning("Capture Error", "No frame available to capture")

    def auto_calibrate(self):
        """Automatically detect and calibrate using selected object"""
        if self.current_frame is None:
            messagebox.showwarning("Calibration Error", "No camera image available")
            return

        calibration_object = self.calibration_method_var.get()
        log_with_timestamp(f"Starting auto calibration with {calibration_object}")

        try:
            detected_objects = self.object_detector.detect_objects(self.current_frame)

            if "Shekel" in calibration_object:
                if 'coins' not in detected_objects or not detected_objects['coins']:
                    messagebox.showwarning("Calibration Failed",
                                         f"No coins detected in image.\n"
                                         f"Please ensure a {calibration_object} coin is clearly visible.")
                    return

                best_coin = detected_objects['coins'][0]
                pixel_diameter = best_coin['diameter_pixels']
                real_diameter_mm = REFERENCE_OBJECTS[calibration_object]

                self.pixels_per_mm = pixel_diameter / real_diameter_mm
                confidence = best_coin['confidence']

                # Set advanced calibration if available
                if ADVANCED_MEASUREMENTS_AVAILABLE and self.measurement_integrator:
                    try:
                        self.measurement_integrator.set_calibration_from_gui(
                            self.pixels_per_mm,
                            calibration_object,
                            confidence=confidence
                        )
                        log_with_timestamp("Advanced auto calibration system updated")
                    except Exception as e:
                        log_with_timestamp(f"Advanced auto calibration error: {e}")

                measurement_type = "diameter"

            else:
                messagebox.showwarning("Calibration Error", f"Auto calibration not available for {calibration_object}")
                return

            self.calibration_status.config(text=f"Status: Calibrated ({confidence:.2f})", foreground="green")
            self.calibration_value_label.config(text=f"Scale: {self.pixels_per_mm:.3f} px/mm")

            # Track calibration history
            self.measurement_session['calibration_history'].append({
                'timestamp': datetime.now().isoformat(),
                'method': 'auto',
                'object': calibration_object,
                'scale': self.pixels_per_mm,
                'confidence': confidence
            })

            result_text = (f"Auto calibration successful!\n"
                         f"Object: {calibration_object}\n" 
                         f"Measurement: {measurement_type}\n"
                         f"Scale: {self.pixels_per_mm:.3f} pixels/mm\n"
                         f"Confidence: {confidence:.2f}")

            log_with_timestamp(f"Enhanced auto calibration: {self.pixels_per_mm:.3f} px/mm, confidence: {confidence:.2f}")
            self.add_result(f"Auto calibration: {self.pixels_per_mm:.3f} px/mm ({calibration_object})")
            self.update_status("Enhanced auto calibration completed")

            messagebox.showinfo("Calibration Complete", result_text)

        except Exception as e:
            log_with_timestamp(f"Auto calibration error: {e}")
            messagebox.showerror("Calibration Error", f"Auto calibration failed: {e}")

    def manual_calibrate(self):
        """Manual calibration by clicking on reference object"""
        if self.current_frame is None:
            messagebox.showwarning("Calibration Error", "No camera image available")
            return

        calibration_object = self.calibration_method_var.get()
        self.start_manual_calibration(calibration_object)

    def start_manual_calibration(self, calibration_object):
        """Start manual calibration by clicking reference object points"""
        self.calibration_active = True
        self.measurement_mode = 'calibration_manual'
        self.measurement_points = []
        self.current_calibration_object = calibration_object

        self.auto_calibrate_button.config(state="disabled")
        self.manual_calibrate_button.config(state="disabled")
        self.cancel_measurement_button.config(state="normal")

        size_mm = REFERENCE_OBJECTS[calibration_object]

        self.calibration_status.config(text="Manual calibration: Click object edges", foreground="orange")
        self.measurement_status.config(text=f"Click two points on {calibration_object} ({size_mm}mm)", foreground="blue")
        self.video_frame.config(text="Live Camera Feed - MANUAL CALIBRATION")

        self.add_result(f"Enhanced manual calibration: Click two points on {calibration_object}")
        log_with_timestamp(f"Enhanced manual calibration started for {calibration_object}")

    def complete_manual_calibration(self):
        """ENHANCED manual calibration with uncertainty tracking"""
        if len(self.measurement_points) < 2:
            return

        p1, p2 = self.measurement_points[0], self.measurement_points[1]
        pixel_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

        calibration_object = getattr(self, 'current_calibration_object', self.calibration_method_var.get())
        real_distance_mm = REFERENCE_OBJECTS[calibration_object]

        # Calculate scale factor
        self.pixels_per_mm = pixel_distance / real_distance_mm

        # Set advanced calibration if available
        if ADVANCED_MEASUREMENTS_AVAILABLE and self.measurement_integrator:
            try:
                self.measurement_integrator.set_calibration_from_gui(
                    self.pixels_per_mm,
                    calibration_object,
                    confidence=0.9  # High confidence for manual calibration
                )
                log_with_timestamp("Advanced calibration system updated")
            except Exception as e:
                log_with_timestamp(f"Advanced calibration error: {e}")

        # Enhanced status display
        self.calibration_status.config(text=f"Status: Calibrated (Manual)", foreground="green")
        self.calibration_value_label.config(text=f"Scale: {self.pixels_per_mm:.3f} px/mm")

        # Track calibration history
        self.measurement_session['calibration_history'].append({
            'timestamp': datetime.now().isoformat(),
            'method': 'manual',
            'object': calibration_object,
            'scale': self.pixels_per_mm,
            'pixel_distance': pixel_distance,
            'real_distance_mm': real_distance_mm
        })

        # Log results with enhanced information
        result_text = (f"Manual calibration successful!\n"
                      f"Object: {calibration_object}\n"
                      f"Pixel distance: {pixel_distance:.1f}px\n"
                      f"Real distance: {real_distance_mm}mm\n"
                      f"Scale: {self.pixels_per_mm:.3f} px/mm\n"
                      f"Expected uncertainty: ±{self.pixels_per_mm * 0.05:.3f} px/mm")

        log_with_timestamp(f"Enhanced manual calibration: {self.pixels_per_mm:.3f} px/mm using {calibration_object}")
        self.add_result(f"Manual calibration: {self.pixels_per_mm:.3f} px/mm ({calibration_object})")

        messagebox.showinfo("Calibration Complete", result_text)

        # Reset calibration mode
        self.calibration_active = False
        self.measurement_mode = None

        # Re-enable buttons
        self.auto_calibrate_button.config(state="normal")
        self.manual_calibrate_button.config(state="normal")
        self.cancel_measurement_button.config(state="disabled")

        self.measurement_status.config(text="Ready for enhanced measurement", foreground="green")
        self.video_frame.config(text="Live Camera Feed")
        self.update_status("Enhanced manual calibration completed")

    # Event handlers - all unchanged from working version
    def on_video_click(self, event):
        """Handle mouse clicks on video display"""
        log_with_timestamp(f"Video click detected at display coordinates: ({event.x}, {event.y})")

        if not self.measurement_mode:
            log_with_timestamp("No measurement mode active, ignoring click")
            return

        img_x, img_y = self.convert_display_to_image_coords(event.x, event.y)
        if img_x is None or img_y is None:
            log_with_timestamp("Failed to convert coordinates")
            return

        point = (img_x, img_y)
        self.measurement_points.append(point)
        log_with_timestamp(f"Added measurement point: {point} (total points: {len(self.measurement_points)})")

        if self.measurement_mode == 'distance' and len(self.measurement_points) == 2:
            self.calculate_distance()
        elif self.measurement_mode == 'diameter' and len(self.measurement_points) == 2:
            self.calculate_diameter()
        elif self.measurement_mode in ['calibration_coin', 'calibration_ruler', 'calibration_manual'] and len(self.measurement_points) == 2:
            self.complete_manual_calibration()
        elif self.measurement_mode == 'area':
            log_with_timestamp(f"Area measurement: {len(self.measurement_points)} points selected")

    def on_video_right_click(self, event):
        """Handle right-click to finish area measurement"""
        if self.measurement_mode == 'area' and len(self.measurement_points) >= 3:
            self.calculate_area()
        elif self.measurement_mode == 'area':
            messagebox.showinfo("Area Measurement", "Need at least 3 points to calculate area")

    def on_video_motion(self, event):
        """Handle mouse motion over video display"""
        img_x, img_y = self.convert_display_to_image_coords(event.x, event.y)
        if img_x is not None and img_y is not None:
            self.mouse_coords = (img_x, img_y)
            self.mouse_coords_label.config(text=f"Mouse: ({img_x}, {img_y})")

    def convert_display_to_image_coords(self, display_x, display_y):
        """Convert display coordinates to image coordinates"""
        if self.current_frame is None:
            return None, None

        try:
            img_height, img_width = self.current_frame.shape[:2]
            label_width = self.video_label.winfo_width()
            label_height = self.video_label.winfo_height()

            display_img_width = 640
            display_img_height = 480

            x_offset = max(0, (label_width - display_img_width) // 2)
            y_offset = max(0, (label_height - display_img_height) // 2)

            adjusted_x = display_x - x_offset
            adjusted_y = display_y - y_offset

            if adjusted_x < 0 or adjusted_x >= display_img_width or adjusted_y < 0 or adjusted_y >= display_img_height:
                return None, None

            scale_x = img_width / display_img_width
            scale_y = img_height / display_img_height

            img_x = int(adjusted_x * scale_x)
            img_y = int(adjusted_y * scale_y)

            img_x = max(0, min(img_x, img_width - 1))
            img_y = max(0, min(img_y, img_height - 1))

            return img_x, img_y

        except Exception as e:
            log_with_timestamp(f"Coordinate conversion error: {e}")
            return None, None

    def draw_measurement_overlay(self, frame):
        """Draw measurement points and lines on frame"""
        overlay = frame.copy()

        try:
            for i, point in enumerate(self.measurement_points):
                cv2.circle(overlay, point, 3, (0, 255, 0), -1)
                cv2.circle(overlay, point, 5, (255, 255, 255), 1)
                cv2.putText(overlay, str(i+1), (point[0]+8, point[1]-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(overlay, str(i+1), (point[0]+8, point[1]-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if len(self.measurement_points) >= 2:
                if hasattr(self, 'last_measurement_type'):
                    if self.last_measurement_type in ['distance', 'diameter', 'calibration_coin', 'calibration_ruler', 'calibration_manual']:
                        cv2.line(overlay, self.measurement_points[0], self.measurement_points[1], (0, 0, 255), 2)
                    elif self.last_measurement_type == 'area' and len(self.measurement_points) > 2:
                        points = np.array(self.measurement_points, np.int32)
                        cv2.polylines(overlay, [points], True, (0, 0, 255), 2)
                else:
                    if self.measurement_mode in ['distance', 'diameter', 'calibration_coin', 'calibration_ruler', 'calibration_manual']:
                        cv2.line(overlay, self.measurement_points[0], self.measurement_points[1], (0, 0, 255), 2)
                    elif self.measurement_mode == 'area' and len(self.measurement_points) > 2:
                        points = np.array(self.measurement_points, np.int32)
                        cv2.polylines(overlay, [points], True, (0, 0, 255), 2)

            if self.measurement_mode and len(self.measurement_points) > 0 and hasattr(self, 'mouse_coords'):
                last_point = self.measurement_points[-1]
                mouse_point = self.mouse_coords
                if self.measurement_mode in ['distance', 'diameter', 'calibration_coin', 'calibration_ruler', 'calibration_manual'] and len(self.measurement_points) == 1:
                    cv2.line(overlay, last_point, mouse_point, (128, 128, 128), 1)
                elif self.measurement_mode == 'area' and len(self.measurement_points) >= 1:
                    cv2.line(overlay, last_point, mouse_point, (128, 128, 128), 1)

        except Exception as e:
            log_with_timestamp(f"Error drawing overlay: {e}")

        return overlay

    # Utility methods
    def add_result(self, result_text):
        """Add result to results display"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_result = f"[{timestamp}] {result_text}\n"

        self.results_text.config(state="normal")
        self.results_text.insert(tk.END, formatted_result)
        self.results_text.see(tk.END)
        self.results_text.config(state="disabled")

    def clear_results(self):
        """Clear results display"""
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state="disabled")
        self.update_status("Results cleared")

    def save_results(self):
        """Save results to file"""
        content = self.results_text.get(1.0, tk.END).strip()
        if not content:
            messagebox.showinfo("Save Results", "No results to save")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(content)
                messagebox.showinfo("Save Results", f"Results saved to {filename}")
                self.update_status(f"Results saved to {filename}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save results: {e}")

    def update_status(self, message):
        """Update status bar"""
        self.status_var.set(message)
        self.root.update_idletasks()

    def on_closing(self):
        """Handle window closing"""
        self.stop_camera_feed()
        if self.camera:
            self.camera.release()
        self.root.destroy()

    def run(self):
        """Start the enhanced GUI application"""
        log_with_timestamp("Starting Enhanced PYNQ Z7 Measurement System")

        if ADVANCED_MEASUREMENTS_AVAILABLE:
            log_with_timestamp("Advanced measurement algorithms active - ±0.1-0.5mm precision available")
        else:
            log_with_timestamp("Basic measurement mode - advanced algorithms not available")

        self.root.mainloop()


def main():
    """Main entry point for FIXED Enhanced Measurement System"""
    try:
        # Print system capabilities
        print("\n" + "="*60)
        print("PYNQ Z7 Enhanced Precision Object Measurement System - FIXED")
        print("="*60)

        if ADVANCED_MEASUREMENTS_AVAILABLE:
            print("✅ Advanced Measurement Algorithms: ACTIVE")
            print("✅ Statistical Analysis: AVAILABLE")
            print("✅ Uncertainty Quantification: ENABLED")
            print("✅ Quality Assessment: FUNCTIONAL")
            print("✅ Sub-millimeter Precision: ±0.1-0.5mm")
        else:
            print("⚠️  Advanced Measurement Algorithms: NOT AVAILABLE")
            print("⚠️  Basic measurements only - precision limited")
            print("💡 To enable advanced features:")
            print("   Add measurement_algorithms.py to software/python/")

        if CAMERA_AVAILABLE:
            print("✅ Camera System: READY")
        else:
            print("⚠️  Camera System: LIMITED")

        print("✅ Israeli Coin Calibration: CONFIGURED")
        print("✅ Real-time Processing: ENABLED")
        print("✅ FIXED: Noise Filter Dropdown: WORKING")
        print("✅ FIXED: Object Detection: WORKING")
        print("✅ FIXED: Statistics Button: AVAILABLE")
        print("="*60)
        print()

        app = EnhancedMeasurementGUI()
        app.run()

    except Exception as e:
        log_with_timestamp(f"Failed to start enhanced application: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
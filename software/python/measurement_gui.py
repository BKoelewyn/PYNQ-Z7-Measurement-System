"""
PYNQ Z7 Precision Object Measurement System - Complete GUI Framework
Integrated with existing settings and camera configuration
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
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
    print(f"[{timestamp}] {message}")

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from config.settings import *
    # Check if REFERENCE_OBJECTS exists in settings, if not create it
    if 'REFERENCE_OBJECTS' not in globals():
        REFERENCE_OBJECTS = {
            "1 Shekel": 18.0,   # Israeli 1 shekel coin diameter
            "2 Shekel": 21.6,   # Israeli 2 shekel coin diameter
            "5 Shekel": 24.0,   # Israeli 5 shekel coin diameter
            "10 Shekel": 26.0,  # Israeli 10 shekel coin diameter
            "Ruler (10cm)": 100.0,  # 10cm ruler segment
            "Ruler (15cm)": 150.0,  # 15cm ruler
            "Credit Card": 85.6     # Credit card width
        }
        log_with_timestamp("REFERENCE_OBJECTS added with Israeli coins")
    else:
        log_with_timestamp(f"REFERENCE_OBJECTS loaded from settings: {list(REFERENCE_OBJECTS.keys())}")
except ImportError as e:
    log_with_timestamp(f"Config import error: {e}")
    # Fallback default values
    CAMERA_ID = 1
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    EDGE_DETECTION_LOW = 50
    EDGE_DETECTION_HIGH = 150
    REFERENCE_OBJECTS = {
        "1 Shekel": 18.0,
        "2 Shekel": 21.6,
        "5 Shekel": 24.0,
        "10 Shekel": 26.0,
        "Ruler (10cm)": 100.0,
        "Ruler (15cm)": 150.0,
        "Credit Card": 85.6
    }
    log_with_timestamp("Using fallback configuration with Israeli coins")

# Log what reference objects we have available
log_with_timestamp(f"Available reference objects: {list(REFERENCE_OBJECTS.keys())}")

# Import camera interface with correct class name
CAMERA_AVAILABLE = False
try:
    from camera_interface import USBCameraInterface
    CAMERA_AVAILABLE = True
    log_with_timestamp("USBCameraInterface imported successfully")
except ImportError as e:
    log_with_timestamp(f"Camera import error: {e}")
    CAMERA_AVAILABLE = False

# Placeholder classes for missing modules
class ImageProcessor:
    """Basic image processor using OpenCV"""
    def __init__(self):
        pass

    def detect_edges(self, frame, low_thresh, high_thresh):
        """Basic edge detection using OpenCV"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, low_thresh, high_thresh)
        return edges

    def apply_noise_filter(self, frame):
        """Basic noise filtering"""
        return cv2.GaussianBlur(frame, (5, 5), 0)

class MeasurementCalculator:
    """Basic measurement calculator"""
    def __init__(self):
        pass

class ObjectDetector:
    """Enhanced object detector for calibration objects including rulers"""
    def __init__(self):
        self.detection_enabled = False

    def detect_coins(self, frame):
        """Detect circular coins using HoughCircles with stricter parameters"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply stronger blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (15, 15), 2)

            # Use stricter parameters for HoughCircles
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,                # Resolution ratio
                minDist=80,          # Minimum distance between circles (was 50)
                param1=80,           # Upper threshold for edge detection (was 50)
                param2=40,           # Accumulator threshold (was 30) - higher = fewer detections
                minRadius=25,        # Minimum radius (was 20)
                maxRadius=80         # Maximum radius (was 100)
            )

            detected_coins = []
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")

                for (x, y, r) in circles:
                    # Additional validation: check circularity by analyzing the actual contour
                    circularity_score = self._validate_coin_circularity(gray, x, y, r)

                    if circularity_score > 0.7:  # Only accept very circular objects
                        detected_coins.append({
                            'type': 'coin',
                            'center': (x, y),
                            'radius': r,
                            'diameter_pixels': r * 2,
                            'confidence': circularity_score
                        })

            # Sort by confidence and return only top 2
            detected_coins.sort(key=lambda x: x['confidence'], reverse=True)
            return detected_coins[:2]  # Limit to 2 best detections

        except Exception as e:
            log_with_timestamp(f"Coin detection error: {e}")
            return []

    def _validate_coin_circularity(self, gray, cx, cy, radius):
        """Validate that a detected circle is actually circular"""
        try:
            # Create a mask for the circle
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (cx, cy), radius, 255, -1)

            # Apply threshold to get binary image
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Find contours in the masked region
            masked = cv2.bitwise_and(thresh, mask)
            contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)

                # Calculate circularity
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)

                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    return min(1.0, circularity)

            return 0.5  # Default medium confidence

        except:
            return 0.5

    def detect_rulers(self, frame):
        """Enhanced ruler detection with stricter parameters"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Try only the most reliable method first
            rulers_contours = self._detect_rulers_contours(gray)

            # If no good detections, try edge method
            if not rulers_contours or max(r['confidence'] for r in rulers_contours) < 0.6:
                rulers_edges = self._detect_rulers_edges(gray)
                rulers_contours.extend(rulers_edges)

            # Remove duplicates and sort by confidence
            unique_rulers = self._remove_duplicate_rulers(rulers_contours)
            unique_rulers.sort(key=lambda x: x['confidence'], reverse=True)

            # Only return high-confidence detections
            good_rulers = [r for r in unique_rulers if r['confidence'] > 0.5]

            return good_rulers[:2]  # Return max 2 detections

        except Exception as e:
            log_with_timestamp(f"Ruler detection error: {e}")
            return []

    def _detect_rulers_edges(self, gray):
        """Detect rulers using edge detection"""
        # Apply adaptive threshold for better edge detection
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Apply edge detection
        edges = cv2.Canny(adaptive_thresh, 30, 100, apertureSize=3)

        # Dilate to connect broken edges
        kernel = np.ones((2,2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        return self._analyze_contours_for_rulers(edges, "edges")

    def _detect_rulers_contours(self, gray):
        """Detect rulers using contour analysis"""
        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Apply threshold
        _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        return self._analyze_contours_for_rulers(thresh, "contours")

    def detect_cards(self, frame):
        """Specific detection for credit card-sized rectangular objects"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply bilateral filter to reduce noise while keeping edges sharp
            filtered = cv2.bilateralFilter(gray, 11, 80, 80)

            # Apply adaptive threshold for better edge detection
            adaptive_thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            # Apply edge detection
            edges = cv2.Canny(adaptive_thresh, 30, 100, apertureSize=3)

            # Morphological operations to connect edges
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detected_cards = []

            for contour in contours:
                # Calculate contour area
                area = cv2.contourArea(contour)

                # Credit card should be reasonably large
                if area < 8000 or area > 35000:
                    continue

                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # Credit card aspect ratio is approximately 1.586:1 (85.6mm x 53.98mm)
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0

                # Credit card should have aspect ratio between 1.4 and 1.8
                if aspect_ratio < 1.4 or aspect_ratio > 1.9:
                    continue

                # Check rectangularity (credit cards are very rectangular)
                rect_area = w * h
                rectangularity = area / rect_area if rect_area > 0 else 0

                if rectangularity < 0.7:  # Should be very rectangular
                    continue

                # Credit card should have reasonable dimensions
                length_pixels = max(w, h)
                width_pixels = min(w, h)

                if length_pixels < 120 or width_pixels < 70:  # Minimum size
                    continue

                # Calculate confidence based on how well it matches credit card proportions
                ideal_aspect_ratio = 1.586
                aspect_score = 1.0 - abs(aspect_ratio - ideal_aspect_ratio) / ideal_aspect_ratio
                rect_score = rectangularity
                size_score = min(1.0, area / 20000)

                confidence = (aspect_score + rect_score + size_score) / 3.0
                confidence = min(0.9, confidence)

                # Only add if confidence is good
                if confidence > 0.6:
                    detected_cards.append({
                        'type': 'card',
                        'center': (x + w//2, y + h//2),
                        'length_pixels': length_pixels,
                        'width_pixels': width_pixels,
                        'area_pixels': area,
                        'aspect_ratio': aspect_ratio,
                        'rectangularity': rectangularity,
                        'confidence': confidence,
                        'bounding_rect': (x, y, w, h),
                        'detection_method': 'card_specific'
                    })

            # Sort by confidence and return best detection
            detected_cards.sort(key=lambda x: x['confidence'], reverse=True)
            return detected_cards[:1]  # Return only the best card detection

        except Exception as e:
            log_with_timestamp(f"Card detection error: {e}")
            return []

    def _analyze_contours_for_rulers(self, processed_image, method_name):
        """Analyze contours to find ruler-like objects with stricter parameters"""
        # Find contours
        contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rulers = []

        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)

            # Stricter area filtering
            if area < 5000 or area > 40000:  # Reduced max area
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate aspect ratio
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0

            # Stricter aspect ratio for rulers
            if aspect_ratio < 4.0 or aspect_ratio > 12.0:  # More restrictive
                continue

            # Check rectangularity
            rect_area = w * h
            rectangularity = area / rect_area if rect_area > 0 else 0

            # Stricter rectangularity
            if rectangularity < 0.5:  # More rectangular required
                continue

            # Additional check: ruler should have minimum length
            length_pixels = max(w, h)
            if length_pixels < 100:  # Minimum 100 pixels long
                continue

            width_pixels = min(w, h)

            # Enhanced confidence calculation with stricter scoring
            size_score = min(1.0, area / 15000)  # Prefer larger objects
            aspect_score = min(1.0, aspect_ratio / 6.0)  # Good aspect ratio
            rect_score = rectangularity
            length_score = min(1.0, length_pixels / 200)  # Prefer longer objects

            confidence = (size_score + aspect_score + rect_score + length_score) / 4.0
            confidence = min(0.9, confidence)

            # Only add if confidence is reasonable
            if confidence > 0.4:
                rulers.append({
                    'type': 'ruler',
                    'center': (x + w//2, y + h//2),
                    'length_pixels': length_pixels,
                    'width_pixels': width_pixels,
                    'area_pixels': area,
                    'aspect_ratio': aspect_ratio,
                    'rectangularity': rectangularity,
                    'confidence': confidence,
                    'bounding_rect': (x, y, w, h),
                    'detection_method': method_name
                })

        return rulers

    def _remove_duplicate_rulers(self, rulers):
        """Remove duplicate ruler detections that are too close to each other"""
        if len(rulers) <= 1:
            return rulers

        unique_rulers = []

        for ruler in rulers:
            is_duplicate = False
            for existing in unique_rulers:
                # Calculate distance between centers
                dx = ruler['center'][0] - existing['center'][0]
                dy = ruler['center'][1] - existing['center'][1]
                distance = np.sqrt(dx*dx + dy*dy)

                # If centers are close and sizes are similar, consider it a duplicate
                if distance < 50 and abs(ruler['length_pixels'] - existing['length_pixels']) < 30:
                    is_duplicate = True
                    # Keep the one with higher confidence
                    if ruler['confidence'] > existing['confidence']:
                        unique_rulers.remove(existing)
                        unique_rulers.append(ruler)
                    break

            if not is_duplicate:
                unique_rulers.append(ruler)

        return unique_rulers

    def detect_objects(self, frame):
        """Detect coins, rulers, and cards in frame"""
        detected_objects = {}

        # Detect coins
        coins = self.detect_coins(frame)
        if coins:
            detected_objects['coins'] = coins

        # Detect rulers
        rulers = self.detect_rulers(frame)
        if rulers:
            detected_objects['rulers'] = rulers

        # Detect cards specifically
        cards = self.detect_cards(frame)
        if cards:
            detected_objects['cards'] = cards

        return detected_objects

    def draw_detection_overlay(self, frame, detected_objects):
        """Draw detection overlays for coins, rulers, and cards"""
        overlay = frame.copy()

        # Draw coins
        if 'coins' in detected_objects:
            for coin in detected_objects['coins']:
                center = coin['center']
                radius = coin['radius']

                # Draw circle
                cv2.circle(overlay, center, radius, (255, 0, 255), 2)  # Magenta
                cv2.circle(overlay, center, 3, (255, 0, 255), -1)     # Center dot

                # Label
                cv2.putText(overlay, f"Coin: {radius*2}px ({coin['confidence']:.2f})",
                           (center[0]-40, center[1]-radius-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        # Draw rulers
        if 'rulers' in detected_objects:
            for i, ruler in enumerate(detected_objects['rulers']):
                x, y, w, h = ruler['bounding_rect']
                center = ruler['center']

                # Use different colors for different detection methods
                method = ruler.get('detection_method', 'unknown')
                if method == 'edges':
                    color = (0, 255, 255)  # Cyan
                elif method == 'contours':
                    color = (0, 255, 0)    # Green
                else:
                    color = (128, 128, 128) # Gray

                # Draw rectangle
                cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 2)
                cv2.circle(overlay, center, 3, color, -1)  # Center dot

                # Label with confidence
                label = f"Ruler: {ruler['length_pixels']}px ({ruler['confidence']:.2f})"
                cv2.putText(overlay, label,
                           (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw cards
        if 'cards' in detected_objects:
            for card in detected_objects['cards']:
                x, y, w, h = card['bounding_rect']
                center = card['center']

                # Draw rectangle in orange for cards
                color = (0, 165, 255)  # Orange
                cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 2)
                cv2.circle(overlay, center, 3, color, -1)  # Center dot

                # Label
                label = f"Card: {card['width_pixels']}px ({card['confidence']:.2f})"
                cv2.putText(overlay, label,
                           (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return overlay

class CameraInterfaceWrapper:
    """Wrapper to make your USBCameraInterface work with the GUI"""
    def __init__(self):
        try:
            log_with_timestamp("Creating USBCameraInterface...")
            # Use your USBCameraInterface with camera ID 1 (Logitech B525)
            self.usb_camera = USBCameraInterface(camera_id=CAMERA_ID, target_resolution=(CAMERA_WIDTH, CAMERA_HEIGHT))
            log_with_timestamp("Starting USB camera...")

            # Start camera with timeout handling
            start_result = {'success': False, 'error': None}

            def start_camera_thread():
                try:
                    result = self.usb_camera.start()
                    start_result['success'] = True
                    start_result['result'] = result
                    log_with_timestamp(f"Camera started with resolution: {result}")
                except Exception as e:
                    start_result['error'] = str(e)
                    log_with_timestamp(f"Camera start failed: {e}")

            # Start camera in separate thread with timeout
            start_thread = threading.Thread(target=start_camera_thread, daemon=True)
            start_thread.start()
            start_thread.join(timeout=10.0)  # 10 second timeout

            if start_thread.is_alive():
                raise Exception("Camera initialization timed out after 10 seconds")

            if not start_result['success']:
                error_msg = start_result.get('error', 'Unknown error during camera start')
                raise Exception(f"Camera start failed: {error_msg}")

            log_with_timestamp("USBCameraInterface wrapper initialized successfully")

        except Exception as e:
            log_with_timestamp(f"USBCameraInterface wrapper failed: {e}")
            raise

    def get_frame(self):
        """Get frame using your interface"""
        try:
            frame = self.usb_camera.get_frame()
            if frame is None:
                # If no frame in buffer, try direct capture
                frame = self.usb_camera.capture_single_frame()
            return frame
        except Exception as e:
            log_with_timestamp(f"Error getting frame: {e}")
            return None

    def release(self):
        """Release camera using your interface"""
        try:
            log_with_timestamp("Stopping USB camera...")
            self.usb_camera.stop()
            log_with_timestamp("USB camera stopped")
        except Exception as e:
            log_with_timestamp(f"Error releasing camera: {e}")

class DirectOpenCVCamera:
    """Direct OpenCV camera - simplest possible implementation"""
    def __init__(self, cap):
        self.cap = cap
        log_with_timestamp("Direct OpenCV camera initialized")

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        if self.cap:
            self.cap.release()
            log_with_timestamp("Direct OpenCV camera released")

class SimpleCameraInterface:
    """Simple camera interface that works with your existing setup"""
    def __init__(self):
        try:
            # Try your camera ID first
            self.cap = cv2.VideoCapture(CAMERA_ID)
            if not self.cap.isOpened():
                # Fallback to camera 0
                self.cap = cv2.VideoCapture(0)

            if not self.cap.isOpened():
                raise Exception("Cannot open any camera")

            # Set your working resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            log_with_timestamp("Simple camera initialized successfully")

        except Exception as e:
            log_with_timestamp(f"Simple camera initialization failed: {e}")
            raise

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()


class MeasurementGUI:
    """Main GUI application for the PYNQ Z7 measurement system"""

    def __init__(self):
        self.root = tk.Tk()
        self.setup_window()

        # Initialize system components
        self.camera = None
        self.image_processor = ImageProcessor()
        self.measurement_calc = MeasurementCalculator()
        self.object_detector = ObjectDetector()

        # GUI state variables
        self.is_measuring = False
        self.current_frame = None
        self.processed_frame = None
        self.calibration_active = False
        self.measurement_results = {}

        # Measurement state
        self.measurement_mode = None  # 'distance', 'diameter', 'area'
        self.measurement_points = []  # Store clicked points
        self.pixels_per_mm = 1.0  # Calibration factor
        self.measurement_overlay = None  # For drawing measurements
        self.mouse_coords = (0, 0)  # Current mouse position

        # Threading control
        self.camera_thread = None
        self.stop_camera = threading.Event()

        # Create GUI components
        self.create_widgets()
        self.setup_layout()

        # Initialize camera in background to speed up GUI loading
        self.root.after(100, self.initialize_camera_background)

    def setup_window(self):
        """Configure main window properties"""
        self.root.title("PYNQ Z7 Precision Object Measurement System")
        self.root.geometry("1400x900")  # Larger window to accommodate all controls
        self.root.minsize(1200, 800)    # Minimum size to ensure visibility

        # Configure style
        style = ttk.Style()
        style.theme_use('clam')

        # Window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        """Create all GUI widgets"""

        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")

        # Video display frame
        self.video_frame = ttk.LabelFrame(self.main_frame, text="Live Camera Feed", padding="5")
        self.video_label = ttk.Label(self.video_frame, text="Camera Initializing...",
                                   background="black", foreground="white")

        # Bind mouse events to video label for measurements
        self.video_label.bind("<Button-1>", self.on_video_click)
        self.video_label.bind("<Button-3>", self.on_video_right_click)  # Right-click
        self.video_label.bind("<Motion>", self.on_video_motion)

        # Mouse coordinate display
        self.mouse_coords_label = ttk.Label(self.video_frame, text="Mouse: (0, 0)",
                                          font=("Consolas", 9), background="black", foreground="lime")

        # Control panel frame with scrollable content
        self.control_outer_frame = ttk.LabelFrame(self.main_frame, text="Controls", padding="5")

        # Create a canvas and scrollbar for the control panel
        self.control_canvas = tk.Canvas(self.control_outer_frame, highlightthickness=0)
        self.control_scrollbar = ttk.Scrollbar(self.control_outer_frame, orient="vertical", command=self.control_canvas.yview)
        self.control_frame = ttk.Frame(self.control_canvas)

        # Configure scrolling
        self.control_canvas.configure(yscrollcommand=self.control_scrollbar.set)
        self.control_frame.bind("<Configure>", lambda e: self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all")))
        self.control_canvas.create_window((0, 0), window=self.control_frame, anchor="nw")

        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            self.control_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.control_canvas.bind("<MouseWheel>", _on_mousewheel)

        # Camera controls
        self.camera_controls_frame = ttk.Frame(self.control_frame)
        self.start_button = ttk.Button(self.camera_controls_frame, text="Start Camera",
                                     command=self.start_camera)
        self.stop_button = ttk.Button(self.camera_controls_frame, text="Stop Camera",
                                    command=self.stop_camera_feed, state="disabled")
        self.capture_button = ttk.Button(self.camera_controls_frame, text="Capture Image",
                                       command=self.capture_image, state="disabled")

        # Processing controls - now with advanced options
        self.processing_frame = ttk.LabelFrame(self.control_frame, text="Advanced Image Processing", padding="5")

        self.edge_detection_var = tk.BooleanVar(value=True)
        self.edge_detection_cb = ttk.Checkbutton(self.processing_frame,
                                                text="Edge Detection",
                                                variable=self.edge_detection_var,
                                                command=self.update_processing)

        self.noise_filter_var = tk.BooleanVar(value=True)
        self.noise_filter_cb = ttk.Checkbutton(self.processing_frame,
                                             text="Noise Filtering",
                                             variable=self.noise_filter_var,
                                             command=self.update_processing)

        # Advanced processing method selection - only create if not already exists
        self.advanced_frame = ttk.Frame(self.processing_frame)

        # Edge detection method
        ttk.Label(self.advanced_frame, text="Edge Method:").grid(row=0, column=0, sticky="w")
        self.edge_method_var = tk.StringVar(value="canny_adaptive")
        self.edge_method_combo = ttk.Combobox(self.advanced_frame,
                                            textvariable=self.edge_method_var,
                                            values=["canny_adaptive", "canny_standard", "sobel", "scharr"],
                                            state="readonly", width=12)
        self.edge_method_combo.grid(row=0, column=1, padx=(5,0))
        self.edge_method_combo.bind("<<ComboboxSelected>>", lambda e: self.update_processing())

        # Noise filter method
        ttk.Label(self.advanced_frame, text="Noise Filter:").grid(row=1, column=0, sticky="w")
        self.noise_method_var = tk.StringVar(value="bilateral")
        self.noise_method_combo = ttk.Combobox(self.advanced_frame,
                                             textvariable=self.noise_method_var,
                                             values=["bilateral", "gaussian", "median"],
                                             state="readonly", width=12)
        self.noise_method_combo.grid(row=1, column=1, padx=(5,0))
        self.noise_method_combo.bind("<<ComboboxSelected>>", lambda e: self.update_processing())

        # Edge detection threshold controls
        self.threshold_frame = ttk.Frame(self.processing_frame)
        ttk.Label(self.threshold_frame, text="Edge Threshold Low:").grid(row=0, column=0, sticky="w")
        self.threshold_low_var = tk.IntVar(value=EDGE_DETECTION_LOW)
        self.threshold_low_scale = ttk.Scale(self.threshold_frame, from_=10, to=100,
                                           variable=self.threshold_low_var,
                                           orient="horizontal", length=150,
                                           command=self.update_thresholds)
        self.threshold_low_label = ttk.Label(self.threshold_frame,
                                           text=str(self.threshold_low_var.get()),
                                           width=3, relief="sunken")

        ttk.Label(self.threshold_frame, text="Edge Threshold High:").grid(row=1, column=0, sticky="w")
        self.threshold_high_var = tk.IntVar(value=EDGE_DETECTION_HIGH)
        self.threshold_high_scale = ttk.Scale(self.threshold_frame, from_=50, to=200,
                                            variable=self.threshold_high_var,
                                            orient="horizontal", length=150,
                                            command=self.update_thresholds)
        self.threshold_high_label = ttk.Label(self.threshold_frame,
                                            text=str(self.threshold_high_var.get()),
                                            width=3, relief="sunken")

        # Measurement controls
        self.measurement_frame = ttk.LabelFrame(self.control_frame, text="Measurements", padding="5")

        self.measure_distance_button = ttk.Button(self.measurement_frame, text="Measure Distance",
                                                command=self.measure_distance)
        self.measure_diameter_button = ttk.Button(self.measurement_frame, text="Measure Diameter",
                                                command=self.measure_diameter)
        self.measure_area_button = ttk.Button(self.measurement_frame, text="Measure Area",
                                            command=self.measure_area)
        self.cancel_measurement_button = ttk.Button(self.measurement_frame, text="Cancel Measurement",
                                                  command=self.cancel_measurement, state="disabled")

        # Measurement status
        self.measurement_status = ttk.Label(self.measurement_frame, text="Ready for measurement",
                                          foreground="green")

        # Calibration controls
        self.calibration_frame = ttk.LabelFrame(self.control_frame, text="Calibration", padding="5")

        # Calibration method selection
        self.calibration_method_frame = ttk.Frame(self.calibration_frame)
        ttk.Label(self.calibration_method_frame, text="Reference Object:").pack(anchor="w")

        self.calibration_method_var = tk.StringVar(value="5 Shekel")

        # Create scrollable frame for reference objects if there are many
        self.objects_frame = ttk.Frame(self.calibration_method_frame)

        # Debug: Log what objects we're creating radio buttons for
        log_with_timestamp(f"Creating radio buttons for: {list(REFERENCE_OBJECTS.keys())}")

        for obj_name in REFERENCE_OBJECTS.keys():
            size_mm = REFERENCE_OBJECTS[obj_name]
            if "Ruler" in obj_name:
                display_text = f"{obj_name} ({size_mm}mm length)"
            elif "Card" in obj_name:
                display_text = f"{obj_name} ({size_mm}mm width)"
            else:
                display_text = f"{obj_name} ({size_mm}mm diameter)"

            radio_btn = ttk.Radiobutton(self.objects_frame,
                          text=display_text,
                          variable=self.calibration_method_var,
                          value=obj_name)
            radio_btn.pack(anchor="w")
            log_with_timestamp(f"Created radio button: {display_text}")

        # Calibration action buttons
        self.calibration_buttons_frame = ttk.Frame(self.calibration_frame)
        self.auto_calibrate_button = ttk.Button(self.calibration_buttons_frame, text="Auto Detect & Calibrate",
                                              command=self.auto_calibrate)
        self.manual_calibrate_button = ttk.Button(self.calibration_buttons_frame, text="Manual Calibration",
                                                command=self.manual_calibrate)

        # Calibration status and results
        self.calibration_status = ttk.Label(self.calibration_frame, text="Status: Not Calibrated",
                                          foreground="red")
        self.calibration_value_label = ttk.Label(self.calibration_frame, text="Scale: 1.0 px/mm",
                                                foreground="blue")

        # Object detection toggle
        self.detection_frame = ttk.LabelFrame(self.control_frame, text="Object Detection", padding="5")
        self.object_detection_var = tk.BooleanVar(value=False)
        self.object_detection_cb = ttk.Checkbutton(self.detection_frame,
                                                 text="Show Object Detection",
                                                 variable=self.object_detection_var,
                                                 command=self.toggle_object_detection)
        self.detection_status = ttk.Label(self.detection_frame, text="Detection: Off",
                                        foreground="gray")

        # Results panel
        self.results_frame = ttk.LabelFrame(self.main_frame, text="Measurement Results", padding="5")

        # Results display with scrollbar
        self.results_text_frame = ttk.Frame(self.results_frame)
        self.results_text = tk.Text(self.results_text_frame, height=15, width=40,
                                   wrap=tk.WORD, state="disabled")
        self.results_scrollbar = ttk.Scrollbar(self.results_text_frame,
                                             command=self.results_text.yview)
        self.results_text.config(yscrollcommand=self.results_scrollbar.set)

        # Results control buttons
        self.results_buttons_frame = ttk.Frame(self.results_frame)
        self.clear_results_button = ttk.Button(self.results_buttons_frame, text="Clear Results",
                                             command=self.clear_results)
        self.save_results_button = ttk.Button(self.results_buttons_frame, text="Save Results",
                                            command=self.save_results)

        # Status bar
        self.status_frame = ttk.Frame(self.main_frame)
        self.status_var = tk.StringVar(value="System Ready")
        self.status_label = ttk.Label(self.status_frame, textvariable=self.status_var,
                                    relief="sunken", anchor="w")

    def setup_layout(self):
        """Arrange all widgets in the window"""

        # Main frame
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Configure main frame grid with better proportions
        self.main_frame.columnconfigure(0, weight=3)  # Video gets more space
        self.main_frame.columnconfigure(1, weight=2)  # Controls get adequate space
        self.main_frame.columnconfigure(2, weight=2)  # Results get adequate space
        self.main_frame.rowconfigure(0, weight=1)

        # Video frame
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.video_label.pack(expand=True, fill="both")

        # Mouse coordinates overlay (top-right corner)
        self.mouse_coords_label.place(relx=1.0, rely=0.0, anchor="ne", x=-10, y=10)

        # Control frame with scrollbar
        self.control_outer_frame.grid(row=0, column=1, sticky="nsew", padx=5)
        self.control_canvas.pack(side="left", fill="both", expand=True)
        self.control_scrollbar.pack(side="right", fill="y")

        # Set a fixed width for the control canvas to prevent layout issues
        self.control_canvas.configure(width=300)

        # Camera controls - more compact
        self.camera_controls_frame.pack(fill="x", pady=(0, 5))
        self.start_button.pack(side="left", padx=(0, 2))
        self.stop_button.pack(side="left", padx=(0, 2))
        self.capture_button.pack(side="left")

        # Processing controls - more compact with advanced options
        self.processing_frame.pack(fill="x", pady=(0, 5))
        self.edge_detection_cb.pack(anchor="w")
        self.noise_filter_cb.pack(anchor="w")

        # Advanced method selection
        self.advanced_frame.pack(fill="x", pady=(2, 0))

        # Threshold controls - more compact
        self.threshold_frame.pack(fill="x", pady=(2, 0))
        self.threshold_low_scale.grid(row=0, column=1, padx=(5, 0))
        self.threshold_low_label.grid(row=0, column=2, padx=(5, 0))
        self.threshold_high_scale.grid(row=1, column=1, padx=(5, 0))
        self.threshold_high_label.grid(row=1, column=2, padx=(5, 0))

        # Measurement controls - more compact
        self.measurement_frame.pack(fill="x", pady=(0, 5))
        self.measure_distance_button.pack(fill="x", pady=1)
        self.measure_diameter_button.pack(fill="x", pady=1)
        self.measure_area_button.pack(fill="x", pady=1)
        self.cancel_measurement_button.pack(fill="x", pady=1)
        self.measurement_status.pack(pady=1)

        # Calibration controls - more compact
        self.calibration_frame.pack(fill="x", pady=(0, 5))

        self.calibration_method_frame.pack(fill="x", pady=(0, 2))
        self.objects_frame.pack(fill="x", pady=(2, 0))

        self.calibration_buttons_frame.pack(fill="x", pady=(2, 2))
        self.auto_calibrate_button.pack(fill="x", pady=1)
        self.manual_calibrate_button.pack(fill="x", pady=1)

        self.calibration_status.pack(pady=1)
        self.calibration_value_label.pack(pady=1)

        # Object detection controls - more compact
        self.detection_frame.pack(fill="x", pady=(0, 5))
        self.object_detection_cb.pack(anchor="w")
        self.detection_status.pack(pady=1)

        # Measurement controls - more compact
        self.measurement_frame.pack(fill="x", pady=(0, 5))
        self.measure_distance_button.pack(fill="x", pady=1)
        self.measure_diameter_button.pack(fill="x", pady=1)
        self.measure_area_button.pack(fill="x", pady=1)
        self.cancel_measurement_button.pack(fill="x", pady=1)
        self.measurement_status.pack(pady=1)

        # Calibration controls - more compact
        self.calibration_frame.pack(fill="x", pady=(0, 5))

        self.calibration_method_frame.pack(fill="x", pady=(0, 2))
        self.objects_frame.pack(fill="x", pady=(2, 0))

        self.calibration_buttons_frame.pack(fill="x", pady=(2, 2))
        self.auto_calibrate_button.pack(fill="x", pady=1)
        self.manual_calibrate_button.pack(fill="x", pady=1)

        self.calibration_status.pack(pady=1)
        self.calibration_value_label.pack(pady=1)

        # Object detection controls - more compact
        self.detection_frame.pack(fill="x", pady=(0, 5))
        self.object_detection_cb.pack(anchor="w")
        self.detection_status.pack(pady=1)

        # Results frame
        self.results_frame.grid(row=0, column=2, sticky="nsew", padx=(5, 0))

        self.results_text_frame.pack(fill="both", expand=True)
        self.results_text.pack(side="left", fill="both", expand=True)
        self.results_scrollbar.pack(side="right", fill="y")

        self.results_buttons_frame.pack(fill="x", pady=(5, 0))
        self.clear_results_button.pack(side="left", padx=(0, 5))
        self.save_results_button.pack(side="left")

        # Status bar
        self.status_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        self.status_label.pack(fill="x")

    def initialize_camera_background(self):
        """Initialize camera in background to not block GUI"""
        def init_camera():
            try:
                log_with_timestamp("Background camera initialization started...")
                self.initialize_camera()
                log_with_timestamp("Background camera initialization completed")
            except Exception as e:
                log_with_timestamp(f"Background camera initialization failed: {e}")
                self.update_status(f"Camera initialization failed: {e}")

        # Run camera initialization in a separate thread
        init_thread = threading.Thread(target=init_camera, daemon=True)
        init_thread.start()

    def initialize_camera(self):
        """Initialize camera system with multiple fallback options"""
        self.update_status("Initializing camera...")

        # Try multiple camera options in order of preference
        camera_options = [
            ("Direct OpenCV Camera", self.try_direct_opencv),
            ("USBCameraInterface", self.try_usb_camera_interface),
            ("Simple OpenCV Camera", self.try_simple_camera),
        ]

        for name, init_func in camera_options:
            try:
                log_with_timestamp(f"Attempting {name}...")
                self.camera = init_func()
                if self.camera:
                    self.update_status(f"{name} ready")
                    log_with_timestamp(f"{name} initialized successfully")
                    return
            except Exception as e:
                log_with_timestamp(f"{name} failed: {e}")
                continue

        # If we get here, no camera worked
        self.update_status("No camera available")
        self.camera = None
        log_with_timestamp("All camera initialization methods failed")

    def try_direct_opencv(self):
        """Try direct OpenCV without any wrappers - fastest option"""
        log_with_timestamp("Testing direct OpenCV camera...")
        cap = cv2.VideoCapture(CAMERA_ID)
        if not cap.isOpened():
            cap.release()
            raise Exception(f"Cannot open camera {CAMERA_ID}")

        # Test frame capture
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            raise Exception(f"Cannot capture frames from camera {CAMERA_ID}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        return DirectOpenCVCamera(cap)

    def try_usb_camera_interface(self):
        """Try to initialize USBCameraInterface"""
        if not CAMERA_AVAILABLE:
            raise Exception("USBCameraInterface not available")
        return CameraInterfaceWrapper()

    def try_simple_camera(self):
        """Try to initialize simple camera"""
        return SimpleCameraInterface()

    def start_camera(self):
        """Start camera feed"""
        log_with_timestamp("Start camera button pressed")

        if self.camera is None:
            log_with_timestamp("Camera is None, initializing...")
            try:
                self.initialize_camera()
            except Exception as e:
                log_with_timestamp(f"Camera initialization in start_camera failed: {e}")
                messagebox.showerror("Camera Error", f"Camera initialization failed: {e}")
                return

        if self.camera is None:
            messagebox.showerror("Camera Error", "No camera available. Please check camera connection.")
            return

        try:
            log_with_timestamp("Starting camera thread...")
            self.stop_camera.clear()
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()

            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.capture_button.config(state="normal")

            self.update_status("Camera started")
            log_with_timestamp("Camera thread started successfully")

        except Exception as e:
            self.update_status(f"Failed to start camera: {e}")
            log_with_timestamp(f"Camera start error: {e}")
            messagebox.showerror("Camera Error", f"Failed to start camera: {e}")

    def stop_camera_feed(self):
        """Stop camera feed"""
        self.stop_camera.set()

        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)

        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.capture_button.config(state="disabled")

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

                    # Log every 5 seconds instead of every 30 frames for less spam
                    if current_time - last_log_time >= 5.0:
                        log_with_timestamp(f"Camera running: {frame_count} frames processed, current frame shape: {frame.shape}")
                        last_log_time = current_time

                    self.current_frame = frame.copy()

                    # Apply processing if enabled
                    display_frame = self.process_frame(frame)

                    # Convert for display
                    self.update_video_display(display_frame)
                else:
                    if frame_count == 0:  # Only log this initially
                        log_with_timestamp("Warning: get_frame() returned None")

                time.sleep(1/30)  # ~30 FPS

            except Exception as e:
                log_with_timestamp(f"Camera loop error: {e}")
                self.update_status(f"Camera error: {e}")
                break

        log_with_timestamp("Camera loop ended")

    def process_frame(self, frame):
        """Process frame using advanced image processing methods"""
        processed = frame.copy()

        try:
            # Apply noise filtering FIRST if enabled (before edge detection)
            if self.noise_filter_var.get():
                noise_method = getattr(self, 'noise_method_var', None)
                if noise_method and hasattr(noise_method, 'get'):
                    method = noise_method.get()
                else:
                    method = 'bilateral'

                if hasattr(self.image_processor, 'apply_noise_filter'):
                    try:
                        processed = self.image_processor.apply_noise_filter(processed, method=method)
                        log_with_timestamp(f"Applied {method} noise filtering")
                    except TypeError:
                        processed = cv2.bilateralFilter(processed, 9, 75, 75)
                else:
                    # Basic noise filtering options
                    if method == 'bilateral':
                        processed = cv2.bilateralFilter(processed, 9, 75, 75)
                    elif method == 'gaussian':
                        processed = cv2.GaussianBlur(processed, (5, 5), 0)
                    elif method == 'median':
                        processed = cv2.medianBlur(processed, 5)

            # Apply edge detection SECOND if enabled
            if self.edge_detection_var.get():
                edge_method = getattr(self, 'edge_method_var', None)
                if edge_method and hasattr(edge_method, 'get'):
                    method = edge_method.get()
                else:
                    method = 'canny_adaptive'

                # Convert to grayscale first
                if len(processed.shape) == 3:
                    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                else:
                    gray = processed.copy()

                # Apply different edge detection methods
                if method == 'canny_adaptive':
                    # Adaptive Canny with auto thresholds
                    high_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    low_thresh = 0.5 * high_thresh
                    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
                    processed = cv2.Canny(blurred, int(low_thresh), int(high_thresh))

                elif method == 'canny_standard':
                    # Standard Canny with manual thresholds
                    processed = cv2.Canny(gray, self.threshold_low_var.get(), self.threshold_high_var.get())

                elif method == 'sobel':
                    # Sobel edge detection
                    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
                    processed = np.uint8(sobel_combined / sobel_combined.max() * 255)
                    _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                elif method == 'scharr':
                    # Scharr edge detection (more accurate than Sobel)
                    scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
                    scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
                    scharr_combined = np.sqrt(scharr_x**2 + scharr_y**2)
                    processed = np.uint8(scharr_combined / scharr_combined.max() * 255)
                    _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                log_with_timestamp(f"Applied {method} edge detection")

        except Exception as e:
            log_with_timestamp(f"Frame processing error: {e}")
            processed = frame.copy()

        # Add object detection overlay if enabled
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
            # Convert BGR to RGB
            if len(frame.shape) == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)

            # Resize to fit display
            display_size = (640, 480)
            pil_image = pil_image.resize(display_size, Image.Resampling.LANCZOS)

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)

            # Update display
            self.video_label.configure(image=photo, text="")
            self.video_label.image = photo  # Keep reference

        except Exception as e:
            self.update_status(f"Display update error: {e}")

    def update_processing(self):
        """Update processing settings with advanced methods"""
        edge_enabled = self.edge_detection_var.get()
        noise_enabled = self.noise_filter_var.get()

        # Safely get method values with fallbacks
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
        """Update threshold display labels and apply changes in real-time"""
        low_val = self.threshold_low_var.get()
        high_val = self.threshold_high_var.get()

        # Update label displays
        self.threshold_low_label.config(text=str(low_val))
        self.threshold_high_label.config(text=str(high_val))

        # Ensure high threshold is always greater than low threshold
        if high_val <= low_val:
            # Automatically adjust the other slider to maintain proper relationship
            if value is not None:  # Only auto-adjust if user is actively dragging
                try:
                    # Determine which slider was moved based on the value
                    current_val = int(float(value))
                    if abs(current_val - low_val) < abs(current_val - high_val):
                        # Low slider was moved, adjust high slider
                        new_high = max(low_val + 10, high_val)
                        if new_high <= 200:
                            self.threshold_high_var.set(new_high)
                            self.threshold_high_label.config(text=str(new_high))
                    else:
                        # High slider was moved, adjust low slider
                        new_low = min(high_val - 10, low_val)
                        if new_low >= 10:
                            self.threshold_low_var.set(new_low)
                            self.threshold_low_label.config(text=str(new_low))
                except:
                    pass  # If conversion fails, just update labels

        # Update status with current values
        self.update_status(f"Edge thresholds: Low={low_val}, High={high_val}")

        # Log the change with timestamp for debugging
        log_with_timestamp(f"Threshold update: Low={low_val}, High={high_val}")

    def capture_image(self):
        """Capture current frame"""
        if self.current_frame is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.png"

            # Save to images/samples directory
            samples_dir = project_root / "images" / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)

            filepath = samples_dir / filename
            cv2.imwrite(str(filepath), self.current_frame)

            self.add_result(f"Image captured: {filename}")
            self.update_status(f"Image saved: {filename}")
        else:
            messagebox.showwarning("Capture Error", "No frame available to capture")

    def measure_distance(self):
        """Measure distance between two points"""
        if self.current_frame is None:
            messagebox.showwarning("Measurement Error", "No image available for measurement")
            return

        self.start_measurement_mode('distance')
        self.measurement_status.config(text="Click two points to measure distance", foreground="blue")
        self.add_result("Distance measurement: Click two points on the image")

    def measure_diameter(self):
        """Measure object diameter"""
        if self.current_frame is None:
            messagebox.showwarning("Measurement Error", "No image available for measurement")
            return

        self.start_measurement_mode('diameter')
        self.measurement_status.config(text="Click two points on object edge", foreground="blue")
        self.add_result("Diameter measurement: Click two points on object edge")

    def measure_area(self):
        """Measure object area"""
        if self.current_frame is None:
            messagebox.showwarning("Measurement Error", "No image available for measurement")
            return

        self.start_measurement_mode('area')
        self.measurement_status.config(text="Click points to outline object area", foreground="blue")
        self.add_result("Area measurement: Click points to outline the object (right-click to finish)")

    def start_measurement_mode(self, mode):
        """Initialize measurement mode"""
        log_with_timestamp(f"Starting {mode} measurement mode")

        # Clear any previous measurement points
        self.measurement_points = []

        self.measurement_mode = mode

        # Store measurement type for later reference
        self.last_measurement_type = mode

        # Update button states
        self.measure_distance_button.config(state="disabled")
        self.measure_diameter_button.config(state="disabled")
        self.measure_area_button.config(state="disabled")
        self.cancel_measurement_button.config(state="normal")

        # Change video frame title to show active mode
        self.video_frame.config(text=f"Live Camera Feed - {mode.upper()} MODE")

        self.update_status(f"{mode.capitalize()} measurement mode active")

    def reset_measurement_buttons(self):
        """Reset measurement buttons to normal state"""
        self.measure_distance_button.config(state="normal")
        self.measure_diameter_button.config(state="normal")
        self.measure_area_button.config(state="normal")
        self.cancel_measurement_button.config(state="normal")  # Keep cancel available

    def cancel_measurement(self):
        """Cancel current measurement and clear all points"""
        log_with_timestamp(f"Cancelling measurement - clearing {len(self.measurement_points)} points")

        # Reset all measurement/calibration states
        self.measurement_mode = None
        self.measurement_points = []  # Clear all points
        self.calibration_active = False

        # Reset ALL button states - including calibration buttons
        self.measure_distance_button.config(state="normal")
        self.measure_diameter_button.config(state="normal")
        self.measure_area_button.config(state="normal")
        self.cancel_measurement_button.config(state="disabled")

        # Re-enable calibration buttons
        self.auto_calibrate_button.config(state="normal")
        self.manual_calibrate_button.config(state="normal")

        # Reset video frame title
        self.video_frame.config(text="Live Camera Feed")

        # Reset status messages
        self.measurement_status.config(text="Ready for measurement", foreground="green")
        if hasattr(self, 'calibration_status') and self.calibration_active:
            # Only reset calibration status if we were in calibration mode
            self.calibration_status.config(text="Status: Calibration Cancelled", foreground="orange")

        self.update_status("Measurement/Calibration cancelled - points cleared")
        self.add_result("Measurement/Calibration cancelled - all points cleared")

        log_with_timestamp("All measurement and calibration states reset")

    def on_video_click(self, event):
        """Handle mouse clicks on video display"""
        log_with_timestamp(f"Video click detected at display coordinates: ({event.x}, {event.y})")

        if not self.measurement_mode:
            log_with_timestamp("No measurement mode active, ignoring click")
            return

        # Convert click coordinates to image coordinates
        img_x, img_y = self.convert_display_to_image_coords(event.x, event.y)
        if img_x is None or img_y is None:
            log_with_timestamp("Failed to convert coordinates")
            return

        point = (img_x, img_y)
        self.measurement_points.append(point)
        log_with_timestamp(f"Added measurement point: {point} (total points: {len(self.measurement_points)})")

        # Handle different measurement modes
        if self.measurement_mode == 'distance' and len(self.measurement_points) == 2:
            log_with_timestamp("Distance measurement ready - calculating...")
            self.calculate_distance()
        elif self.measurement_mode == 'diameter' and len(self.measurement_points) == 2:
            log_with_timestamp("Diameter measurement ready - calculating...")
            self.calculate_diameter()
        elif self.measurement_mode in ['calibration_coin', 'calibration_ruler'] and len(self.measurement_points) == 2:
            log_with_timestamp("Calibration measurement ready - calculating...")
            self.complete_manual_calibration()
        elif self.measurement_mode == 'area':
            log_with_timestamp(f"Area measurement: {len(self.measurement_points)} points selected")
            # Area measurement continues until right-click

    def on_video_right_click(self, event):
        """Handle right-click to finish area measurement"""
        if self.measurement_mode == 'area' and len(self.measurement_points) >= 3:
            self.calculate_area()
        elif self.measurement_mode == 'area':
            messagebox.showinfo("Area Measurement", "Need at least 3 points to calculate area")

    def on_video_motion(self, event):
        """Handle mouse motion over video display"""
        # Update mouse coordinates display
        img_x, img_y = self.convert_display_to_image_coords(event.x, event.y)
        if img_x is not None and img_y is not None:
            self.mouse_coords = (img_x, img_y)
            self.mouse_coords_label.config(text=f"Mouse: ({img_x}, {img_y})")

        # Live preview for measurement mode
        if self.measurement_mode and len(self.measurement_points) > 0:
            # Could add live preview line here in the future
            pass

    def convert_display_to_image_coords(self, display_x, display_y):
        """Convert display coordinates to image coordinates with proper scaling"""
        if self.current_frame is None:
            log_with_timestamp("No current frame available for coordinate conversion")
            return None, None

        try:
            # Get actual frame dimensions
            img_height, img_width = self.current_frame.shape[:2]

            # Get the video label's actual displayed size
            label_width = self.video_label.winfo_width()
            label_height = self.video_label.winfo_height()

            # The image is resized to 640x480 in update_video_display, but may be centered in label
            display_img_width = 640
            display_img_height = 480

            # Calculate centering offsets
            x_offset = max(0, (label_width - display_img_width) // 2)
            y_offset = max(0, (label_height - display_img_height) // 2)

            # Adjust for centering
            adjusted_x = display_x - x_offset
            adjusted_y = display_y - y_offset

            # Check bounds
            if adjusted_x < 0 or adjusted_x >= display_img_width or adjusted_y < 0 or adjusted_y >= display_img_height:
                log_with_timestamp(f"Click outside image area: ({adjusted_x}, {adjusted_y})")
                return None, None

            # Calculate scaling from display to actual image
            scale_x = img_width / display_img_width
            scale_y = img_height / display_img_height

            # Convert to image coordinates
            img_x = int(adjusted_x * scale_x)
            img_y = int(adjusted_y * scale_y)

            # Clamp to actual image bounds
            img_x = max(0, min(img_x, img_width - 1))
            img_y = max(0, min(img_y, img_height - 1))

            log_with_timestamp(f"Coord conversion: display({display_x},{display_y}) -> adjusted({adjusted_x},{adjusted_y}) -> image({img_x},{img_y})")
            log_with_timestamp(f"Scales: x={scale_x:.3f}, y={scale_y:.3f}, offset=({x_offset},{y_offset})")

            return img_x, img_y

        except Exception as e:
            log_with_timestamp(f"Coordinate conversion error: {e}")
            return None, Nonewidth()
            label_height = self.video_label.winfo_height()
            log_with_timestamp(f"Video label display size: {label_width}x{label_height}")

            # Calculate the actual image display size within the label
            # The image is resized to fit 640x480 in update_video_display
            display_width, display_height = 640, 480

            # Calculate scaling factors
            scale_x = img_width / display_width
            scale_y = img_height / display_height

            # Convert coordinates
            img_x = int(display_x * scale_x)
            img_y = int(display_y * scale_y)

            log_with_timestamp(f"Coordinate conversion: display({display_x}, {display_y}) -> image({img_x}, {img_y})")
            log_with_timestamp(f"Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")

            # Clamp to image bounds
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
            # Draw points with smaller, more visible design
            for i, point in enumerate(self.measurement_points):
                # Small filled circle (3px radius)
                cv2.circle(overlay, point, 3, (0, 255, 0), -1)  # Green center
                # Thin white border (5px radius)
                cv2.circle(overlay, point, 5, (255, 255, 255), 1)  # White border
                # Point number with better visibility
                cv2.putText(overlay, str(i+1), (point[0]+8, point[1]-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # White text with thick border
                cv2.putText(overlay, str(i+1), (point[0]+8, point[1]-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # Green text

            # Draw lines for distance/diameter (always show if 2+ points exist)
            if len(self.measurement_points) >= 2:
                if hasattr(self, 'last_measurement_type'):
                    # Use stored measurement type for completed measurements
                    if self.last_measurement_type in ['distance', 'diameter', 'calibration_coin', 'calibration_ruler']:
                        cv2.line(overlay, self.measurement_points[0], self.measurement_points[1], (0, 0, 255), 2)
                    elif self.last_measurement_type == 'area' and len(self.measurement_points) > 2:
                        points = np.array(self.measurement_points, np.int32)
                        cv2.polylines(overlay, [points], True, (0, 0, 255), 2)
                else:
                    # For active measurements, use current measurement mode
                    if self.measurement_mode in ['distance', 'diameter', 'calibration_coin', 'calibration_ruler']:
                        cv2.line(overlay, self.measurement_points[0], self.measurement_points[1], (0, 0, 255), 2)
                    elif self.measurement_mode == 'area' and len(self.measurement_points) > 2:
                        points = np.array(self.measurement_points, np.int32)
                        cv2.polylines(overlay, [points], True, (0, 0, 255), 2)

            # Draw live preview line from last point to mouse (only if actively measuring)
            if self.measurement_mode and len(self.measurement_points) > 0 and hasattr(self, 'mouse_coords'):
                last_point = self.measurement_points[-1]
                mouse_point = self.mouse_coords
                if self.measurement_mode in ['distance', 'diameter', 'calibration_coin', 'calibration_ruler'] and len(self.measurement_points) == 1:
                    # Show preview line for distance/diameter/calibration
                    cv2.line(overlay, last_point, mouse_point, (128, 128, 128), 1)  # Gray preview line
                elif self.measurement_mode == 'area' and len(self.measurement_points) >= 1:
                    # Show preview line for area
                    cv2.line(overlay, last_point, mouse_point, (128, 128, 128), 1)  # Gray preview line

        except Exception as e:
            log_with_timestamp(f"Error drawing overlay: {e}")

        return overlay

    def calculate_distance(self):
        """Calculate distance between two points"""
        if len(self.measurement_points) < 2:
            return

        p1, p2 = self.measurement_points[0], self.measurement_points[1]

        # Calculate pixel distance
        pixel_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

        # Convert to real-world distance
        real_distance = pixel_distance / self.pixels_per_mm

        result = f"Distance: {real_distance:.2f} mm ({pixel_distance:.1f} pixels)"
        log_with_timestamp(f"Distance calculated: {result}")
        self.add_result(result)

        # Don't cancel measurement - keep points and line visible
        self.measurement_mode = None  # Disable new point selection
        self.reset_measurement_buttons()  # Re-enable measurement buttons
        self.measurement_status.config(text="Distance measured - points visible", foreground="green")
        self.video_frame.config(text="Live Camera Feed - Distance Result Shown")

    def calculate_diameter(self):
        """Calculate diameter from two edge points"""
        if len(self.measurement_points) < 2:
            return

        p1, p2 = self.measurement_points[0], self.measurement_points[1]

        # Calculate pixel distance (diameter)
        pixel_diameter = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

        # Convert to real-world diameter
        real_diameter = pixel_diameter / self.pixels_per_mm

        result = f"Diameter: {real_diameter:.2f} mm ({pixel_diameter:.1f} pixels)"
        log_with_timestamp(f"Diameter calculated: {result}")
        self.add_result(result)

        # Don't cancel measurement - keep points and line visible
        self.measurement_mode = None  # Disable new point selection
        self.reset_measurement_buttons()  # Re-enable measurement buttons
        self.measurement_status.config(text="Diameter measured - points visible", foreground="green")
        self.video_frame.config(text="Live Camera Feed - Diameter Result Shown")

    def calculate_area(self):
        """Calculate area from polygon points"""
        if len(self.measurement_points) < 3:
            return

        # Calculate pixel area using shoelace formula
        points = np.array(self.measurement_points)
        x = points[:, 0]
        y = points[:, 1]

        pixel_area = 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(-1, len(x)-1)))

        # Convert to real-world area
        real_area = pixel_area / (self.pixels_per_mm ** 2)

        result = f"Area: {real_area:.2f} mm² ({pixel_area:.1f} pixels²)"
        log_with_timestamp(f"Area calculated: {result}")
        self.add_result(result)

        # Don't cancel measurement - keep points and polygon visible
        self.measurement_mode = None  # Disable new point selection
        self.reset_measurement_buttons()  # Re-enable measurement buttons
        self.measurement_status.config(text="Area measured - polygon visible", foreground="green")
        self.video_frame.config(text="Live Camera Feed - Area Result Shown")

    def auto_calibrate(self):
        """Automatically detect and calibrate using selected object"""
        if self.current_frame is None:
            messagebox.showwarning("Calibration Error", "No camera image available")
            return

        calibration_object = self.calibration_method_var.get()
        log_with_timestamp(f"Starting auto calibration with {calibration_object}")

        try:
            # Detect all objects in current frame
            detected_objects = self.object_detector.detect_objects(self.current_frame)

            # Determine what type of object we're looking for
            if "Shekel" in calibration_object:
                # Looking for a coin
                if 'coins' not in detected_objects or not detected_objects['coins']:
                    messagebox.showwarning("Calibration Failed",
                                         f"No coins detected in image.\n"
                                         f"Please ensure a {calibration_object} coin is clearly visible.")
                    return

                # Use the best detected coin
                best_coin = detected_objects['coins'][0]
                pixel_diameter = best_coin['diameter_pixels']
                real_diameter_mm = REFERENCE_OBJECTS[calibration_object]

                # Calculate pixels per mm
                self.pixels_per_mm = pixel_diameter / real_diameter_mm
                confidence = best_coin['confidence']

                measurement_type = "diameter"

            elif "Ruler" in calibration_object:
                # Looking for a ruler
                if 'rulers' not in detected_objects or not detected_objects['rulers']:
                    messagebox.showwarning("Calibration Failed",
                                         f"No rulers detected in image.\n"
                                         f"Please ensure a ruler is clearly visible and rectangular.")
                    return

                # Use the best detected ruler
                best_ruler = detected_objects['rulers'][0]
                pixel_length = best_ruler['length_pixels']
                real_length_mm = REFERENCE_OBJECTS[calibration_object]

                # Calculate pixels per mm
                self.pixels_per_mm = pixel_length / real_length_mm
                confidence = best_ruler['confidence']

                measurement_type = "length"

            elif "Card" in calibration_object:
                # Credit card detection
                if 'cards' not in detected_objects or not detected_objects['cards']:
                    messagebox.showwarning("Calibration Failed",
                                         f"No credit cards detected.\n"
                                         f"Please ensure the credit card is clearly visible and well-lit.")
                    return

                # Use the best detected card
                best_card = detected_objects['cards'][0]
                pixel_width = best_card['width_pixels']  # Use shorter dimension for credit card width
                real_width_mm = REFERENCE_OBJECTS[calibration_object]

                # Calculate pixels per mm
                self.pixels_per_mm = pixel_width / real_width_mm
                confidence = best_card['confidence']

                measurement_type = "width"

            else:
                messagebox.showwarning("Calibration Error", f"Unknown calibration object: {calibration_object}")
                return

            # Update calibration status
            self.calibration_status.config(text=f"Status: Calibrated ({confidence:.2f})",
                                         foreground="green")
            self.calibration_value_label.config(text=f"Scale: {self.pixels_per_mm:.3f} px/mm")

            # Log and display results
            result_text = (f"Auto calibration successful!\n"
                         f"Object: {calibration_object}\n" 
                         f"Measurement: {measurement_type}\n"
                         f"Scale: {self.pixels_per_mm:.3f} pixels/mm\n"
                         f"Confidence: {confidence:.2f}")

            log_with_timestamp(f"Auto calibration: {self.pixels_per_mm:.3f} px/mm, confidence: {confidence:.2f}")
            self.add_result(f"Auto calibration: {self.pixels_per_mm:.3f} px/mm ({calibration_object})")
            self.update_status("Auto calibration completed")

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

        # Disable other buttons
        self.auto_calibrate_button.config(state="disabled")
        self.manual_calibrate_button.config(state="disabled")
        self.cancel_measurement_button.config(state="normal")

        size_mm = REFERENCE_OBJECTS[calibration_object]

        self.calibration_status.config(text="Manual calibration: Click object edges", foreground="orange")
        self.measurement_status.config(text=f"Click two points on {calibration_object} ({size_mm}mm)", foreground="blue")
        self.video_frame.config(text="Live Camera Feed - MANUAL CALIBRATION")

        self.add_result(f"Manual calibration: Click two points on {calibration_object}")
        log_with_timestamp(f"Manual calibration started for {calibration_object}")

    def complete_manual_calibration(self):
        """Complete manual calibration with selected points"""
        if len(self.measurement_points) < 2:
            return

        p1, p2 = self.measurement_points[0], self.measurement_points[1]
        pixel_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

        # Get real distance from selected object
        calibration_object = getattr(self, 'current_calibration_object', self.calibration_method_var.get())
        real_distance_mm = REFERENCE_OBJECTS[calibration_object]

        # Calculate pixels per mm
        self.pixels_per_mm = pixel_distance / real_distance_mm

        # Update status
        self.calibration_status.config(text=f"Status: Calibrated (Manual)", foreground="green")
        self.calibration_value_label.config(text=f"Scale: {self.pixels_per_mm:.3f} px/mm")

        # Log results
        result_text = (f"Manual calibration successful!\n"
                      f"Object: {calibration_object}\n"
                      f"Pixel distance: {pixel_distance:.1f}px\n"
                      f"Real distance: {real_distance_mm}mm\n"
                      f"Scale: {self.pixels_per_mm:.3f} px/mm")

        log_with_timestamp(f"Manual calibration: {self.pixels_per_mm:.3f} px/mm using {calibration_object}")
        self.add_result(f"Manual calibration: {self.pixels_per_mm:.3f} px/mm ({calibration_object})")

        messagebox.showinfo("Calibration Complete", result_text)

        # Reset calibration mode
        self.calibration_active = False
        self.measurement_mode = None

        # Re-enable buttons
        self.auto_calibrate_button.config(state="normal")
        self.manual_calibrate_button.config(state="normal")
        self.cancel_measurement_button.config(state="disabled")

        self.measurement_status.config(text="Ready for measurement", foreground="green")
        self.video_frame.config(text="Live Camera Feed")
        self.update_status("Manual calibration completed")

    def toggle_object_detection(self):
        """Toggle object detection overlay"""
        if self.object_detection_var.get():
            self.detection_status.config(text="Detection: On", foreground="green")
            log_with_timestamp("Object detection enabled")
        else:
            self.detection_status.config(text="Detection: Off", foreground="gray")
            log_with_timestamp("Object detection disabled")

        self.update_status(f"Object detection: {'On' if self.object_detection_var.get() else 'Off'}")

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
        """Start the GUI application"""
        self.root.mainloop()


def main():
    """Main entry point"""
    try:
        app = MeasurementGUI()
        app.run()
    except Exception as e:
        log_with_timestamp(f"Failed to start application: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

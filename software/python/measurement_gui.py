"""
PYNQ Z7 Precision Object Measurement System - GUI Framework
Main user interface for real-time measurement system
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
except ImportError as e:
    log_with_timestamp(f"Config import error: {e}")
    # Fallback default values
    CAMERA_ID = 1
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    EDGE_DETECTION_LOW = 40
    EDGE_DETECTION_HIGH = 120

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
    """Placeholder image processor until real module is created"""
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
    """Placeholder measurement calculator until real module is created"""
    def __init__(self):
        pass

class CameraInterfaceWrapper:
    """Wrapper to make your USBCameraInterface work with the GUI"""
    def __init__(self):
        try:
            log_with_timestamp("Creating USBCameraInterface...")
            # Use your USBCameraInterface with camera ID 1 (Logitech B525)
            self.usb_camera = USBCameraInterface(camera_id=1, target_resolution=(640, 480))
            log_with_timestamp("Starting USB camera...")

            # Start camera with timeout handling
            import threading
            import time

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
        print("Direct OpenCV camera initialized")

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        if self.cap:
            self.cap.release()
            print("Direct OpenCV camera released")

class SimpleCameraInterface:
    """Simple camera interface that works with your existing setup"""
    def __init__(self):
        try:
            # Try your camera ID 1 first
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                # Fallback to camera 0
                self.cap = cv2.VideoCapture(0)

            if not self.cap.isOpened():
                raise Exception("Cannot open any camera")

            # Set your working resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print("Simple camera initialized successfully")

        except Exception as e:
            print(f"Simple camera initialization failed: {e}")
            raise

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()

# Remove the old function-based camera class since it's not needed


class MeasurementGUI:
    """Main GUI application for the PYNQ Z7 measurement system"""

    def __init__(self):
        self.root = tk.Tk()
        self.setup_window()

        # Initialize system components
        self.camera = None
        self.image_processor = ImageProcessor()
        self.measurement_calc = MeasurementCalculator()

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
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)

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

        # Control panel frame
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Controls", padding="5")

        # Camera controls
        self.camera_controls_frame = ttk.Frame(self.control_frame)
        self.start_button = ttk.Button(self.camera_controls_frame, text="Start Camera",
                                     command=self.start_camera)
        self.stop_button = ttk.Button(self.camera_controls_frame, text="Stop Camera",
                                    command=self.stop_camera_feed, state="disabled")
        self.capture_button = ttk.Button(self.camera_controls_frame, text="Capture Image",
                                       command=self.capture_image, state="disabled")

        # Processing controls
        self.processing_frame = ttk.LabelFrame(self.control_frame, text="Image Processing", padding="5")

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

        # Edge detection threshold controls
        self.threshold_frame = ttk.Frame(self.processing_frame)
        ttk.Label(self.threshold_frame, text="Edge Threshold Low:").grid(row=0, column=0, sticky="w")
        self.threshold_low_var = tk.IntVar(value=EDGE_DETECTION_LOW)
        self.threshold_low_scale = ttk.Scale(self.threshold_frame, from_=10, to=100,
                                           variable=self.threshold_low_var,
                                           orient="horizontal", length=150,
                                           command=self.update_thresholds)
        self.threshold_low_label = ttk.Label(self.threshold_frame,
                                           text=str(self.threshold_low_var.get()))

        ttk.Label(self.threshold_frame, text="Edge Threshold High:").grid(row=1, column=0, sticky="w")
        self.threshold_high_var = tk.IntVar(value=EDGE_DETECTION_HIGH)
        self.threshold_high_scale = ttk.Scale(self.threshold_frame, from_=50, to=200,
                                            variable=self.threshold_high_var,
                                            orient="horizontal", length=150,
                                            command=self.update_thresholds)
        self.threshold_high_label = ttk.Label(self.threshold_frame,
                                            text=str(self.threshold_high_var.get()))

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
        self.calibrate_button = ttk.Button(self.calibration_frame, text="Calibrate with Coin",
                                         command=self.start_calibration)
        self.calibration_status = ttk.Label(self.calibration_frame, text="Status: Not Calibrated",
                                          foreground="red")

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

        # Configure main frame grid
        self.main_frame.columnconfigure(0, weight=2)  # Video gets more space
        self.main_frame.columnconfigure(1, weight=1)  # Controls
        self.main_frame.columnconfigure(2, weight=1)  # Results
        self.main_frame.rowconfigure(0, weight=1)

        # Video frame
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.video_label.pack(expand=True, fill="both")

        # Control frame
        self.control_frame.grid(row=0, column=1, sticky="nsew", padx=5)

        # Camera controls
        self.camera_controls_frame.pack(fill="x", pady=(0, 10))
        self.start_button.pack(side="left", padx=(0, 5))
        self.stop_button.pack(side="left", padx=(0, 5))
        self.capture_button.pack(side="left")

        # Processing controls
        self.processing_frame.pack(fill="x", pady=(0, 10))
        self.edge_detection_cb.pack(anchor="w")
        self.noise_filter_cb.pack(anchor="w")

        # Threshold controls
        self.threshold_frame.pack(fill="x", pady=(5, 0))
        self.threshold_low_scale.grid(row=0, column=1, padx=(5, 0))
        self.threshold_low_label.grid(row=0, column=2, padx=(5, 0))
        self.threshold_high_scale.grid(row=1, column=1, padx=(5, 0))
        self.threshold_high_label.grid(row=1, column=2, padx=(5, 0))

        # Measurement controls
        self.measurement_frame.pack(fill="x", pady=(0, 10))
        self.measure_distance_button.pack(fill="x", pady=2)
        self.measure_diameter_button.pack(fill="x", pady=2)
        self.measure_area_button.pack(fill="x", pady=2)
        self.cancel_measurement_button.pack(fill="x", pady=2)
        self.measurement_status.pack(pady=2)

        # Calibration controls
        self.calibration_frame.pack(fill="x", pady=(0, 10))
        self.calibrate_button.pack(fill="x", pady=2)
        self.calibration_status.pack(pady=2)

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
                print("Background camera initialization started...")
                self.initialize_camera()
                print("Background camera initialization completed")
            except Exception as e:
                print(f"Background camera initialization failed: {e}")
                self.update_status(f"Camera initialization failed: {e}")

        # Run camera initialization in a separate thread
        init_thread = threading.Thread(target=init_camera, daemon=True)
        init_thread.start()

    def initialize_camera(self):
        """Initialize camera system with your USBCameraInterface"""
        self.update_status("Initializing camera...")

        # Try multiple camera options in order of preference
        camera_options = [
            ("Direct OpenCV Camera", self.try_direct_opencv),
            ("USBCameraInterface", self.try_usb_camera_interface),
            ("Simple OpenCV Camera", self.try_simple_camera),
        ]

        for name, init_func in camera_options:
            try:
                print(f"Attempting {name}...")
                self.camera = init_func()
                if self.camera:
                    self.update_status(f"{name} ready")
                    print(f"{name} initialized successfully")
                    return
            except Exception as e:
                print(f"{name} failed: {e}")
                continue

        # If we get here, no camera worked
        self.update_status("No camera available")
        self.camera = None
        print("All camera initialization methods failed")

    def try_direct_opencv(self):
        """Try direct OpenCV without any wrappers - fastest option"""
        print("Testing direct OpenCV camera 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            cap.release()
            raise Exception("Cannot open camera 1")

        # Test frame capture
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            raise Exception("Cannot capture frames from camera 1")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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
        print("Start camera button pressed")

        if self.camera is None:
            print("Camera is None, initializing...")
            try:
                self.initialize_camera()
            except Exception as e:
                print(f"Camera initialization in start_camera failed: {e}")
                messagebox.showerror("Camera Error", f"Camera initialization failed: {e}")
                return

        if self.camera is None:
            messagebox.showerror("Camera Error", "No camera available. Please check camera connection.")
            return

        try:
            print("Starting camera thread...")
            self.stop_camera.clear()
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()

            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.capture_button.config(state="normal")

            self.update_status("Camera started")
            print("Camera thread started successfully")

        except Exception as e:
            self.update_status(f"Failed to start camera: {e}")
            print(f"Camera start error: {e}")
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
        print("Camera loop started")
        frame_count = 0

        while not self.stop_camera.is_set():
            try:
                frame = self.camera.get_frame()
                if frame is not None:
                    frame_count += 1
                    if frame_count % 30 == 0:  # Log every 30 frames
                        print(f"Processing frame {frame_count}")

                    self.current_frame = frame.copy()

                    # Apply processing if enabled
                    display_frame = self.process_frame(frame)

                    # Convert for display
                    self.update_video_display(display_frame)
                else:
                    print("Warning: get_frame() returned None")

                time.sleep(1/30)  # ~30 FPS

            except Exception as e:
                print(f"Camera loop error: {e}")
                self.update_status(f"Camera error: {e}")
                break

        print("Camera loop ended")

    def process_frame(self, frame):
        """Process frame based on current settings"""
        processed = frame.copy()

        if self.edge_detection_var.get():
            processed = self.image_processor.detect_edges(
                processed,
                self.threshold_low_var.get(),
                self.threshold_high_var.get()
            )

        if self.noise_filter_var.get():
            processed = self.image_processor.apply_noise_filter(processed)

        # Add measurement overlay if in measurement mode
        if self.measurement_mode and len(self.measurement_points) > 0:
            processed = self.draw_measurement_overlay(processed)

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
        """Update processing settings"""
        self.update_status("Processing settings updated")

    def update_thresholds(self, value=None):
        """Update threshold display labels"""
        self.threshold_low_label.config(text=str(self.threshold_low_var.get()))
        self.threshold_high_label.config(text=str(self.threshold_high_var.get()))

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
        self.measurement_mode = mode
        self.measurement_points = []

        # Update button states
        self.measure_distance_button.config(state="disabled")
        self.measure_diameter_button.config(state="disabled")
        self.measure_area_button.config(state="disabled")
        self.cancel_measurement_button.config(state="normal")

        self.update_status(f"{mode.capitalize()} measurement mode active")

    def cancel_measurement(self):
        """Cancel current measurement"""
        self.measurement_mode = None
        self.measurement_points = []

        # Reset button states
        self.measure_distance_button.config(state="normal")
        self.measure_diameter_button.config(state="normal")
        self.measure_area_button.config(state="normal")
        self.cancel_measurement_button.config(state="disabled")

        self.measurement_status.config(text="Ready for measurement", foreground="green")
        self.update_status("Measurement cancelled")
        self.add_result("Measurement cancelled")

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

        if self.measurement_mode == 'distance' and len(self.measurement_points) == 2:
            log_with_timestamp("Distance measurement ready - calculating...")
            self.calculate_distance()
        elif self.measurement_mode == 'diameter' and len(self.measurement_points) == 2:
            log_with_timestamp("Diameter measurement ready - calculating...")
            self.calculate_diameter()
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
        if self.measurement_mode and len(self.measurement_points) > 0:
            # Could add live preview line here
            pass

    def convert_display_to_image_coords(self, display_x, display_y):
        """Convert display coordinates to image coordinates"""
        if self.current_frame is None:
            log_with_timestamp("No current frame available for coordinate conversion")
            return None, None

        try:
            # Get actual frame dimensions
            img_height, img_width = self.current_frame.shape[:2]
            log_with_timestamp(f"Current frame dimensions: {img_width}x{img_height}")

            # Get the video label's actual displayed size
            label_width = self.video_label.winfo_width()
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
            # Draw points
            for i, point in enumerate(self.measurement_points):
                cv2.circle(overlay, point, 8, (0, 255, 0), -1)  # Larger green dots
                cv2.circle(overlay, point, 10, (255, 255, 255), 2)  # White border
                cv2.putText(overlay, str(i+1), (point[0]+15, point[1]-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw lines for distance/diameter
            if self.measurement_mode in ['distance', 'diameter'] and len(self.measurement_points) >= 2:
                cv2.line(overlay, self.measurement_points[0], self.measurement_points[1], (0, 0, 255), 3)

            # Draw polygon for area
            elif self.measurement_mode == 'area' and len(self.measurement_points) > 2:
                points = np.array(self.measurement_points, np.int32)
                cv2.polylines(overlay, [points], True, (0, 0, 255), 3)

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
        self.add_result(result)

        self.cancel_measurement()

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
        self.add_result(result)

        self.cancel_measurement()

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
        self.add_result(result)

        self.cancel_measurement()

    def start_calibration(self):
        """Start calibration process"""
        if self.current_frame is None:
            messagebox.showwarning("Calibration Error", "No image available for calibration")
            return

        self.calibration_active = True
        self.add_result("Calibration started - Place reference coin in view")
        self.update_status("Calibration mode active")

        # Placeholder for calibration logic
        self.calibration_status.config(text="Status: Calibrating...", foreground="orange")

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
        print(f"Failed to start application: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

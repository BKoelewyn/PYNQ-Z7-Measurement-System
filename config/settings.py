"""
PYNQ Z7 Precision Object Measurement System - Configuration Settings
Enhanced with calibration objects and object detection parameters
"""

# Camera Configuration
CAMERA_ID = 0  # Logitech B525 USB camera
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Image Processing Parameters
EDGE_DETECTION_LOW = 40
EDGE_DETECTION_HIGH = 120
GAUSSIAN_BLUR_SIZE = 5

# Measurement System
TARGET_ACCURACY_MM = 0.5

# Calibration Objects Configuration
CALIBRATION_OBJECTS = {
    'coin': {
        'name': 'US Quarter',
        'diameter_mm': 24.26,  # US Quarter diameter in millimeters
        'thickness_mm': 1.75,
        'color_range_hsv': {
            'lower': [10, 50, 50],   # HSV lower bound for coin detection
            'upper': [25, 255, 255]  # HSV upper bound for coin detection
        },
        'detection_params': {
            'min_area': 2000,        # Minimum pixel area for coin detection
            'max_area': 15000,       # Maximum pixel area for coin detection
            'circularity_min': 0.7,  # Minimum circularity (1.0 = perfect circle)
            'aspect_ratio_tolerance': 0.3  # Tolerance for width/height ratio
        }
    },
    'ruler': {
        'name': 'Standard Ruler',
        'length_mm': 150,        # 15cm ruler
        'width_mm': 25,          # Standard ruler width
        'marking_interval_mm': 10,  # Major markings every 10mm
        'color_range_hsv': {
            'lower': [0, 0, 200],    # HSV for white/light colored ruler
            'upper': [180, 30, 255]
        },
        'detection_params': {
            'min_area': 5000,        # Minimum pixel area for ruler detection
            'max_area': 40000,       # Maximum pixel area for ruler detection
            'aspect_ratio_min': 4.0, # Length/width ratio (ruler is long and thin)
            'aspect_ratio_max': 8.0,
            'rectangularity_min': 0.8  # How rectangular the shape is
        }
    }
}


# Reference objects for calibration (in millimeters)
REFERENCE_OBJECTS = {
    "US Quarter": 24.26,
    "US Penny": 19.05,
    "US Nickel": 21.21,
    "US Dime": 17.91
}

# You can also add these if you want more calibration options
ADDITIONAL_REFERENCE_OBJECTS = {
    "Credit Card": 85.60,  # Credit card width
    "Business Card": 89.0,  # Standard business card width
    "AA Battery": 50.5,    # AA battery length
    "Standard Paperclip": 50.0,  # Large paperclip length
}

# Object detection settings (if you want to add object detection later)
OBJECT_DETECTION_ENABLED = True
COIN_DETECTION_PARAMS = {
    'min_radius': 20,
    'max_radius': 100,
    'param1': 50,
    'param2': 30,
    'min_dist': 50
}


# Object Detection Configuration
OBJECT_DETECTION = {
    'contour_detection': {
        'mode': 'RETR_EXTERNAL',     # Only detect outer contours
        'method': 'CHAIN_APPROX_SIMPLE'  # Compress contours
    },
    'morphology': {
        'kernel_size': 5,            # Morphological operations kernel size
        'erosion_iterations': 1,      # Noise removal
        'dilation_iterations': 2      # Fill gaps
    },
    'filtering': {
        'gaussian_blur': 5,          # Blur before processing
        'median_blur': 5,            # Remove salt-and-pepper noise
        'bilateral_filter': {
            'd': 9,                  # Diameter of pixel neighborhood
            'sigma_color': 75,       # Filter sigma in color space
            'sigma_space': 75        # Filter sigma in coordinate space
        }
    }
}

# HSV Color Space Ranges for Common Objects
HSV_COLOR_RANGES = {
    'red': {
        'lower1': [0, 50, 50],       # Red range 1 (lower part of HSV)
        'upper1': [10, 255, 255],
        'lower2': [170, 50, 50],     # Red range 2 (upper part of HSV)
        'upper2': [180, 255, 255]
    },
    'blue': {
        'lower': [100, 50, 50],
        'upper': [130, 255, 255]
    },
    'green': {
        'lower': [40, 50, 50],
        'upper': [80, 255, 255]
    },
    'yellow': {
        'lower': [20, 50, 50],
        'upper': [40, 255, 255]
    },
    'white': {
        'lower': [0, 0, 200],
        'upper': [180, 30, 255]
    },
    'black': {
        'lower': [0, 0, 0],
        'upper': [180, 255, 50]
    }
}

# Measurement Validation
MEASUREMENT_VALIDATION = {
    'min_pixel_distance': 10,       # Minimum distance between measurement points
    'max_measurement_error': 5.0,   # Maximum acceptable measurement error (%)
    'calibration_confidence_min': 0.8,  # Minimum confidence for calibration
    'repeat_measurements': 3         # Number of measurements to average
}

# Display and UI Configuration
DISPLAY_CONFIG = {
    'point_radius': 3,              # Measurement point display radius
    'line_thickness': 2,            # Measurement line thickness
    'text_font_scale': 0.5,         # OpenCV text font scale
    'overlay_alpha': 0.7,           # Transparency for overlays
    'colors': {
        'measurement_points': [0, 255, 0],     # Green
        'measurement_lines': [0, 0, 255],      # Red
        'detection_bounds': [255, 255, 0],     # Yellow
        'calibration_object': [255, 0, 255],   # Magenta
        'preview_line': [128, 128, 128]        # Gray
    }
}

# System Performance
PERFORMANCE_CONFIG = {
    'frame_processing_interval': 1,  # Process every N frames for detection
    'detection_timeout_seconds': 5.0,  # Timeout for object detection
    'calibration_stability_frames': 10,  # Frames to verify stable calibration
    'max_detection_attempts': 50    # Maximum detection attempts before timeout
}

# File Paths
PATHS = {
    'calibration_data': 'config/calibration.json',
    'measurement_logs': 'logs/measurements.txt',
    'captured_images': 'images/samples/',
    'calibration_images': 'images/calibration/'
}

# Debug and Logging
DEBUG_CONFIG = {
    'enable_detection_debug': True,   # Show detection debug info
    'enable_measurement_debug': True, # Show measurement debug info
    'save_detection_images': False,   # Save images with detection overlays
    'log_level': 'INFO'              # Logging level
}

# Validation functions
def validate_calibration_object(obj_name):
    """Validate that a calibration object exists and has required parameters"""
    if obj_name not in CALIBRATION_OBJECTS:
        raise ValueError(f"Calibration object '{obj_name}' not found")

    obj = CALIBRATION_OBJECTS[obj_name]
    required_fields = ['name', 'color_range_hsv', 'detection_params']

    for field in required_fields:
        if field not in obj:
            raise ValueError(f"Calibration object '{obj_name}' missing required field: {field}")

    return True

def get_calibration_object(obj_name):
    """Get calibration object configuration"""
    validate_calibration_object(obj_name)
    return CALIBRATION_OBJECTS[obj_name]

def get_available_calibration_objects():
    """Get list of available calibration objects"""
    return list(CALIBRATION_OBJECTS.keys())

# Print configuration summary
if __name__ == "__main__":
    print("PYNQ Z7 Measurement System Configuration")
    print("=" * 50)
    print(f"Camera: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps (ID: {CAMERA_ID})")
    print(f"Target Accuracy: {TARGET_ACCURACY_MM}mm")
    print(f"Edge Detection: Low={EDGE_DETECTION_LOW}, High={EDGE_DETECTION_HIGH}")
    print("\nAvailable Calibration Objects:")
    for obj_name, obj_config in CALIBRATION_OBJECTS.items():
        if obj_name == 'coin':
            print(f"  • {obj_config['name']}: {obj_config['diameter_mm']}mm diameter")
        elif obj_name == 'ruler':
            print(f"  • {obj_config['name']}: {obj_config['length_mm']}mm length")
    print("\nObject Detection Enabled: HSV color ranges configured")
    print(f"Debug Mode: {DEBUG_CONFIG['enable_detection_debug']}")

# Israeli coins for calibration (in millimeters) - OVERRIDE
REFERENCE_OBJECTS = {
    "1 Shekel": 18.0,   # Israeli 1 shekel coin diameter
    "2 Shekel": 21.6,   # Israeli 2 shekel coin diameter
    "5 Shekel": 24.0,   # Israeli 5 shekel coin diameter
    "10 Shekel": 26.0,  # Israeli 10 shekel coin diameter
    "Ruler (10cm)": 100.0,  # 10cm ruler segment
    "Ruler (15cm)": 150.0,  # 15cm ruler
    "Credit Card": 85.6     # Credit card width
}

print("Israeli coins loaded for calibration:", list(REFERENCE_OBJECTS.keys()))

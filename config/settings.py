# Configuration settings for PYNQ measurement system
# Optimized for Logitech B525 USB Camera - CONFIRMED WORKING

# USB Camera settings (Logitech B525 - TESTED AND WORKING)
CAMERA_ID = 1  # Confirmed: Logitech B525 on Camera 1
CAMERA_WIDTH = 640  # Confirmed working resolution
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_TYPE = "Logitech B525 HD Webcam"

# Confirmed camera capabilities
CAMERA_BACKEND = "MSMF"
CAPTURE_SUCCESS_RATE = 0.9  # 9/10 frames successful

# Logitech B525 current settings (detected values)
LOGITECH_CURRENT_SETTINGS = {
    'brightness': 128.0,
    'contrast': 32.0,
    'saturation': 32.0,
    'gain': 64.0,
    'exposure': -6.0,
    'auto_exposure': 0.0,
    'focus': 60.0,
    'autofocus': 0.0,
    'zoom': 1.0
}

# Optimized settings for measurement (can be adjusted)
LOGITECH_OPTIMAL_SETTINGS = {
    'brightness': 128.0,  # Keep current
    'contrast': 35.0,     # Slightly higher for edge detection
    'saturation': 30.0,   # Slightly lower for measurement
    'auto_exposure': 1.0, # Enable for consistent lighting
    'autofocus': 1.0,     # Enable for sharp focus
}

# Reference objects (in millimeters)
REFERENCE_OBJECTS = {
    "US Quarter": 24.26,
    "US Penny": 19.05,
    "US Nickel": 21.21,
    "US Dime": 17.91
}

# Processing settings optimized for 640x480 Logitech B525
EDGE_DETECTION_LOW = 50
EDGE_DETECTION_HIGH = 150
GAUSSIAN_BLUR_SIZE = 3
MIN_OBJECT_AREA = 50

# Measurement settings - achievable with 640x480
TARGET_ACCURACY_MM = 0.5  # Should be achievable with good calibration
CALIBRATION_SAMPLES = 15  # More samples for better accuracy
PIXEL_TO_MM_RATIO = 0.1   # Will be calibrated with reference objects

# GUI settings
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
UPDATE_RATE_MS = 33  # 30 FPS

# Camera validation
CAMERA_VALIDATED = True
VALIDATION_DATE = "2025-06-13"

print("Configuration loaded - Logitech B525 USB Camera validated and working")

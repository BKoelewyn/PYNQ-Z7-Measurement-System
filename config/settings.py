# Basic configuration settings for the measurement system

# Camera settings
CAMERA_ID = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Reference objects (in millimeters)
REFERENCE_OBJECTS = {
    "US Quarter": 24.26,
    "US Penny": 19.05,
    "US Nickel": 21.21,
    "US Dime": 17.91
}

# Processing settings
EDGE_DETECTION_LOW = 50
EDGE_DETECTION_HIGH = 150
MIN_OBJECT_AREA = 100

# GUI settings
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
UPDATE_RATE_MS = 33  # About 30 FPS

# Measurement accuracy target
TARGET_ACCURACY_MM = 0.5

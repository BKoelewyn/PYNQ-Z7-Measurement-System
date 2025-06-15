"""
USB Camera interface specifically for external cameras like Logitech B525
Skips internal cameras and focuses on USB camera functionality
"""

import cv2
import numpy as np
import threading
import time

class USBCameraInterface:
    """USB Camera interface - skips built-in cameras"""

    def __init__(self, camera_id=1, target_resolution=(640, 480)):
        self.camera_id = camera_id
        self.target_resolution = target_resolution
        self.cap = None
        self.current_frame = None
        self.is_running = False
        self.capture_thread = None

    def start(self):
        """Start USB camera with best available resolution"""
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open USB camera {self.camera_id}")

        # Try to set the target resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_resolution[1])

        # Get actual resolution
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"USB Camera {self.camera_id} opened: {actual_width}x{actual_height}")

        # Start capture thread
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        return actual_width, actual_height

    def _capture_loop(self):
        """Background frame capture"""
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
            time.sleep(0.03)  # ~30 FPS

    def get_frame(self):
        """Get the latest frame"""
        if self.current_frame is not None:
            return self.current_frame.copy()
        return None

    def capture_single_frame(self):
        """Capture a single frame directly"""
        if not self.cap or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def stop(self):
        """Stop camera capture"""
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        print(f"USB Camera {self.camera_id} stopped")

    def get_info(self):
        """Get camera information"""
        if not self.cap or not self.cap.isOpened():
            return None

        return {
            'camera_id': self.camera_id,
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'backend': self.cap.getBackendName(),
            'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
            'exposure': self.cap.get(cv2.CAP_PROP_EXPOSURE)
        }

def find_usb_cameras():
    """Find USB cameras only (skip camera 0 which is usually internal)"""
    usb_cameras = []

    print("Scanning for USB cameras (skipping internal camera)...")

    # Start from camera 1 to skip internal camera
    for camera_id in range(1, 5):
        print(f"Testing USB camera {camera_id}...")

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"  Camera {camera_id}: Not available")
            continue

        # Try to capture a test frame
        ret, frame = cap.read()
        if ret and frame is not None:
            height, width = frame.shape[:2]

            # Test stability with multiple frames
            stable_frames = 0
            for _ in range(3):
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    stable_frames += 1
                time.sleep(0.1)

            if stable_frames >= 2:
                # Get camera properties
                backend = cap.getBackendName()
                brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
                contrast = cap.get(cv2.CAP_PROP_CONTRAST)

                usb_cameras.append({
                    'id': camera_id,
                    'width': width,
                    'height': height,
                    'backend': backend,
                    'controls': brightness != -1 or contrast != -1,
                    'stable': True
                })
                print(f"  Camera {camera_id}: Working - {width}x{height} ({backend})")
            else:
                print(f"  Camera {camera_id}: Unstable capture")
        else:
            print(f"  Camera {camera_id}: Cannot capture frames")

        cap.release()
        time.sleep(0.2)

    return usb_cameras

def select_logitech_camera(usb_cameras):
    """Select the best USB camera (likely Logitech B525)"""
    if not usb_cameras:
        return None

    print(f"\nFound {len(usb_cameras)} USB camera(s):")
    for cam in usb_cameras:
        print(f"  Camera {cam['id']}: {cam['width']}x{cam['height']} - Controls: {cam['controls']}")

    # Prefer camera with more controls (likely external USB camera)
    best_camera = usb_cameras[0]
    for camera in usb_cameras:
        if camera['controls'] and not best_camera['controls']:
            best_camera = camera
        elif camera['controls'] == best_camera['controls']:
            # If controls are same, prefer higher resolution
            if camera['width'] * camera['height'] > best_camera['width'] * best_camera['height']:
                best_camera = camera

    return best_camera

def test_logitech_features(camera_id):
    """Test Logitech-specific features"""
    print(f"\nTesting Logitech B525 features on camera {camera_id}...")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Cannot open camera for feature testing")
        return {}

    features = {}

    # Test camera controls
    controls_to_test = {
        'brightness': cv2.CAP_PROP_BRIGHTNESS,
        'contrast': cv2.CAP_PROP_CONTRAST,
        'saturation': cv2.CAP_PROP_SATURATION,
        'hue': cv2.CAP_PROP_HUE,
        'gain': cv2.CAP_PROP_GAIN,
        'exposure': cv2.CAP_PROP_EXPOSURE,
        'auto_exposure': cv2.CAP_PROP_AUTO_EXPOSURE,
        'focus': cv2.CAP_PROP_FOCUS,
        'autofocus': cv2.CAP_PROP_AUTOFOCUS,
        'zoom': cv2.CAP_PROP_ZOOM,
        'white_balance': cv2.CAP_PROP_WB_TEMPERATURE
    }

    for name, prop in controls_to_test.items():
        value = cap.get(prop)
        if value != -1:
            features[name] = value
            print(f"  {name}: {value}")

    cap.release()
    return features

def main():
    """Test USB camera interface"""
    print("USB Camera Interface Test (Logitech B525)")
    print("=" * 50)

    # Find USB cameras only
    usb_cameras = find_usb_cameras()

    if not usb_cameras:
        print("ERROR: No USB cameras found")
        print("Check that your Logitech B525 is connected and recognized by Windows")
        return

    # Select best USB camera
    best_camera = select_logitech_camera(usb_cameras)
    print(f"\nSelected USB Camera {best_camera['id']}: {best_camera['width']}x{best_camera['height']}")

    # Test Logitech features
    features = test_logitech_features(best_camera['id'])

    # Test the camera interface
    print(f"\nTesting USBCameraInterface...")

    camera = USBCameraInterface(
        camera_id=best_camera['id'],
        target_resolution=(best_camera['width'], best_camera['height'])
    )

    try:
        # Start camera
        actual_width, actual_height = camera.start()
        print(f"USB Camera started successfully: {actual_width}x{actual_height}")

        # Capture test frames
        print("Capturing test frames...")
        successful_frames = 0
        for i in range(10):
            frame = camera.get_frame()
            if frame is not None:
                successful_frames += 1
                if i % 2 == 0:  # Print every other frame
                    print(f"  Frame {i+1}: {frame.shape} - OK")
            else:
                print(f"  Frame {i+1}: Failed")
            time.sleep(0.5)

        print(f"Capture success rate: {successful_frames}/10 frames")

        # Save test image
        if successful_frames > 0:
            test_frame = camera.capture_single_frame()
            if test_frame is not None:
                cv2.imwrite("images/samples/logitech_b525_test.jpg", test_frame)
                print("Test image saved: images/samples/logitech_b525_test.jpg")

        # Show camera info
        info = camera.get_info()
        print(f"\nLogitech B525 Camera Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        camera.stop()

    print(f"\nRECOMMENDED CONFIGURATION FOR LOGITECH B525:")
    print(f"CAMERA_ID = {best_camera['id']}")
    print(f"CAMERA_WIDTH = {best_camera['width']}")
    print(f"CAMERA_HEIGHT = {best_camera['height']}")
    print(f"CAMERA_TYPE = 'Logitech B525 HD Webcam'")

if __name__ == "__main__":
    main()

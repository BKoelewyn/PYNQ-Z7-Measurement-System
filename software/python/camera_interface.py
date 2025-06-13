"""
Camera interface module for PYNQ measurement system
Handles camera detection, configuration, and basic operations
"""

import cv2
import numpy as np
import threading
import time


class CameraInterface:
    """Camera interface for Logitech B525 and other USB cameras"""

    def __init__(self, camera_id=0, resolution=(1280, 720)):
        self.camera_id = camera_id
        self.resolution = resolution
        self.cap = None
        self.frame = None
        self.running = False
        self.thread = None

    def start(self):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise Exception(f"Cannot open camera {self.camera_id}")

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Start capture thread
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.start()

        print(f"Camera {self.camera_id} started at {self.resolution}")

    def _capture_loop(self):
        """Continuous frame capture"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame.copy()
            time.sleep(0.01)  # ~100 FPS max

    def get_frame(self):
        """Get latest frame"""
        return self.frame.copy() if self.frame is not None else None

    def stop(self):
        """Stop camera capture"""
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()
        print("Camera stopped")

    def get_camera_info(self):
        """Get camera information"""
        if not self.cap or not self.cap.isOpened():
            return None

        info = {
            'id': self.camera_id,
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'backend': self.cap.getBackendName()
        }
        return info


def detect_cameras():
    """Detect all available cameras"""
    available_cameras = []

    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Get basic info
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            available_cameras.append({
                'id': i,
                'resolution': f"{width}x{height}",
                'width': width,
                'height': height
            })
            cap.release()

    return available_cameras


if __name__ == "__main__":
    # Test camera detection
    print("Detecting cameras...")
    cameras = detect_cameras()

    for camera in cameras:
        print(f"Camera {camera['id']}: {camera['resolution']}")

    if cameras:
        # Test first camera
        print(f"\nTesting Camera {cameras[0]['id']}...")
        camera = CameraInterface(cameras[0]['id'])
        camera.start()

        # Capture for 3 seconds
        time.sleep(3)

        # Get a frame
        frame = camera.get_frame()
        if frame is not None:
            print(f"Successfully captured frame: {frame.shape}")

        camera.stop()
    else:
        print("No cameras detected")

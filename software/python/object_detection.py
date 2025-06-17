"""
PYNQ Z7 Precision Object Measurement System - Object Detection Module
Detects calibration objects (coins, rulers) for automatic calibration
"""

import cv2
import numpy as np
import time
from datetime import datetime
from pathlib import Path
import json

try:
    from config.settings import *
except ImportError:
    # Fallback configuration if settings not available
    CALIBRATION_OBJECTS = {
        'coin': {
            'name': 'US Quarter',
            'diameter_mm': 24.26,
            'color_range_hsv': {'lower': [10, 50, 50], 'upper': [25, 255, 255]},
            'detection_params': {'min_area': 2000, 'max_area': 15000, 'circularity_min': 0.7}
        }
    }


def log_with_timestamp(message):
    """Print message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] DETECTION: {message}")


class ObjectDetector:
    """Object detection for calibration objects"""

    def __init__(self):
        self.detection_history = []
        self.calibration_active = False
        self.last_detection_time = 0
        self.stable_detections = 0

    def detect_objects(self, frame, object_types=None):
        """
        Detect specified objects in frame
        Args:
            frame: Input image frame
            object_types: List of object types to detect (default: all available)
        Returns:
            Dictionary of detected objects with their properties
        """
        if object_types is None:
            object_types = list(CALIBRATION_OBJECTS.keys())

        detected_objects = {}

        for obj_type in object_types:
            if obj_type in CALIBRATION_OBJECTS:
                detections = self._detect_object_type(frame, obj_type)
                if detections:
                    detected_objects[obj_type] = detections
                    log_with_timestamp(f"Detected {len(detections)} {obj_type}(s)")

        return detected_objects

    def _detect_object_type(self, frame, obj_type):
        """Detect specific object type in frame"""
        obj_config = CALIBRATION_OBJECTS[obj_type]

        try:
            if obj_type == 'coin':
                return self._detect_coins(frame, obj_config)
            elif obj_type == 'ruler':
                return self._detect_rulers(frame, obj_config)
            else:
                log_with_timestamp(f"Unknown object type: {obj_type}")
                return []
        except Exception as e:
            log_with_timestamp(f"Error detecting {obj_type}: {e}")
            return []

    def _detect_coins(self, frame, config):
        """Detect circular coins in frame"""
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create color mask
        color_range = config['color_range_hsv']
        lower = np.array(color_range['lower'])
        upper = np.array(color_range['upper'])
        mask = cv2.inRange(hsv, lower, upper)

        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        coins = []
        params = config['detection_params']

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if area < params['min_area'] or area > params['max_area']:
                continue

            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            if circularity < params['circularity_min']:
                continue

            # Get bounding circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            # Check aspect ratio of bounding rectangle
            rect = cv2.boundingRect(contour)
            aspect_ratio = rect[2] / rect[3] if rect[3] > 0 else 0
            if abs(aspect_ratio - 1.0) > params.get('aspect_ratio_tolerance', 0.3):
                continue

            coin_data = {
                'type': 'coin',
                'center': center,
                'radius': radius,
                'diameter_pixels': radius * 2,
                'area_pixels': area,
                'circularity': circularity,
                'confidence': min(1.0, circularity / 0.9),  # Confidence based on circularity
                'contour': contour,
                'real_diameter_mm': config['diameter_mm']
            }

            coins.append(coin_data)
            log_with_timestamp(
                f"Coin detected: center={center}, radius={radius}, confidence={coin_data['confidence']:.3f}")

        # Sort by confidence (best detections first)
        coins.sort(key=lambda x: x['confidence'], reverse=True)
        return coins

    def _detect_rulers(self, frame, config):
        """Detect rectangular rulers in frame"""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create color mask for ruler (typically white/light colored)
        color_range = config['color_range_hsv']
        lower = np.array(color_range['lower'])
        upper = np.array(color_range['upper'])
        mask = cv2.inRange(hsv, lower, upper)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rulers = []
        params = config['detection_params']

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if area < params['min_area'] or area > params['max_area']:
                continue

            # Get bounding rectangle
            rect = cv2.boundingRect(contour)
            x, y, w, h = rect

            # Check aspect ratio (ruler should be long and thin)
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            if aspect_ratio < params['aspect_ratio_min'] or aspect_ratio > params['aspect_ratio_max']:
                continue

            # Check rectangularity (how well the contour fits a rectangle)
            rect_area = w * h
            rectangularity = area / rect_area if rect_area > 0 else 0
            if rectangularity < params['rectangularity_min']:
                continue

            # Determine orientation and measurements
            if w > h:  # Horizontal ruler
                length_pixels = w
                width_pixels = h
                angle = 0
            else:  # Vertical ruler
                length_pixels = h
                width_pixels = w
                angle = 90

            ruler_data = {
                'type': 'ruler',
                'center': (x + w // 2, y + h // 2),
                'length_pixels': length_pixels,
                'width_pixels': width_pixels,
                'angle': angle,
                'area_pixels': area,
                'rectangularity': rectangularity,
                'confidence': min(1.0, rectangularity * (aspect_ratio / 6.0)),  # Confidence based on shape
                'contour': contour,
                'bounding_rect': rect,
                'real_length_mm': config['length_mm']
            }

            rulers.append(ruler_data)
            log_with_timestamp(
                f"Ruler detected: center={ruler_data['center']}, length={length_pixels}px, confidence={ruler_data['confidence']:.3f}")

        # Sort by confidence
        rulers.sort(key=lambda x: x['confidence'], reverse=True)
        return rulers

    def calculate_calibration(self, detected_objects):
        """
        Calculate pixels per millimeter from detected calibration objects
        Returns: dict with calibration data
        """
        calibration_results = {}

        for obj_type, detections in detected_objects.items():
            if not detections:
                continue

            best_detection = detections[0]  # Highest confidence detection

            if obj_type == 'coin':
                # Use coin diameter for calibration
                pixels_per_mm = best_detection['diameter_pixels'] / best_detection['real_diameter_mm']
                calibration_results[obj_type] = {
                    'pixels_per_mm': pixels_per_mm,
                    'reference_size_mm': best_detection['real_diameter_mm'],
                    'measured_size_pixels': best_detection['diameter_pixels'],
                    'confidence': best_detection['confidence'],
                    'method': 'coin_diameter'
                }
                log_with_timestamp(f"Coin calibration: {pixels_per_mm:.3f} pixels/mm")

            elif obj_type == 'ruler':
                # Use ruler length for calibration
                pixels_per_mm = best_detection['length_pixels'] / best_detection['real_length_mm']
                calibration_results[obj_type] = {
                    'pixels_per_mm': pixels_per_mm,
                    'reference_size_mm': best_detection['real_length_mm'],
                    'measured_size_pixels': best_detection['length_pixels'],
                    'confidence': best_detection['confidence'],
                    'method': 'ruler_length'
                }
                log_with_timestamp(f"Ruler calibration: {pixels_per_mm:.3f} pixels/mm")

        return calibration_results

    def draw_detection_overlay(self, frame, detected_objects):
        """Draw detection results on frame"""
        overlay = frame.copy()

        for obj_type, detections in detected_objects.items():
            for detection in detections:
                if obj_type == 'coin':
                    # Draw circle for coin
                    center = detection['center']
                    radius = detection['radius']
                    cv2.circle(overlay, center, radius, (255, 0, 255), 2)  # Magenta circle
                    cv2.circle(overlay, center, 3, (255, 0, 255), -1)  # Center dot

                    # Label
                    label = f"Coin: {detection['real_diameter_mm']}mm"
                    cv2.putText(overlay, label, (center[0] - 50, center[1] - radius - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                elif obj_type == 'ruler':
                    # Draw rectangle for ruler
                    rect = detection['bounding_rect']
                    x, y, w, h = rect
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 0), 2)  # Yellow rectangle

                    # Center point
                    center = detection['center']
                    cv2.circle(overlay, center, 3, (255, 255, 0), -1)

                    # Label
                    label = f"Ruler: {detection['real_length_mm']}mm"
                    cv2.putText(overlay, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        return overlay


def test_object_detection():
    """Test object detection with webcam"""
    log_with_timestamp("Starting object detection test")

    detector = ObjectDetector()
    cap = cv2.VideoCapture(1)  # Use your camera ID

    if not cap.isOpened():
        log_with_timestamp("Cannot open camera for testing")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    log_with_timestamp("Object detection test running. Press 'q' to quit.")
    log_with_timestamp("Available objects: " + ", ".join(CALIBRATION_OBJECTS.keys()))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects
        detected = detector.detect_objects(frame)

        # Draw detection overlay
        display_frame = detector.draw_detection_overlay(frame, detected)

        # Calculate calibration if objects detected
        if detected:
            calibration = detector.calculate_calibration(detected)

            # Display calibration info
            y_offset = 30
            for obj_type, cal_data in calibration.items():
                text = f"{obj_type}: {cal_data['pixels_per_mm']:.2f} px/mm"
                cv2.putText(display_frame, text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25

        # Show instructions
        cv2.putText(display_frame, "Place coin or ruler in view for detection",
                    (10, display_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Object Detection Test', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    log_with_timestamp("Object detection test completed")


if __name__ == "__main__":
    test_object_detection()

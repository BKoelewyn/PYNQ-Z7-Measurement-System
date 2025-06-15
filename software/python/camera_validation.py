"""
Final validation script for Logitech B525 camera
Confirms all functionality needed for measurement system
"""

import cv2
import numpy as np
import time
from camera_interface import USBCameraInterface


def validate_camera_for_measurement():
    """Validate camera is ready for precision measurement"""
    print("Logitech B525 Validation for Precision Measurement")
    print("=" * 55)

    camera = USBCameraInterface(camera_id=1, target_resolution=(640, 480))

    try:
        # Start camera
        actual_width, actual_height = camera.start()
        print(f"Camera Resolution: {actual_width}x{actual_height}")

        # Test frame consistency
        print("\nTesting frame consistency...")
        consistent_frames = 0
        total_frames = 20

        for i in range(total_frames):
            frame = camera.get_frame()
            if frame is not None and frame.shape == (480, 640, 3):
                consistent_frames += 1
            time.sleep(0.1)

        consistency_rate = consistent_frames / total_frames
        print(f"Frame consistency: {consistent_frames}/{total_frames} ({consistency_rate:.1%})")

        # Test image quality
        print("\nTesting image quality...")
        test_frame = camera.capture_single_frame()
        if test_frame is not None:
            # Calculate image statistics
            gray = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            brightness_std = np.std(gray)

            print(f"Image brightness: {mean_brightness:.1f} (std: {brightness_std:.1f})")

            # Test edge detection capability
            edges = cv2.Canny(gray, 50, 150)
            edge_pixels = np.sum(edges > 0)
            edge_percentage = (edge_pixels / (640 * 480)) * 100

            print(f"Edge detection: {edge_pixels} edge pixels ({edge_percentage:.1f}%)")

            # Save validation images
            cv2.imwrite("images/samples/validation_original.jpg", test_frame)
            cv2.imwrite("images/samples/validation_edges.jpg", edges)
            print("Validation images saved")

        # Test camera controls
        print("\nTesting camera controls...")
        info = camera.get_info()
        controllable_features = 0

        if info['brightness'] != -1:
            controllable_features += 1
            print(f"  Brightness control: Available ({info['brightness']})")

        if info['contrast'] != -1:
            controllable_features += 1
            print(f"  Contrast control: Available ({info['contrast']})")

        if info['exposure'] != -1:
            controllable_features += 1
            print(f"  Exposure control: Available ({info['exposure']})")

        print(f"Total controllable features: {controllable_features}")

        # Overall assessment
        print(f"\nVALIDATION RESULTS:")
        print(f"{'=' * 25}")

        if consistency_rate >= 0.8:
            print("Frame Consistency: PASS")
        else:
            print("Frame Consistency: FAIL")

        if test_frame is not None and mean_brightness > 50:
            print("Image Quality: PASS")
        else:
            print("Image Quality: FAIL")

        if edge_percentage > 5:
            print("Edge Detection: PASS")
        else:
            print("Edge Detection: FAIL")

        if controllable_features >= 2:
            print("Camera Controls: PASS")
        else:
            print("Camera Controls: FAIL")

        # Final verdict
        overall_pass = (consistency_rate >= 0.8 and
                        test_frame is not None and
                        mean_brightness > 50 and
                        edge_percentage > 5 and
                        controllable_features >= 2)

        if overall_pass:
            print(f"\nOVERALL: CAMERA READY FOR MEASUREMENT SYSTEM")
            print(f"The Logitech B525 is validated for precision measurements")
        else:
            print(f"\nOVERALL: CAMERA NEEDS ATTENTION")
            print(f"Some issues detected that may affect measurement accuracy")

        return overall_pass

    except Exception as e:
        print(f"VALIDATION ERROR: {e}")
        return False
    finally:
        camera.stop()


if __name__ == "__main__":
    success = validate_camera_for_measurement()
    if success:
        print(f"\nReady to proceed with GUI framework development")
    else:
        print(f"\nResolve camera issues before proceeding")

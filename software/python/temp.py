"""
Visual Camera Test - Show both cameras so you can identify which is which
"""

import cv2
import numpy as np

def show_both_cameras():
    """Show both Camera 0 and Camera 1 side by side so you can identify which is USB vs built-in"""

    print("Opening both cameras for visual identification...")
    print("This will show both camera feeds so you can see which is which")
    print("Press 'q' to quit, 's' to save screenshots")

    # Open both cameras
    cap0 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    if not cap0.isOpened():
        print("Cannot open Camera 0")
        return

    if not cap1.isOpened():
        print("Cannot open Camera 1")
        cap0.release()
        return

    # Set both to same resolution for comparison
    cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\nCamera windows opened:")
    print("- Left side: Camera 0")
    print("- Right side: Camera 1")
    print("\nLook for differences:")
    print("- USB camera (Logitech B525): Usually positioned where you placed it")
    print("- Built-in camera: Usually at top of laptop screen")
    print("- Different viewing angles")
    print("- Different image quality")

    screenshot_count = 0

    while True:
        # Read from both cameras
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if ret0 and ret1:
            # Add labels to frames
            frame0_labeled = frame0.copy()
            frame1_labeled = frame1.copy()

            # Add text labels
            cv2.putText(frame0_labeled, "Camera 0", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame1_labeled, "Camera 1", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Add instructions
            cv2.putText(frame0_labeled, "Press 'q' to quit, 's' to save", (10, 460),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame1_labeled, "Press 'q' to quit, 's' to save", (10, 460),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Combine frames side by side
            combined = np.hstack((frame0_labeled, frame1_labeled))

            # Add separator line
            cv2.line(combined, (640, 0), (640, 480), (255, 255, 255), 2)

            # Add title
            title_img = np.zeros((50, 1280, 3), dtype=np.uint8)
            cv2.putText(title_img, "Camera Identification - Which is your USB camera?",
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Combine title with camera feeds
            final_display = np.vstack((title_img, combined))

            # Show the combined image
            cv2.imshow('Camera Identification Test', final_display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                screenshot_count += 1
                filename = f"camera_comparison_{screenshot_count}.png"
                cv2.imwrite(filename, final_display)
                print(f"Screenshot saved: {filename}")

        elif not ret0:
            print("Camera 0 stopped working")
            break
        elif not ret1:
            print("Camera 1 stopped working")
            break

    # Cleanup
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

    print("\nNow you should know which camera is which!")
    print("Next steps:")
    print("1. If Camera 0 was your USB camera (Logitech B525):")
    print("   - Your settings are CORRECT (CAMERA_ID = 1 should work)")
    print("   - The GUI issue is elsewhere")
    print("")
    print("2. If Camera 1 was your USB camera (Logitech B525):")
    print("   - Keep your current settings (CAMERA_ID = 1)")
    print("   - The GUI should work with the fixed code")
    print("")
    print("3. If you're still not sure:")
    print("   - Try physically covering/disconnecting your USB camera")
    print("   - Run this test again to see which camera disappears")

def test_camera_with_disconnect():
    """Test what happens when you disconnect the USB camera"""
    print("USB Camera Disconnect Test")
    print("=" * 40)
    print("1. Make sure your USB camera is connected")
    print("2. Press Enter to test both cameras")
    input("Press Enter to continue...")

    # Test both cameras initially
    print("\nTesting with USB camera connected:")
    for cam_id in [0, 1]:
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"  Camera {cam_id}: WORKING")
            else:
                print(f"  Camera {cam_id}: Opens but no frame")
            cap.release()
        else:
            print(f"  Camera {cam_id}: Cannot open")

    print("\n3. Now DISCONNECT your USB camera (unplug it)")
    print("4. Press Enter to test again")
    input("Press Enter after disconnecting USB camera...")

    # Test both cameras after disconnect
    print("\nTesting with USB camera disconnected:")
    for cam_id in [0, 1]:
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"  Camera {cam_id}: WORKING (this is your built-in camera)")
            else:
                print(f"  Camera {cam_id}: Opens but no frame")
            cap.release()
        else:
            print(f"  Camera {cam_id}: Cannot open (this was your USB camera)")

    print("\nNow you know which Camera ID corresponds to your USB camera!")

if __name__ == "__main__":
    print("Camera Identification Utility")
    print("Choose a test method:")
    print("1. Visual comparison (shows both cameras)")
    print("2. Disconnect test (unplug USB camera to identify)")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        show_both_cameras()
    elif choice == "2":
        test_camera_with_disconnect()
    else:
        print("Invalid choice. Running visual comparison...")
        show_both_cameras()
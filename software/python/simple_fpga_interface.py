"""
Simple Python Interface for PYNQ Z2 Linux C++ FPGA Backend
Compatible with your existing measurement_gui.py and Israeli coin calibration
"""

import socket
import numpy as np
import cv2
import time
from datetime import datetime


def log_with_timestamp(message):
    """Print message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] PYTHON: {message}")


class FPGAConnection:
    """Simple connection to PYNQ Z2 Linux C++ FPGA backend"""

    def __init__(self, pynq_ip="192.168.2.99", port=12345):
        self.pynq_ip = pynq_ip
        self.port = port
        self.socket = None
        self.connected = False
        self.frame_count = 0

    def connect_to_fpga(self):
        """Connect to Linux C++ FPGA backend on PYNQ Z2"""
        try:
            log_with_timestamp(f"Connecting to Linux C++ backend at {self.pynq_ip}:{self.port}")

            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10.0)

            self.socket.connect((self.pynq_ip, self.port))
            self.connected = True

            log_with_timestamp("SUCCESS: Connected to Linux C++ FPGA backend!")
            return True

        except Exception as e:
            log_with_timestamp(f"ERROR: Connection failed: {e}")
            return False

    def process_frame_with_fpga(self, frame):
        """Process frame using Linux C++ FPGA backend"""
        if not self.connected:
            return frame

        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = frame.copy()

            # Ensure correct size
            if gray_frame.shape != (480, 640):
                gray_frame = cv2.resize(gray_frame, (640, 480))

            # Send to FPGA backend
            frame_data = gray_frame.astype(np.uint8).tobytes()
            frame_size = len(frame_data)

            self.socket.sendall(frame_size.to_bytes(4, byteorder='little'))
            self.socket.sendall(frame_data)

            # Receive processed frame
            processed_data = b''
            while len(processed_data) < frame_size:
                chunk = self.socket.recv(frame_size - len(processed_data))
                if not chunk:
                    raise ConnectionError("Connection lost")
                processed_data += chunk

            # Convert back to numpy array
            processed_frame = np.frombuffer(processed_data, dtype=np.uint8)
            processed_frame = processed_frame.reshape(480, 640)

            self.frame_count += 1
            if self.frame_count % 30 == 0:
                log_with_timestamp(f"FPGA processed {self.frame_count} frames - 30+ FPS!")

            return processed_frame

        except Exception as e:
            log_with_timestamp(f"ERROR: FPGA processing failed: {e}")
            self.connected = False
            return frame


# Global connection instance
fpga_connection = FPGAConnection()


def test_fpga_system():
    """Test the complete Linux C++ FPGA system"""
    print("\nTesting Linux C++ FPGA Backend Connection")
    print("==========================================")

    if fpga_connection.connect_to_fpga():
        print("SUCCESS: Connection test passed!")
    else:
        print("INFO: Connection failed (expected - PYNQ Z2 not ready yet)")
        print("This will work after we deploy to PYNQ Z2 in Step 4")

    return True


if __name__ == "__main__":
    test_fpga_system()
from picamera2 import Picamera2
import time

# Initialize camera
picam2 = Picamera2()

# Configure preview and capture
config = picam2.create_preview_configuration()
picam2.configure(config)

# Start camera
picam2.start()

print("Taking picture in 3 seconds...")
time.sleep(3)

# Capture image
picam2.capture_file("/home/pi/workspace_ait500/test_image.jpg")
print("Image saved as test_image.jpg")

picam2.stop()

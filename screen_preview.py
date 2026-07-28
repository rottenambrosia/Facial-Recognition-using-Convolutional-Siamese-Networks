import os
import cv2

PHONE_STREAM_URL = os.getenv("PHONE_STREAM_URL", "http://192.168.0.108:8080/video")
print("Using phone stream URL:", PHONE_STREAM_URL)

cap = cv2.VideoCapture(PHONE_STREAM_URL)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open the phone stream.\n"
        "Check that IP Webcam is running on your phone and that the URL is correct."
    )

# Lower the resolution and use a simpler loop to reduce lag.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

print("Phone stream opened successfully")
print("Press q to stop")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read a frame from the phone stream")
        break

    frame = cv2.resize(frame, (480, 360))
    cv2.imshow("Phone Camera Feed", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == 27:  # Esc
        break

cap.release()
cv2.destroyAllWindows()

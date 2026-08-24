import cv2
import time
from ultralytics import YOLO


MODEL_NAME = "yolo26n.pt"
CAMERA_INDICES = [0, 1]
IMAGE_SIZE = 416
PERSON_CLASS = 0
CONFIDENCE = 0.40


# --------------------------------------------------
# Load YOLO26n
# --------------------------------------------------

print("Loading YOLO26n...")

# Ultralytics automatically downloads yolo26n.pt
# if it is not already available locally.
model = YOLO(MODEL_NAME)

print("Model loaded successfully.")


# --------------------------------------------------
# Open first webcam
# --------------------------------------------------

camera_index = CAMERA_INDICES[1]

cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"Could not open camera {camera_index}")
    exit()

# Capture resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print(f"Using camera {camera_index}")
print("Press 'c' to switch camera")
print("Press 'q' to quit")


# --------------------------------------------------
# FPS variables
# --------------------------------------------------

prev_time = time.perf_counter()
fps = 0


# --------------------------------------------------
# Main loop
# --------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame.")
        break

    # --------------------------------------------------
    # YOLO inference
    # --------------------------------------------------

    results = model.predict(
        source=frame,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        classes=[PERSON_CLASS],
        device="cpu",
        verbose=False
    )

    # --------------------------------------------------
    # Draw detections
    # --------------------------------------------------

    annotated_frame = results[0].plot()

    # --------------------------------------------------
    # Calculate FPS
    # --------------------------------------------------

    current_time = time.perf_counter()

    elapsed = current_time - prev_time

    if elapsed > 0:
        fps = 1 / elapsed

    prev_time = current_time

    # --------------------------------------------------
    # Display FPS and camera information
    # --------------------------------------------------

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Camera: {camera_index}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("YOLO26n Person Detection", annotated_frame)

    # --------------------------------------------------
    # Keyboard controls
    # --------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord("q"):
        break

    # Switch camera
    elif key == ord("c"):

        cap.release()

        # Try the other camera
        if camera_index == 0:
            camera_index = 1
        else:
            camera_index = 0

        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print(f"Could not open camera {camera_index}")

            # Return to previous camera
            if camera_index == 0:
                camera_index = 1
            else:
                camera_index = 0

            cap = cv2.VideoCapture(camera_index)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print(f"Switched to camera {camera_index}")


# --------------------------------------------------
# Cleanup
# --------------------------------------------------

cap.release()
cv2.destroyAllWindows()

print("Program stopped.")
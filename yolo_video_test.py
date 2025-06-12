from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model
model = YOLO("yolov8n.pt")  

# Define colors for known classes (optional customization)
colors = {
    "person": (0, 0, 255),  # Red for person
}

def get_color(label):
    return colors.get(label, (0, 255, 0))  # Default to green

# Open video file or webcam
video_path = 0
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Process detections from the first (and only) result
    result = results[0]
    boxes = result.boxes

    person_count = 0

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            color = get_color(label)

            if label == "person":
                person_count += 1

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show person count on screen
    cv2.putText(frame, f"People detected: {person_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display result
    cv2.imshow("YOLO Object Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

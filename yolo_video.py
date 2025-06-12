from ultralytics import YOLO
import cv2

# Load YOLO model (Use a trained leaf detection model)
#model = YOLO("runs/detect/train/weights/best.pt")  # Replace "best.pt" with your trained model
model = YOLO("yolo11n.pt")

# Open video file or capture from webcam
video_path = 0 # Ganti dengan path video Anda atau gunakan 0 untuk webcam
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    # Run YOLO detection
    results = model(frame)

    # Draw bounding boxes and classify color
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = map(int, box)  # Bounding box coordinates
            label = model.names[cls]  # Class label (e.g., Green, Yellow, Brown)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show video
    cv2.imshow("YOLO Leaf Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

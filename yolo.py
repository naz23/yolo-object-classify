from ultralytics import YOLO
import cv2

# Load YOLO model (Use a trained leaf detection model)
model = YOLO("runs/detect/train/weights/best.pt")  # Replace "best.pt" with your trained model

# Load image
image = cv2.imread("images/t2.jpg")

# Run YOLO detection
results = model(image)

# Draw bounding boxes and classify color
for result in results:
    for box in result.boxes.data:
        x1, y1, x2, y2, conf, cls = map(int, box)  # Bounding box coordinates
        label = model.names[cls]  # Class label (e.g., Green, Yellow, Brown)

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#Show image
cv2.imshow("YOLO Leaf Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

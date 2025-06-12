import cv2
import numpy as np
import os

# Paths
image_folder = "dataset/images/train"  # Change this to your dataset folder
output_folder = "dataset/labels/train"  # Output folder for YOLO labels

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Define HSV color ranges for leaves
color_ranges = {
    "Green": (np.array([35, 40, 40]), np.array([90, 255, 255])),
    "Yellow": (np.array([20, 100, 100]), np.array([30, 255, 255])),
    "Brown": (np.array([10, 50, 20]), np.array([20, 255, 200])),
}

# Assign class IDs
class_ids = {"Green": 0, "Yellow": 1, "Brown": 2, "Unknown": 3}

# Shape Filtering Thresholds
MIN_ASPECT_RATIO = 0.3  # Minimum aspect ratio (Width / Height)
MAX_ASPECT_RATIO = 3.5  # Maximum aspect ratio
MIN_SOLIDITY = 0.5  # Solidity (Contour Area / Convex Hull Area)
MIN_EXTENT = 0.3  # Extent (Contour Area / Bounding Box Area)

# Get list of images
image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    label_path = os.path.join(output_folder, image_file.replace(".jpg", ".txt"))

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load {image_path}, skipping...")
        continue

    h, w, _ = image.shape  # Get image dimensions

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create masks for each color
    masks = {color: cv2.inRange(hsv, lower, upper) for color, (lower, upper) in color_ranges.items()}

    # Find contours for each leaf color
    leaf_detections = []
    for color, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Ignore small noise
                x, y, w_box, h_box = cv2.boundingRect(contour)
                aspect_ratio = w_box / float(h_box)  # Width / Height
                extent = cv2.contourArea(contour) / (w_box * h_box)  # Fill ratio
                hull = cv2.convexHull(contour)
                solidity = cv2.contourArea(contour) / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0

                # Apply shape filtering
                if MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO and solidity >= MIN_SOLIDITY and extent >= MIN_EXTENT:
                    leaf_detections.append((x, y, w_box, h_box, color))

    # If no valid leaf is detected, classify as "Unknown"
    if not leaf_detections:
        x, y, w_box, h_box = 0, 0, w, h  # Full image
        leaf_detections.append((x, y, w_box, h_box, "Unknown"))

    # Save annotations in YOLO format
    with open(label_path, "w") as f:
        for x, y, w_box, h_box, color in leaf_detections:
            x_center = (x + w_box / 2) / w
            y_center = (y + h_box / 2) / h
            width = w_box / w
            height = h_box / h

            f.write(f"{class_ids[color]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            # Draw bounding box for visualization
            color_bgr = (0, 255, 0) if color == "Green" else (0, 255, 255) if color == "Yellow" else (42, 42, 165) if color == "Brown" else (0, 0, 255)
            cv2.rectangle(image, (x, y), (x + w_box, y + h_box), color_bgr, 2)
            cv2.putText(image, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

    print(f"Saved annotation: {label_path}")

    # Optional: Display results
    cv2.imshow("Leaf Detection", image)
    cv2.waitKey(500)  # Show each image for 500ms

cv2.destroyAllWindows()
print("All images processed!")

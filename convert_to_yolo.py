import os
import cv2

# Paths
base_folder = "dataset/images/train"  # Base folder containing class subdirectories
output_folder = "dataset/labels/train"  # Output folder for YOLO labels

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process each class folder
class_folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]

for class_id, class_name in enumerate(class_folders):
    class_folder = os.path.join(base_folder, class_name)
    class_output_folder = os.path.join(output_folder, class_name)
    os.makedirs(class_output_folder, exist_ok=True)

    # Process each image in the class folder
    image_files = [f for f in os.listdir(class_folder) if f.endswith(".jpg")]

    for image_file in image_files:
        image_path = os.path.join(class_folder, image_file)
        label_path = os.path.join(class_output_folder, image_file.replace(".jpg", ".txt"))

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load {image_path}, skipping...")
            continue

        h, w, _ = image.shape

        # Treat the entire image as a single bounding box
        x, y, w_box, h_box = 0, 0, w, h  # Full image dimensions

        # Save annotation in YOLO format
        with open(label_path, "w") as f:
            x_center = (x + w_box / 2) / w
            y_center = (y + h_box / 2) / h
            width = w_box / w
            height = h_box / h
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        print(f"Saved annotation: {label_path}")

print("All images processed!")
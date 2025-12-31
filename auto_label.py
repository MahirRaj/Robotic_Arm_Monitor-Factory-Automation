import cv2
import os
from ultralytics import YOLO

# --- AUTOMATIC PATH FINDING ---
script_folder = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the script
IMAGES_DIR = os.path.join(script_folder, "unlabeled_images_v2")
MODEL_PATH = os.path.join(script_folder, "final.pt") 

print(f"Looking for images in: {IMAGES_DIR}")
print(f"Using model: {MODEL_PATH}")

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print("❌ ERROR: teacher.pt not found! Go to runs/detect/teacher_run_fast/weights, rename best.pt to teacher.pt and move it here.")
    exit()

# Load model
model = YOLO(MODEL_PATH)

# Get list of images
image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
print(f"Found {len(image_files)} images to label.")

count = 0
for img_file in image_files:
    img_path = os.path.join(IMAGES_DIR, img_file)
    
    # Run prediction
    results = model.predict(img_path, conf=0.25, verbose=False)
    
    # Prepare text file name (e.g., frame_0.txt)
    txt_filename = os.path.splitext(img_file)[0] + ".txt"
    txt_path = os.path.join(IMAGES_DIR, txt_filename)
    
    with open(txt_path, "w") as f:
        for r in results:
            for box in r.boxes:
                # Get Normalized Coordinates
                cls_id = int(box.cls[0])
                x, y, w, h = box.xywhn[0]
                
                # Write: class x y w h
                f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    
    count += 1
    if count % 100 == 0:
        print(f"Labeled {count} images...")

print("✅ Auto-labeling complete! Check your folder for .txt files.")
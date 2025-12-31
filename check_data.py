import os
import glob

# POINT THIS TO YOUR DATASET FOLDER
DATASET_PATH = r"D:\Mahir\AI Projects\Robotic Arm Monitor\dataset_v4"

def check_labels(folder_name):
    label_dir = os.path.join(DATASET_PATH, folder_name, "labels")
    files = glob.glob(os.path.join(label_dir, "*.txt"))
    
    print(f"Scanning {len(files)} files in {folder_name}...")
    
    errors = 0
    for file_path in files:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"❌ FORMAT ERROR: {file_path} (Line {i+1})")
                errors += 1
                continue
                
            class_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            
            # CHECK 1: Class ID must be 0
            if class_id != 0:
                print(f"❌ BAD CLASS ID ({class_id}): {file_path}")
                errors += 1
            
            # CHECK 2: Coordinates must be normalized (0 to 1)
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                print(f"❌ BAD COORDINATES: {file_path} (Line {i+1}) -> {x} {y} {w} {h}")
                errors += 1

    if errors == 0:
        print(f"✅ {folder_name} is clean!")
    else:
        print(f"⚠️ Found {errors} errors in {folder_name}.")

# Run check on train and valid folders
check_labels("train")
check_labels("valid")
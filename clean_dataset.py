import os
import glob

# POINT THIS TO YOUR DATASET FOLDER
DATASET_PATH = r"D:\Mahir\AI Projects\Robotic Arm Monitor\dataset_v4"

def clean_folder(folder_name):
    label_dir = os.path.join(DATASET_PATH, folder_name, "labels")
    files = glob.glob(os.path.join(label_dir, "*.txt"))
    
    print(f"🧹 Cleaning {len(files)} files in {folder_name}...")
    
    fixed_files = 0
    deleted_lines = 0
    
    for file_path in files:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        file_changed = False
        
        for line in lines:
            parts = line.strip().split()
            
            # Case 1: Empty line -> Skip it
            if len(parts) == 0:
                file_changed = True
                deleted_lines += 1
                continue
                
            # Case 2: Valid line (5 numbers) -> Keep it
            if len(parts) == 5:
                try:
                    # Verify they are numbers
                    int(parts[0])
                    [float(x) for x in parts[1:]]
                    new_lines.append(line)
                except ValueError:
                    file_changed = True
                    deleted_lines += 1
            
            # Case 3: Extra data (e.g., confidence score at the end) -> Fix it
            elif len(parts) > 5:
                try:
                    # Take only the first 5 parts (Class x y w h)
                    clean_line = " ".join(parts[:5]) + "\n"
                    
                    # Verify they are numbers
                    int(parts[0])
                    [float(x) for x in parts[1:5]]
                    
                    new_lines.append(clean_line)
                    file_changed = True
                except ValueError:
                    deleted_lines += 1
                    file_changed = True
            
            # Case 4: Broken line -> Skip it
            else:
                file_changed = True
                deleted_lines += 1

        # Save back only if changes were made
        if file_changed:
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
            fixed_files += 1

    print(f"✅ Fixed {fixed_files} files.")
    print(f"🗑️ Removed {deleted_lines} bad lines.")

    # CRITICAL: Delete the cache so YOLO re-scans the files
    cache_path = os.path.join(label_dir, "..", "labels.cache") # YOLO saves it in the parent usually, or labels folder
    if os.path.exists(os.path.join(label_dir, "labels.cache")):
        os.remove(os.path.join(label_dir, "labels.cache"))
    if os.path.exists(cache_path):
        os.remove(cache_path)
    print("🔄 Cache cleared.")

# Run cleaning
clean_folder("train")
clean_folder("valid")
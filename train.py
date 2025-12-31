from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # --- CONFIGURATION ---
    # 1. Use the SMART model as the starting point
    # Ensure 'final.pt' is in the same folder, or verify the path.
    model = YOLO('final.pt') 
    
    # 2. Point to the CLEAN dataset (v4)
    # This must match the folder we just cleaned
    dataset_yaml = 'dataset_v4/data.yaml'

    print(f"🚀 Starting training on: {dataset_yaml}")
    
    # 3. Train
    results = model.train(
        data=dataset_yaml, 
        epochs=50, 
        imgsz=640, 
        batch=16, 
        name='final_v2_run', 
        workers=2,
        exist_ok=True  # Allows overwriting if you run it multiple times
    )
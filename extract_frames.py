import os
import cv2

# --- AUTOMATIC PATH FINDING ---
# Get the folder where THIS script is located
script_folder = os.path.dirname(os.path.abspath(__file__))

# Build the full path to the video file
VIDEO_PATH = os.path.join(script_folder, "new_camera.mp4")
OUTPUT_FOLDER = os.path.join(script_folder, "unlabeled_images_v2")

# --- DEBUG PRINT ---
print(f"Looking for video at: {VIDEO_PATH}")
if not os.path.exists(VIDEO_PATH):
    print("❌ ERROR: File still not found!")
    exit()
else:
    print("✅ File found! Starting extraction...")

# --- REST OF YOUR CODE ---
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

cap = cv2.VideoCapture(VIDEO_PATH)
# ... (Keep the rest of your original loop code below) ...
count = 0
saved_count = 0
FRAME_RATE = 1

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    if count % FRAME_RATE == 0:
        filename = os.path.join(OUTPUT_FOLDER, f"frame_{saved_count}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1
        
        if saved_count % 100 == 0:
            print(f"Extracted {saved_count} images...")
            
    count += 1

cap.release()
print(f"✅ DONE! Extracted {saved_count} images.")
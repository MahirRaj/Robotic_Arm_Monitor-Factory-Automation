import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import numpy as np

# --- SETTINGS ---
CAMERA_SOURCE = 0
MODEL_PATH = 'best.pt'

# 1. Load the Model using SAHI's wrapper
#    This prepares the YOLO model to work with the "Slicing" logic
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=MODEL_PATH,
    confidence_threshold=0.25,
    device="cuda:0" # Uses your RTX 4060! Change to "cpu" if needed.
)

cap = cv2.VideoCapture(CAMERA_SOURCE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print("Starting SAHI High-Accuracy Monitor...")

while True:
    success, frame = cap.read()
    if not success:
        break

    # 2. RUN SLICED PREDICTION
    #    - slice_height/width: We cut the image into 512x512 squares.
    #    - overlap: We make squares overlap by 20% so we don't cut an arm in half.
    result = get_sliced_prediction(
        frame,
        detection_model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        verbose=0
    )

    # 3. DRAW RESULTS
    #    SAHI gives us a list of predictions. We loop through them manually.
    object_prediction_list = result.object_prediction_list
    
    arm_count = len(object_prediction_list)

    for prediction in object_prediction_list:
        # Get the box coordinates
        bbox = prediction.bbox
        x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
        
        # Draw precise Green Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Label
        score = prediction.score.value
        label = f"Arm: {int(score * 100)}%"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw Status
    cv2.putText(frame, f"Arms Detected: {arm_count}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("SAHI Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
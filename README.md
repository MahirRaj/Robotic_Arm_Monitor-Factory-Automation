# 🏭 Robotic Arm Monitor (Factory Automation AI)

A real-time AI monitoring system designed to detect, track, and analyze the operational status of robotic arms in a factory environment. This project utilizes **YOLOv11** for high-precision object detection and **Flask** for low-latency live streaming to mobile devices.

## 🚀 Key Features

* **Real-Time Detection:** Uses a custom-trained YOLOv11 model (`final_v2.pt`) to detect robotic arms, optimized for high-resolution scanning (`1920x1080`).
* **Smart Motion Logic:**
    * 🟢 **Active:** Arm is moving normally.
    * 🟡 **Idle:** Arm has paused briefly (< 5 seconds).
    * 🔴 **STOPPED:** Arm has been stationary for >5 seconds (Critical Alert).
* **Mobile Live Stream:** Built-in Flask server streams the analyzed video feed to any mobile device on the same WiFi network.
* **Low-Latency Performance:** Video is automatically resized (800px width) and compressed to ensure smooth, buffer-free streaming on mobile browsers.
* **High-Sensitivity Mode:** Tuned with low confidence thresholds (`0.05`) to detect distant or partially obscured robotic arms in the background.

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **AI/ML:** Ultralytics YOLOv11 (Custom Trained)
* **Computer Vision:** OpenCV, MSS (High-speed Screen Capture)
* **Web Framework:** Flask (Video Streaming)
* **Tracking:** BoT-SORT / ByteTrack

## 📂 Project Structure

```text
├── stream_screen.py        # Main Application (AI Logic + Web Server)
├── train.py                # Training script for the YOLO model
├── final_v2.pt             # Custom trained model weights
├── extract_frames.py       # Utility to extract training images from video
├── auto_label.py           # Utility to auto-label datasets
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
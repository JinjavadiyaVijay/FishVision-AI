<div align="center">
<img src="assets/6eb57068-116b-11ee-a55a-9335f156a1e7 (2).gif" alt="Fish Detection System" width="100%">

<br/>

# 🐠 FishVision-AI System

**Real-time fish species identification powered by YOLOv8 and OAK-D Pro**

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

</div>

---

## ✨ Overview

FishVision-AI is an intelligent pipeline for aquatic species monitoring. It leverages state-of-the-art YOLOv8 object detection combined with stereoscopic depth sensing from OAK-D Pro hardware. 

The system supports identifying 13 unique fish species and provides accurate tracking, length estimation, and deduplication logic, helping you get a complete snapshot of the marine ecosystem.

## 🚀 Quick Start

**1 — Clone**
```bash
git clone https://github.com/YOUR_USERNAME/FishVision-AI.git
cd FishVision-AI
```

**2 — Install Requirements**
```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

**3 — Place Models**
Download `best.pt` (Fish species model) and place it in the `models/` folder.
If deploying with the OAK-D Pro, ensure `yolov8n.pt` (Veto gate model) is also available.

**4 — Run Options**

* **Streamlit Dashboard (Photo Inference):**
  ```bash
  streamlit run app.py
  ```
  *(Alternatively, you can right-click and run `run_app.ps1` on Windows)*

* **Real-time Camera Pipeline (OAK-D Pro required):**
  ```bash
  python -m src.oak_runner
  ```

---

## 🗂️ Project Structure

```
FishVision-AI/
├── app.py                   # Streamlit web application
├── run_app.ps1              # Quick launch script for Windows
├── data.yaml                # YOLO training configuration
├── requirements.txt         # Project dependencies
├── src/                     # Core system modules (Tracking, Camera, Inference)
├── models/                  # YOLOv8 weight files
├── logs/                    # Event logs, CSVs, and capture images
└── assets/                  # Documentation assets
```

## 📄 License
This project is **MIT licensed** — see [LICENSE](LICENSE). Dataset (if used) may have separate licensing.

🚗 Real-Time Driver Drowsiness Detection System
📌 Overview

This project implements a real-time driver drowsiness detection system using computer vision techniques. It monitors eye closure and yawning using facial landmark analysis and computes a dynamic fatigue score over time.

The system detects early signs of fatigue and triggers an alert to prevent potential accidents.

🎯 Key Features

👁 Eye Aspect Ratio (EAR) based drowsiness detection

😮 Mouth Aspect Ratio (MAR) yawning detection

🔔 Real-time alarm system

📊 Event logging to CSV file

📈 Dynamic fatigue scoring model with time decay

📉 Statistical analysis and visualization

🧠 How It Works

The system follows this pipeline:

Camera → Face Mesh (MediaPipe) →
EAR & MAR Calculation →
Threshold-Based Detection →
Event Logging →
Dynamic Fatigue Score Modeling →
Alarm Trigger

🛠 Technologies Used

Python

OpenCV

MediaPipe

NumPy

Pandas

Matplotlib

⚙ Installation
1️⃣ Clone the Repository
git clone https://github.com/yourusername/driver-drowsiness-detection.git
cd driver-drowsiness-detection

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Detection System
python main.py

4️⃣ Run Data Analysis
python analysis.py

📈 Fatigue Modeling Approach

A weighted fatigue scoring model is implemented:

+2 for prolonged eye closure (drowsiness)

+1 for yawning events

Gradual fatigue decay over time

This enables dynamic tracking of fatigue progression rather than simple event detection.

🚀 Future Improvements

Head pose estimation

Deep learning-based eye state classification

Raspberry Pi deployment

Web-based interface

Real-world vehicle testing

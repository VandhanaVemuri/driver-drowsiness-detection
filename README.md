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

import cv2
import mediapipe as mp
import numpy as np
import datetime
import csv
import os
import winsound
import threading

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]

EAR_THRESHOLD = 0.20
MAR_THRESHOLD = 0.60
CONSEC_FRAMES = 15

frame_counter = 0
alarm_on = False

log_file = "drowsiness_log.csv"

if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Event", "Value"])

def calculate_EAR(eye_points):
    A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
    B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
    C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
    return (A + B) / (2.0 * C)

def calculate_MAR(mouth_points):
    vertical = np.linalg.norm(np.array(mouth_points[0]) - np.array(mouth_points[1]))
    horizontal = np.linalg.norm(np.array(mouth_points[2]) - np.array(mouth_points[3]))
    return vertical / horizontal

def play_alarm():
    winsound.PlaySound("assets/alarm.wav", winsound.SND_FILENAME)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            left_eye_points = []
            right_eye_points = []
            mouth_points = []

            for idx in LEFT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                left_eye_points.append((x, y))

            for idx in RIGHT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                right_eye_points.append((x, y))

            for idx in MOUTH:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                mouth_points.append((x, y))
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

            left_EAR = calculate_EAR(left_eye_points)
            right_EAR = calculate_EAR(right_eye_points)
            avg_EAR = (left_EAR + right_EAR) / 2.0

            MAR = calculate_MAR(mouth_points)

            cv2.putText(frame, f"EAR: {avg_EAR:.2f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(frame, f"MAR: {MAR:.2f}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # Drowsiness detection
            if avg_EAR < EAR_THRESHOLD:
                frame_counter += 1

                if frame_counter >= CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!",
                                (30, 120),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 255),
                                3)

                    if not alarm_on:
                        alarm_on = True
                        timestamp = datetime.datetime.now()
                        with open(log_file, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([timestamp, "Drowsiness", avg_EAR])

                        threading.Thread(target=play_alarm).start()

            else:
                frame_counter = 0
                alarm_on = False

            # Yawning detection
            if MAR > MAR_THRESHOLD:
                cv2.putText(frame, "YAWNING DETECTED!",
                            (30, 160),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 0, 0),
                            3)

                timestamp = datetime.datetime.now()
                with open(log_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp, "Yawning", MAR])

    cv2.imshow("Advanced Drowsiness Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import tensorflow as tf
import json
import datetime
import csv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("BASE_DIR:", BASE_DIR)
MODEL_PATH = os.path.join(BASE_DIR, "face_model.h5")  # or .keras
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv")
print("Loading model:", MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, "r") as f:
    labels = json.load(f)
print("Loaded labels:", labels)


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

IMG_SIZE = 128
CONF_THRESHOLD = 0.70
marked = set()

if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

def mark_attendance(name):
    if name in marked:
        return
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, date, time])
    marked.add(name)
    print(f"[ATTENDANCE MARKED] {name} at {time} on {date}")


cap = cv2.VideoCapture(0)
print("Starting webcam... Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 1:
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        faces = [faces[0]]

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face_norm = face_resized / 255.0
        face_input = np.expand_dims(face_norm, axis=[0, -1])
        
        preds = model.predict(face_input, verbose=0)
        class_index = np.argmax(preds)
        confidence = preds[0][class_index]

        if confidence < CONF_THRESHOLD:
            predicted_name = "Unknown"
        else:
            predicted_name = labels[class_index]
            if predicted_name != "Unknown":
                mark_attendance(predicted_name)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        cv2.rectangle(frame, (x, y-35), (x+w, y), (0,255,0), -1)
        text = f"{predicted_name} ({confidence*100:.1f}%)"
        cv2.putText(frame, text, (x+5, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)

    cv2.rectangle(frame, (0,0), (frame.shape[1], 40), (0,150,255), -1)
    cv2.putText(frame, "ATTENDANCE SYSTEM", (10,28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

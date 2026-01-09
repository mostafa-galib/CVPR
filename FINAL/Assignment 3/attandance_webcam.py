import os
import cv2
import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("BASE_DIR:", BASE_DIR)
MODEL_PATH = os.path.join(BASE_DIR, "attendance_model.keras")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")

IMG_SIZE = 224
CONF_THRESHOLD = 0.80

print("Loading model from:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    labels = json.load(f)
print("Loaded labels:", labels)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("Webcam not accessible")
print("Webcam started. Press Q to quit.")

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
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = preprocess_input(face.astype(np.float32))
        face = np.expand_dims(face, axis=0)

        preds = model.predict(face, verbose=0)[0]
        idx = np.argmax(preds)
        conf = preds[idx]

        if conf >= CONF_THRESHOLD:
            label = f"{labels[idx]} ({conf*100:.1f}%)"
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Face Identification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

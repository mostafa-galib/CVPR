import cv2
import numpy as np
import tensorflow as tf
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mnist_cnn_model.h5")

model = tf.keras.models.load_model(MODEL_PATH)
print("CNN model loaded successfully")

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    exit()

print("Put the digit inside the box. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    box_size = 300
    x1 = w // 2 - box_size // 2
    y1 = h // 2 - box_size // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    label = "Digit: ?"
    display_digit = None

    if contours:
        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) > 800:
            x, y, cw, ch = cv2.boundingRect(largest)

            if cw > 20 and ch > 20:

                digit = thresh[y:y+ch, x:x+cw]
                
                size = max(cw, ch)
                square = np.zeros((size, size), dtype=np.uint8)
                x_off = (size - cw) // 2
                y_off = (size - ch) // 2
                square[y_off:y_off+ch, x_off:x_off+cw] = digit

                display_digit = cv2.resize(
                    square, (200, 200),
                    interpolation=cv2.INTER_NEAREST
                )

                digit_28 = cv2.resize(
                    square, (28, 28),
                    interpolation=cv2.INTER_AREA
                )

                digit_norm = digit_28.astype("float32") / 255.0
                digit_input = digit_norm.reshape(1, 28, 28, 1) 

                probs = model.predict(digit_input, verbose=0)[0]
                pred = int(np.argmax(probs))
                conf = float(probs[pred])

                if conf > 0.6:
                    label = f"Digit: {pred} ({conf:.2f})"
                else:
                    label = "Digit: ?"

    if display_digit is not None:
        cv2.imshow("Processed Digit", display_digit)

    cv2.putText(
        frame,
        label,
        (x1 + 10, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

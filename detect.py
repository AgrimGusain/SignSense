# detect.py

import time
import cv2
from ultralytics import YOLO
from postprocess import add_detected_letters, clean_letters, correct_with_nlp, clear_buffer

# === CONFIG ===
MODEL_PATH = "best_pro.pt"  # 🔁 Replace with your actual model path
PAUSE_THRESHOLD = 1.0         # Seconds to wait after last detection

# === Load YOLO model ===
model = YOLO(MODEL_PATH)

# === Inference + Drawing Function ===
def detect_letters_from_frame(frame):
    results = model(frame, verbose=False)[0]  # Run inference
    letters = []

    for box in results.boxes:
        class_id = int(box.cls[0])
        label = model.names[class_id]
        letters.append(label)

        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        label_text = f"{label} {confidence:.2f}"
        cv2.putText(frame, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return letters

# === Main Loop ===
def main():
    cap = cv2.VideoCapture(0)
    last_detection_time = time.time()

    print("🖐️ Show your sign letters — press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Detect letters and draw boxes
        detected_letters = detect_letters_from_frame(frame)

        # 2. Add detected letters to buffer
        if detected_letters:
            add_detected_letters(detected_letters)
            last_detection_time = time.time()

        # 3. If user pauses, process buffer
        if time.time() - last_detection_time > PAUSE_THRESHOLD:
            cleaned = clean_letters()
            if cleaned:
                final_sentence = correct_with_nlp(cleaned)
                print("🧠 NLP Output:", final_sentence)
                clear_buffer()

        # 4. Show webcam feed
        cv2.imshow("Sign Language Detection", frame)

        # 5. Quit if user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# === Run ===
if __name__ == "__main__":
    main()

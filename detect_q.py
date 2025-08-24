# python detect_q.py --use_qnn

import time
import cv2
import torch
from ultralytics import YOLO
from postprocess import add_detected_letters, clean_letters, correct_with_nlp, clear_buffer
from quantum_preprocessor import DenoiseQNN  # Your custom QNN
import argparse

# === CONFIG ===
MODEL_PATH = "best_pro.pt"
DENOISER_PATH = "qnn_denoiser.pth"
PAUSE_THRESHOLD = 1.0

# === Parse arguments ===
parser = argparse.ArgumentParser(description="Sign Language Detection with optional Quantum Denoiser")
parser.add_argument('--use_qnn', action='store_true', help='Use quantum denoiser')
args = parser.parse_args()

USE_QNN = args.use_qnn

# === Load YOLOv8 Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO(MODEL_PATH)
model.to(device)

# === Load QNN Denoiser (if enabled) ===
if USE_QNN:
    print("Warning: Quantum denoiser enabled, but detection is performed on original frame due to low resolution output.")
    qnn_model = DenoiseQNN().to(device)
    qnn_model.load_state_dict(torch.load(DENOISER_PATH, map_location=device))
    qnn_model.eval()

def detect_letters_from_frame(frame):
    results = model(frame, verbose=False)[0]
    letters = []

    for box in results.boxes:
        class_id = int(box.cls[0])
        label = model.names[class_id]
        letters.append(label)

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"{label} {confidence:.2f}"
        cv2.putText(frame, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return letters

def main():
    cap = cv2.VideoCapture(0)
    last_detection_time = time.time()

    print("🖐️ Show your sign letters — press 'q' to quit.")
    print(f"Quantum denoiser enabled: {USE_QNN}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        final_sentence = ""

        # Resize frame to 640x640 for display and original processing
        frame_resized = cv2.resize(frame, (640, 640))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        if USE_QNN:
            # Run denoiser on 32x32 resized frame (for demonstration)
            frame_small = cv2.resize(frame_rgb, (32, 32))
            frame_tensor = torch.from_numpy(frame_small).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            frame_tensor = frame_tensor.to(device)
            with torch.no_grad():
                denoised_tensor = qnn_model(frame_tensor)
                cleaned_frame = denoised_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                cleaned_frame = (cleaned_frame * 255).clip(0, 255).astype("uint8")
            # Detection is performed on original frame_rgb due to resolution limits
            detected_letters = detect_letters_from_frame(frame_rgb)
        else:
            detected_letters = detect_letters_from_frame(frame_rgb)

        if detected_letters:
            add_detected_letters(detected_letters)
            last_detection_time = time.time()

        # After pause threshold, process buffer to form final word
        if time.time() - last_detection_time > PAUSE_THRESHOLD:
            cleaned = clean_letters()
            if cleaned:
                final_sentence = correct_with_nlp(cleaned)
                print("🧠 Final Output:", final_sentence)
                clear_buffer()

        # Display final sentence on original frame
        if final_sentence:
            cv2.putText(frame_resized, final_sentence, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (255, 0, 0), 3, cv2.LINE_AA)

        # Show the original frame with bounding boxes and detected alphabets
        cv2.imshow("Sign Language Detection", frame_rgb)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

from ultralytics import YOLO
model = YOLO("bestv6.pt")

def detect_letters_from_frame(frame):
    results = model(frame)[0]
    letters = [model.names[int(cls)] for cls in results.boxes.cls]
    return letters

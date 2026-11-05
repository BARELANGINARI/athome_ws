import logging
from pathlib import Path
import os
import time

import cv2
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, SiglipForImageClassification, ViTForImageClassification, ViTImageProcessor
import torch

logging.getLogger("ultralytics").setLevel(logging.ERROR)

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6

def classify_crop(crop, model, proc, thresh=0.5):
    if crop.size == 0:
        return "unknown"
    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    inputs = proc(images=pil, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)[0]
    idx = probs.argmax().item()
    return model.config.id2label.get(idx, str(idx)).lower() if probs[idx] > thresh else "unknown"

def classify_orientation(kp, conf, bbox):
    width = max(bbox[2] - bbox[0], 1)
    nose_ok = conf[NOSE] > 0.6
    eyes = sum(conf[i] > 0.5 for i in (L_EYE, R_EYE))
    both_shoulders = conf[L_SHOULDER] > 0.5 and conf[R_SHOULDER] > 0.5

    if both_shoulders:
        shoulder_width = abs(float(kp[L_SHOULDER][0]) - float(kp[R_SHOULDER][0]))
        narrow = shoulder_width < 0.25 * width
    else:
        narrow = True

    if narrow:
        if nose_ok and both_shoulders:
            mid_x = (kp[L_SHOULDER][0] + kp[R_SHOULDER][0]) / 2
            return "RIGHT SIDE" if kp[NOSE][0] > mid_x else "LEFT SIDE"
        if conf[R_EAR] > 0.5 and conf[L_EAR] <= 0.5:
            return "LEFT SIDE"
        if conf[L_EAR] > 0.5 and conf[R_EAR] <= 0.5:
            return "RIGHT SIDE"
        return "SIDE"

    if (nose_ok and eyes >= 1) or conf[NOSE] > 0.75:
        return "FRONT"
    return "BACK"


def main():
    model = YOLO(str(MODEL_DIR / "yolov8n-pose.pt"))
    gender_model = YOLO(str(MODEL_DIR / "gender_cls.pt"))
    color_model = SiglipForImageClassification.from_pretrained(str(MODEL_DIR / "color_model"))
    color_proc = AutoImageProcessor.from_pretrained(str(MODEL_DIR / "color_model"))
    hair_model = ViTForImageClassification.from_pretrained(str(MODEL_DIR / "hair_model"))
    hair_proc = ViTImageProcessor.from_pretrained(str(MODEL_DIR / "hair_model"))
    cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Webcam not found.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model(frame, imgsz=320, conf=0.6, classes=[0], verbose=False, device="cpu")[0]
        if result.keypoints is not None:
            boxes = result.boxes
            for box, kps, kcfs, bconf in zip(boxes.xyxy, result.keypoints.xy, result.keypoints.conf, boxes.conf):
                if bconf < 0.6:
                    continue
                kps, kcfs, box = kps.cpu().numpy(), kcfs.cpu().numpy(), box.cpu().numpy()
                if (kcfs > 0.5).sum() < 5:
                    continue
                orientation = classify_orientation(kps, kcfs, box)
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(x1, 0), max(y1, 0)

                h = y2 - y1
                crop = frame[y1:int(y1 + h * 0.55), x1:x2]
                gender = "unknown"
                if crop.size > 0:
                    g = gender_model(crop, device="cpu", verbose=False)[0]
                    if g.probs.top1conf > 0.7:
                        gender = g.names[g.probs.top1]

                torso = frame[int(y1 + h * 0.2):y2, x1:x2]
                head = frame[y1:int(y1 + h * 0.2), max(x1 + (x2 - x1) // 4, 0):x1 + 3 * (x2 - x1) // 4]
                clothing = classify_crop(torso, color_model, color_proc)
                hair = classify_crop(head, hair_model, hair_proc)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                for j, text in enumerate([f"{orientation}", f"{gender} | shirt:{clothing} | hair:{hair}"]):
                    ty = y1 - 8 - j * 20
                    if ty < 15:
                        ty = y1 + 15 + j * 20
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + 4), (0, 255, 0), -1)
                    cv2.putText(frame, text, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow("Person Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

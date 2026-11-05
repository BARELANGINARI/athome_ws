import json
import logging
import time
from pathlib import Path

import numpy as np

import cv2
import rclpy
from PIL import Image
from rclpy.node import Node
from std_msgs.msg import String
from transformers import AutoImageProcessor, SiglipForImageClassification, ViTForImageClassification, ViTImageProcessor
from ultralytics import YOLO
from insightface.app import FaceAnalysis

from msgs.msg import PersonDetected
from person_det.person_detection import classify_crop, classify_orientation

logging.getLogger("ultralytics").setLevel(logging.ERROR)

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


class PersonDetNode(Node):
    def __init__(self):
        super().__init__('person_det_node')

        self.declare_parameter('camera_index', 2)
        cam_idx = self.get_parameter('camera_index').value

        self.sound_sub = self.create_subscription(String, '/hri/sound_event', self.sound_callback, 10)
        self.name_sub = self.create_subscription(String, '/hri/guest_name', self.name_callback, 10)
        self.det_pub = self.create_publisher(PersonDetected, '/vision/detected_person', 10)

        self.detecting = False
        self.guest_name = ""
        self.guest_name_ts = 0.0

        # Face memory: name -> list of embeddings
        self.memory_path = MODEL_DIR / "face_memory.json"
        self.face_memory = {}
        if self.memory_path.exists():
            try:
                self.face_memory = json.loads(self.memory_path.read_text())
                self.get_logger().info(f"Loaded face memory: {list(self.face_memory.keys())}")
            except Exception as e:
                self.get_logger().error(f"Failed to load face memory: {e}")

        self.get_logger().info("Loading face recognition model (insightface)...")
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(320, 320))
        self.last_faces = []
        self.frame_count = 0
        self.stored_names = set()

        self.get_logger().info("Loading models...")
        self.model = YOLO(str(MODEL_DIR / "yolov8n-pose.pt"))
        self.gender_model = YOLO(str(MODEL_DIR / "gender_cls.pt"))
        self.color_model = SiglipForImageClassification.from_pretrained(str(MODEL_DIR / "color_model"))
        self.color_proc = AutoImageProcessor.from_pretrained(str(MODEL_DIR / "color_model"))
        self.hair_model = ViTForImageClassification.from_pretrained(str(MODEL_DIR / "hair_model"))
        self.hair_proc = ViTImageProcessor.from_pretrained(str(MODEL_DIR / "hair_model"))

        self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            self.get_logger().error(f"Camera {cam_idx} not found.")

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info("PersonDetNode ready. Waiting for doorbell...")

    def sound_callback(self, msg):
        if msg.data == "doorbell" and not self.detecting:
            self.detecting = True
            self.get_logger().info("Doorbell heard -> detection START")

    def name_callback(self, msg):
        self.guest_name = msg.data
        self.guest_name_ts = time.time()
        self.stored_names.discard(msg.data)
        self.get_logger().info(f"Guest name received: '{self.guest_name}'")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        if self.detecting:
            result = self.model(frame, imgsz=320, conf=0.6, classes=[0], verbose=False, device="cpu")[0]
            if result.keypoints is not None:
                boxes = result.boxes
                candidates = []
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
                        g = self.gender_model(crop, device="cpu", verbose=False)[0]
                        if g.probs.top1conf > 0.7:
                            gender = g.names[g.probs.top1]

                    torso = frame[int(y1 + h * 0.2):y2, x1:x2]
                    head = frame[y1:int(y1 + h * 0.2), max(x1 + (x2 - x1) // 4, 0):x1 + 3 * (x2 - x1) // 4]
                    clothing = classify_crop(torso, self.color_model, self.color_proc)
                    hair = classify_crop(head, self.hair_model, self.hair_proc)

                    area = (x2 - x1) * (y2 - y1)
                    candidates.append({
                        "area": area, "bbox": (x1, y1, x2, y2), "orientation": orientation,
                        "gender": gender, "clothing": clothing, "hair": hair,
                    })

                # Bbox terdekat = area terbesar
                nearest = max(candidates, key=lambda c: c["area"]) if candidates else None

                # Face recognition (tiap 3 frame)
                self.frame_count += 1
                if self.frame_count % 3 == 0:
                    try:
                        self.last_faces = self.face_app.get(frame)
                    except Exception:
                        self.last_faces = []

                for cand in candidates:
                    x1, y1, x2, y2 = cand["bbox"]
                    cand["emb"] = None
                    for f in self.last_faces:
                        fx1, fy1, fx2, fy2 = f.bbox.astype(int)
                        cx, cy = (fx1 + fx2) // 2, (fy1 + fy2) // 2
                        if x1 <= cx <= x2 and y1 <= cy <= y2:
                            cand["emb"] = f.embedding
                            break

                # Simpan embedding ke memori saat nama tamu baru diterima
                if nearest is not None and self.guest_name and nearest["emb"] is not None and self.guest_name not in self.stored_names:
                    emb_list = nearest["emb"].astype(float).tolist()
                    self.face_memory.setdefault(self.guest_name, []).append(emb_list)
                    self.face_memory[self.guest_name] = self.face_memory[self.guest_name][-5:]
                    try:
                        self.memory_path.write_text(json.dumps(self.face_memory))
                    except Exception as e:
                        self.get_logger().error(f"Failed to save face memory: {e}")
                    self.stored_names.add(self.guest_name)
                    self.get_logger().info(f"Saved face embedding for '{self.guest_name}'")

                def match_name(emb):
                    if emb is None:
                        return None
                    best_name, best_sim = None, 0.5
                    for name, embs in self.face_memory.items():
                        for e in embs:
                            e = np.array(e, dtype=np.float32)
                            sim = float(np.dot(emb, e) / (np.linalg.norm(emb) * np.linalg.norm(e) + 1e-9))
                            if sim > best_sim:
                                best_name, best_sim = name, sim
                    return best_name

                if nearest is not None:
                    nearest["matched_name"] = match_name(nearest["emb"])

                for cand in candidates:
                    x1, y1, x2, y2 = cand["bbox"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Info kecil (arah + gender + baju + rambut) di bawah bbox
                    info_text = f"{cand['orientation']} | {cand['gender']} | shirt:{cand['clothing']} | hair:{cand['hair']}"
                    (tw, th), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    ty = y2 + th + 10
                    if ty + 6 > frame.shape[0]:
                        ty = y1 + th + 16
                    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 8, ty + 4), (0, 255, 0), -1)
                    cv2.putText(frame, info_text, (x1 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

                if nearest is not None:
                    x1, y1, x2, y2 = nearest["bbox"]

                    # Publish hanya orang terdekat
                    det = PersonDetected()
                    det.name = nearest.get("matched_name") or ""
                    det.gender = nearest["gender"]
                    det.shirt_color = nearest["clothing"]
                    self.det_pub.publish(det)

                    # Nama (besar) hanya di bbox orang terdekat
                    display_name = nearest.get("matched_name")
                    if display_name:
                        name_text = display_name
                        (tw, th), _ = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                        ty = y1 - 10
                        if ty - th - 8 < 0:
                            ty = y1 + th + 8
                        cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 8, ty + 6), (0, 180, 0), -1)
                        cv2.putText(frame, name_text, (x1 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Person Detection", frame)
        cv2.waitKey(1)

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PersonDetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

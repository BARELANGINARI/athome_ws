#!/usr/bin/env python3
"""
Script Pengambil / Pemotong Foto Wajah Host (Revaldo) untuk Receptionist Task.
Penggunaan:
  1. Live Camera:   python3 capture_host.py
  2. Dari File:     python3 capture_host.py /path/ke/foto.jpg
"""

import sys
import os

# Prioritaskan system OpenCV 4.6.0 agar FaceDetectorYN berjalan optimal
if '/usr/lib/python3/dist-packages' in sys.path:
    sys.path.remove('/usr/lib/python3/dist-packages')
    sys.path.insert(0, '/usr/lib/python3/dist-packages')

import cv2
import numpy as np

def get_output_paths():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    ws_dir = os.path.abspath(os.path.join(script_dir, "../../../.."))
    
    paths = [
        os.path.join(ws_dir, "src/athome_vision/models/host.jpg"),
        os.path.join(ws_dir, "models/host.jpg"),
    ]
    share_dir = os.path.join(ws_dir, "install/athome_vision/share/athome_vision/models")
    if os.path.exists(share_dir):
        paths.append(os.path.join(share_dir, "host.jpg"))
    return paths

def get_yunet_model_path():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    ws_dir = os.path.abspath(os.path.join(script_dir, "../../../.."))
    candidates = [
        os.path.join(ws_dir, "src/athome_vision/models/face_detection_yunet.onnx"),
        os.path.join(ws_dir, "models/face_detection_yunet.onnx"),
        "/home/dell/Documents/athome_ws/src/athome_vision/models/face_detection_yunet.onnx"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def crop_face(img, detector=None):
    h, w = img.shape[:2]
    if detector is not None:
        try:
            detector.setInputSize((w, h))
            _, faces = detector.detect(img)
            if faces is not None and len(faces) > 0:
                best_face = max(faces, key=lambda f: f[2] * f[3])
                fx, fy, fw, fh = int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3])
                
                # Beri padding 25% agar seluruh kepala/rambut masuk
                pad_x = int(fw * 0.25)
                pad_y = int(fh * 0.35)
                x1 = max(0, fx - pad_x)
                y1 = max(0, fy - pad_y)
                x2 = min(w, fx + fw + pad_x)
                y2 = min(h, fy + fh + pad_y)
                
                face_img = img[y1:y2, x1:x2]
                if face_img.size > 0:
                    return face_img, (fx, fy, fw, fh)
        except Exception:
            pass

    # Fallback center crop persegi
    side = min(w, h)
    cx, cy = w // 2, h // 2
    return img[cy - side // 2 : cy + side // 2, cx - side // 2 : cx + side // 2], None

def save_host_image(crop_img):
    out_paths = get_output_paths()
    saved = []
    for p in out_paths:
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            cv2.imwrite(p, crop_img)
            saved.append(p)
        except Exception as e:
            print(f"[WARN] Gagal menyimpan ke {p}: {e}")
    return saved

def process_image_file(file_path):
    if not os.path.exists(file_path):
        print(f"[ERROR] File tidak ditemukan: {file_path}")
        return False
    
    img = cv2.imread(file_path)
    if img is None:
        print(f"[ERROR] Gagal membaca gambar: {file_path}")
        return False

    yunet_model = get_yunet_model_path()
    detector = None
    if yunet_model:
        try:
            detector = cv2.FaceDetectorYN.create(yunet_model, "", (img.shape[1], img.shape[0]), 0.5)
        except Exception:
            pass

    cropped, bbox = crop_face(img, detector)
    if cropped is None or cropped.size == 0:
        print("[ERROR] Gagal mendeteksi wajah pada foto.")
        return False

    # Resize ke 112x112 standar model SFace / ArcFace
    resized = cv2.resize(cropped, (112, 112), interpolation=cv2.INTER_AREA)
    saved = save_host_image(resized)
    print("\n=======================================================")
    print(" [SUKSES] Foto Host Berhasil Dipotong & Disimpan!")
    for s in saved:
        print(f"  -> {s}")
    print("=======================================================")
    return True

def run_live_capture(device_idx=0):
    cap = cv2.VideoCapture(device_idx)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print(f"[ERROR] Tidak dapat membuka kamera video /dev/video{device_idx} atau /dev/video1.")
            print("Gunakan opsi file gambar yang ada: ./capture_host.sh /path/ke/foto.jpg")
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    yunet_model = get_yunet_model_path()
    detector = None
    if yunet_model:
        try:
            detector = cv2.FaceDetectorYN.create(yunet_model, "", (640, 480), 0.5)
        except Exception:
            pass

    window_name = "Host Face Capture (Revaldo)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("\n=======================================================")
    print(" [KAMERA LIVE] Pengambilan Foto Wajah Host (Revaldo)")
    print("  1. Posisikan wajah Host tepat di tengah frame")
    print("  2. Tekan [SPASI] atau [ENTER] untuk mengambil foto")
    print("  3. Tekan [Q] atau [ESC] untuk keluar")
    print("=======================================================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        display = frame.copy()

        # Deteksi wajah secara visual
        bbox = None
        if detector is not None:
            try:
                detector.setInputSize((w, h))
                _, faces = detector.detect(frame)
                if faces is not None and len(faces) > 0:
                    best_face = max(faces, key=lambda f: f[2] * f[3])
                    fx, fy, fw, fh = int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3])
                    bbox = (fx, fy, fw, fh)
                    cv2.rectangle(display, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
                    cv2.putText(display, "Host Face OK", (fx, fy - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except Exception:
                pass

        # Petunjuk di layar
        cv2.putText(display, "Tekan [SPASI] untuk Ambil Foto Host | [Q] Batal",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == 27:
            print("Pengambilan foto dibatalkan.")
            break
        elif key == 32 or key == 13: # SPASI atau ENTER
            cropped, _ = crop_face(frame, detector)
            if cropped is not None and cropped.size > 0:
                resized = cv2.resize(cropped, (112, 112), interpolation=cv2.INTER_AREA)
                saved = save_host_image(resized)
                
                # Feedback visual
                flash = np.full_like(display, (0, 255, 0))
                cv2.addWeighted(flash, 0.5, display, 0.5, 0, display)
                cv2.putText(display, "FOTO BERHASIL DISIMPAN!", (w // 2 - 180, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow(window_name, display)
                cv2.waitKey(1000)

                print("\n=======================================================")
                print(" [SUKSES] Foto Host Berhasil Disimpan!")
                for s in saved:
                    print(f"  -> {s}")
                print("=======================================================")
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        process_image_file(sys.argv[1])
    else:
        cam_idx = 0
        if "--cam" in sys.argv:
            try:
                idx = sys.argv.index("--cam")
                cam_idx = int(sys.argv[idx + 1])
            except Exception:
                pass
        run_live_capture(cam_idx)

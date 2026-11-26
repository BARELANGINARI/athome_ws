#!/usr/bin/env python3
"""
Tool Registrasi Foto Wajah Host (Receptionist Task)
Fitur 1: Live Capture Host dari Kamera (Webcam / Laptop / ZED)
Fitur 2: Baca & Deteksi Wajah Host dari File Foto (Import Image File)
"""

import sys
import os
import subprocess

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
        os.path.join(ws_dir, "src/athome_vision/models/receptionist/host.jpg"),
        os.path.join(ws_dir, "src/athome_vision/models/host.jpg"),
        os.path.join(ws_dir, "models/host.jpg"),
    ]
    share_dir = os.path.join(ws_dir, "install/athome_vision/share/athome_vision/models/receptionist")
    if os.path.exists(share_dir):
        paths.append(os.path.join(share_dir, "host.jpg"))
    return paths


def get_yunet_model_path():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    ws_dir = os.path.abspath(os.path.join(script_dir, "../../../.."))
    candidates = [
        os.path.join(ws_dir, "src/athome_vision/models/receptionist/face_detection_yunet.onnx"),
        os.path.join(ws_dir, "src/athome_vision/models/face_detection_yunet.onnx"),
        os.path.join(ws_dir, "models/receptionist/face_detection_yunet.onnx"),
        os.path.join(ws_dir, "models/face_detection_yunet.onnx"),
        "/home/dell/Documents/athome_ws/src/athome_vision/models/receptionist/face_detection_yunet.onnx",
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

                # Beri margin padding 25% horizontal dan 35% vertikal agar seluruh kepala & rambut masuk
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


def notify_ros_vision(file_path):
    try:
        subprocess.run(
            [
                "ros2", "topic", "pub", "--once",
                "/vision/load_host_photo", "std_msgs/msg/String",
                f"{{data: '{file_path}'}}"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2.0
        )
    except Exception:
        pass


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
    """Fitur 2: Membaca foto Host dari file dan mendeteksi wajahnya"""
    file_path = os.path.abspath(os.path.expanduser(file_path))
    if not os.path.exists(file_path):
        print(f"[ERROR] File tidak ditemukan: {file_path}")
        return False

    img = cv2.imread(file_path)
    if img is None:
        print(f"[ERROR] Gagal membaca file gambar: {file_path}")
        return False

    h, w = img.shape[:2]
    print(f"\n[INFO] Membaca foto Host: {file_path} ({w}x{h} px)")

    yunet_model = get_yunet_model_path()
    detector = None
    if yunet_model:
        try:
            detector = cv2.FaceDetectorYN.create(yunet_model, "", (w, h), 0.45)
        except Exception:
            pass

    cropped, bbox = crop_face(img, detector)
    if cropped is None or cropped.size == 0:
        print("[ERROR] Wajah tidak terdeteksi pada foto.")
        return False

    if bbox is not None:
        fx, fy, fw, fh = bbox
        print(f"[INFO] Wajah Host berhasil terdeteksi (area: x={fx}, y={fy}, w={fw}, h={fh})")
    else:
        print("[INFO] Wajah dipotong dengan metode center-crop.")

    # Pertahankan resolusi crop (maks 640px) agar YuNet + warpAffine di vision lebih akurat
    ch, cw = cropped.shape[:2]
    if max(ch, cw) > 640:
        scale = 640.0 / max(ch, cw)
        resized = cv2.resize(cropped, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_AREA)
    else:
        resized = cropped
    saved = save_host_image(resized)

    print("\n=======================================================")
    print(" [SUKSES] FITUR 2: BACA FOTO HOST BERHASIL!")
    print(" Foto wajah Host telah dipotong & disimpan ke:")
    for s in saved:
        print(f"  -> {s}")
    print("=======================================================")

    if saved:
        notify_ros_vision(saved[0])
    return True


def open_camera(device_idx=0):
    """Membuka kamera menggunakan backend V4L2, set resolusi HD720, dan verifikasi frame."""
    indices = []
    for idx in [device_idx, 0, 1, 2]:
        if idx not in indices:
            indices.append(idx)

    for idx in indices:
        for backend in (cv2.CAP_V4L2, cv2.CAP_ANY):
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                h, w = frame.shape[:2]
                # Set resolusi HD720 (2560x720 untuk stereo ZED, 1280x720 untuk kamera biasa)
                if w >= 2 * h and w >= 1280:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                else:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                ret2, frame2 = cap.read()
                if ret2 and frame2 is not None and frame2.size > 0:
                    frame = frame2
                    h, w = frame.shape[:2]
                disp_w = w // 2 if (w >= 2 * h and w >= 1280) else w
                return cap, idx, (disp_w, h)
            cap.release()
    return None, None, None


def run_live_capture(device_idx=0):
    """Fitur 1: Mengambil foto Host secara live dari kamera"""
    cap, opened_idx, res = open_camera(device_idx)
    if cap is None:
        print(f"\n[ERROR] Tidak dapat membuka atau membaca frame dari kamera (/dev/video{device_idx}).")
        print("Kemungkinan penyebab:")
        print("  1. Kamera sedang dipakai oleh node ROS 'vision' (camera.launch.py).")
        print("     -> Hentikan camera.launch.py sementara, ATAU gunakan ./register_host.sh")
        print("  2. Tidak ada kamera yang terhubung.")
        print("Alternatif Fitur 2 (Baca Foto File):")
        print("  ./capture_host.sh --photo /path/ke/foto.jpg")
        return False

    print(f"[INFO] Kamera /dev/video{opened_idx} berhasil dibuka (resolusi: {res[0]}x{res[1]}).")

    yunet_model = get_yunet_model_path()
    detector = None
    if yunet_model:
        try:
            detector = cv2.FaceDetectorYN.create(yunet_model, "", (640, 480), 0.45)
        except Exception:
            pass

    window_name = "Host Capture (Live Camera)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("\n=======================================================")
    print(" [FITUR 1] PENGAMBILAN FOTO LIVE HOST DARI KAMERA")
    print("  1. Arahkan wajah Host ke kamera")
    print("  2. Kotak HIJAU menandakan wajah Host terdeteksi")
    print("  3. Tekan [SPASI] atau [ENTER] untuk mengambil foto")
    print("  4. Tekan [Q] atau [ESC] untuk keluar/batal")
    print("=======================================================\n")

    success = False
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] Gagal membaca frame dari kamera.")
            break

        # Jika kamera stereo side-by-side (mis. ZED 2i), ambil lensa kiri saja
        h, w = frame.shape[:2]
        if w >= 2 * h and w >= 1280:
            frame = frame[:, : w // 2]
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

        cv2.putText(display, "Tekan [SPASI] Ambil Foto | [Q] Batal",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == 27:
            print("Pengambilan foto dibatalkan.")
            break
        elif key == 32 or key == 13: # SPASI atau ENTER
            cropped, _ = crop_face(frame, detector)
            if cropped is not None and cropped.size > 0:
                ch, cw = cropped.shape[:2]
                if max(ch, cw) > 640:
                    scale = 640.0 / max(ch, cw)
                    resized = cv2.resize(cropped, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_AREA)
                else:
                    resized = cropped
                saved = save_host_image(resized)

                # Efek flash hijau tanda sukses
                flash = np.full_like(display, (0, 255, 0))
                cv2.addWeighted(flash, 0.5, display, 0.5, 0, display)
                cv2.putText(display, "FOTO BERHASIL DIAMBIL!", (w // 2 - 180, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow(window_name, display)
                cv2.waitKey(1000)

                print("\n=======================================================")
                print(" [SUKSES] FITUR 1: CAPTURE FOTO HOST BERHASIL!")
                for s in saved:
                    print(f"  -> Tersimpan ke: {s}")
                print("=======================================================")
                if saved:
                    notify_ros_vision(saved[0])
                success = True
                break

    cap.release()
    cv2.destroyAllWindows()
    return success


def interactive_menu():
    """Tampilan menu interaktif jika script dijalankan tanpa argumen"""
    while True:
        print("\n=======================================================")
        print("          REGISTRASI FOTO HOST RECEPTIONIST            ")
        print("=======================================================")
        print("  Pilih fitur yang ingin digunakan:")
        print("  [1] Capture Host dari Kamera (Live Camera)")
        print("  [2] Baca Foto Host dari File Gambar (Read Photo File)")
        print("  [3] Keluar")
        print("=======================================================")
        try:
            choice = input("Pilihan Anda [1/2/3]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nKeluar.")
            break

        if choice == "1":
            run_live_capture(0)
            break
        elif choice == "2":
            try:
                photo_path = input("Masukkan path file foto Host: ").strip().strip("'\"")
            except (EOFError, KeyboardInterrupt):
                break
            if photo_path:
                if process_image_file(photo_path):
                    break
            else:
                print("[ERROR] Path file foto tidak boleh kosong.")
        elif choice == "3" or choice.lower() == "q":
            print("Selesai.")
            break
        else:
            print("[WARN] Pilihan tidak valid. Silakan pilih 1, 2, atau 3.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ["--camera", "-c", "camera", "live"]:
            cam_idx = 0
            if len(sys.argv) > 2 and sys.argv[2].isdigit():
                cam_idx = int(sys.argv[2])
            run_live_capture(cam_idx)
        elif arg in ["--photo", "-p", "photo", "file"]:
            if len(sys.argv) > 2:
                process_image_file(sys.argv[2])
            else:
                try:
                    p = input("Masukkan path file foto Host: ").strip().strip("'\"")
                    if p:
                        process_image_file(p)
                except (EOFError, KeyboardInterrupt):
                    pass
        elif not arg.startswith("-") and os.path.exists(sys.argv[1]):
            # Penggunaan langsung: python3 capture_host.py /path/ke/foto.jpg
            process_image_file(sys.argv[1])
        else:
            interactive_menu()
    else:
        if sys.stdin.isatty():
            interactive_menu()
        else:
            run_live_capture(0)

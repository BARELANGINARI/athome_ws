import csv
import queue
import threading
import time
from collections import deque
import numpy as np
import sounddevice as sd
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from faster_whisper import WhisperModel

# --- 1. KONFIGURASI SISTEM & MODEL ---
SAMPLE_RATE = 16000
BLOCK_DURATION = 0.03  # 30 ms per frame
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)

# Parameter State Machine VAD
PRE_ROLL_FRAMES = 10      # Simpan 300 ms sebelum suara terdeteksi (cegah huruf awal terpotong)
HANGOVER_FRAMES = 25      # Tunggu 750 ms hening sebelum memutuskan kalimat selesai
MIN_SPEECH_FRAMES = 15    # Minimal durasi bicara 450 ms agar diproses ASR (abaikan batuk/klik)

# Konfigurasi Whisper (Ganti device="cpu", compute_type="int8" jika ingin tes CPU)
WHISPER_MODEL_SIZE = "base"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE = "float16"

# --- 2. GLOBAL QUEUES & STATE ---
audio_queue = queue.Queue()
asr_queue = queue.Queue()

threshold_dbfs = -40.0 #-35.0
threshold_flatness = 0.40  # Speech biasanya memiliki flatness < 0.45
latest_transcript = "[Menunggu suara...]"
latest_asr_metrics = "RTF: 0.00 | Durasi: 0.0s"

# State VAD
is_speaking = False
speech_buffer = []
pre_roll_buffer = deque(maxlen=PRE_ROLL_FRAMES)
silence_counter = 0

# Filter Bandpass Vokal (300 - 3400 Hz)
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, [lowcut / nyq, highcut / nyq], btype='band')

b_band, a_band = butter_bandpass(300, 3400, SAMPLE_RATE)

def compute_features(indata):
    eps = 1e-10
    signal = indata[:, 0]
    
    # Full-band & Speech-band dBFS
    rms = np.sqrt(np.mean(signal**2) + eps)
    dbfs = 20 * np.log10(rms + eps)
    
    speech_sig = lfilter(b_band, a_band, signal)
    rms_speech = np.sqrt(np.mean(speech_sig**2) + eps)
    dbfs_speech = 20 * np.log10(rms_speech + eps)
    
    # Spectral Flatness
    fft_mag = np.abs(np.fft.rfft(signal * np.hanning(len(signal)))) + eps
    flatness = np.exp(np.mean(np.log(fft_mag))) / np.mean(fft_mag)
    
    return signal, fft_mag, dbfs, dbfs_speech, flatness

# filter kalimat halusinasi
HALLUCINATION_BLACKLIST = ["Terima kasih", "terimakasih", "Thank You", "Thanks", "thank you", "Thank you"]

# --- 3. BACKGROUND WORKER THREAD UNTUK ASR ---
def asr_worker():
    global latest_transcript, latest_asr_metrics
    print(f"[ASR INIT] Memuat Model Whisper ({WHISPER_MODEL_SIZE}) pada {WHISPER_DEVICE}...")
    try:
        model = WhisperModel("deepdml/faster-whisper-large-v3-turbo-ct2", device="cuda", compute_type="int8_float16")
    except Exception as e:
        print(f"[WARNING] Gagal memuat di {WHISPER_DEVICE} ({e}). Beralih ke CPU int8...")
        model = WhisperModel("deepdml/faster-whisper-large-v3-turbo-ct2", device="cpu", compute_type="int8")
    print("[ASR READY] Siap menerima segmen suara dari VAD.")

    while True:
        audio_segment = asr_queue.get()
        if audio_segment is None:
            break
            
        dur = len(audio_segment) / SAMPLE_RATE
        t0 = time.perf_counter()
        
        # Transkripsi langsung dari array 
        segments, _ = model.transcribe(
            audio_segment,
            beam_size=2,
            language="en",
            condition_on_previous_text=True,
            no_speech_threshold=0.5,
            log_prob_threshold=-0.8,
            compression_ratio_threshold=1.8,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400, speech_pad_ms=200)
        )

        valid_text = [
            seg.text.strip() for seg in segments
            if seg.no_speech_prob < 0.5
        ]
        
        text = " ".join(valid_text).strip()
        
        proc_time = time.perf_counter() - t0
        rtf = proc_time / dur if dur > 0 else 0

        # post filter nlacklist dan panjang kalimat
        if text and text.lower().rstrip(".!") not in HALLUCINATION_BLACKLIST and len(text)>2:

            latest_transcript = f'"{text}"'
            latest_asr_metrics = f"RTF: {rtf:.3f} | Latensi: {proc_time:.2f}s | Audio: {dur:.2f}s"
            print(f"[ASR] {latest_transcript} ({latest_asr_metrics})")
        else:
            print(f"[ASR] Ditolak - noise halusinasi Raw: {text} ({proc_time:.2f}s)")

        asr_queue.task_done()

asr_thread = threading.Thread(target=asr_worker, daemon=True)
asr_thread.start()

# --- 4. CALLBACK AUDIO ---
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

# --- 5. SETUP GUI MATPLOTLIB ---
fig, (ax_wave, ax_spec, ax_dbfs) = plt.subplots(3, 1, figsize=(10, 8.5))
plt.subplots_adjust(hspace=0.5, bottom=0.15)

# Plot 1: Waveform & Indikator Status VAD
x_wave = np.arange(BLOCK_SIZE) / SAMPLE_RATE * 1000
line_wave, = ax_wave.plot(x_wave, np.zeros(BLOCK_SIZE), color='#007acc')
ax_wave.set_ylim(-1.0, 1.0)
title_wave = ax_wave.set_title("VAD Status: SILENCE | menunggu input...", fontweight="bold", color="gray")
ax_wave.set_ylabel("Amplitude")

# Plot 2: FFT Spectrum
freqs = np.fft.rfftfreq(BLOCK_SIZE, 1.0 / SAMPLE_RATE)
line_spec, = ax_spec.plot(freqs, np.zeros(len(freqs)), color='#d9534f')
ax_spec.axvspan(300, 3400, color='green', alpha=0.15, label='Speech Band (300-3400 Hz)')
ax_spec.set_xlim(0, 8000)
ax_spec.set_ylim(-60, 20)
ax_spec.set_title("Frequency Spectrum (FFT dB)")
ax_spec.set_ylabel("dB")
ax_spec.legend(loc="upper right")

# Plot 3: Rolling Energy & Threshold
HISTORY_LEN = 100
dbfs_hist = np.full(HISTORY_LEN, -80.0)
speech_dbfs_hist = np.full(HISTORY_LEN, -80.0)
vad_state_hist = np.full(HISTORY_LEN, -80.0)

line_dbfs, = ax_dbfs.plot(dbfs_hist, label="Full dBFS", color='gray', alpha=0.5)
line_speech_dbfs, = ax_dbfs.plot(speech_dbfs_hist, label="Speech-Band dBFS", color='#2ca02c')
line_thresh = ax_dbfs.axhline(threshold_dbfs, color='red', linestyle='--', label=f"Thresh ({threshold_dbfs:.1f} dBFS)")
ax_dbfs.set_ylim(-80, 0)
ax_dbfs.set_title(f"VAD Energy | [UP/DOWN] dBFS Thresh | [LEFT/RIGHT] Flatness Max: {threshold_flatness:.2f}")
ax_dbfs.set_ylabel("dBFS")
ax_dbfs.legend(loc="upper left")

# Kotak Teks Hasil Transkripsi ASR di Bagian Bawah
asr_text_box = fig.text(0.5, 0.03, f"ASR: {latest_transcript}\n{latest_asr_metrics}", 
                        ha="center", va="center", fontsize=11, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4f8", edgecolor="#007acc"))

def on_key(event):
    global threshold_dbfs, threshold_flatness
    if event.key == 'up':
        threshold_dbfs = min(0.0, threshold_dbfs + 1.0)
    elif event.key == 'down':
        threshold_dbfs = max(-80.0, threshold_dbfs - 1.0)
    elif event.key == 'right':
        threshold_flatness = min(1.0, threshold_flatness + 0.05)
    elif event.key == 'left':
        threshold_flatness = max(0.05, threshold_flatness - 0.05)
        
    line_thresh.set_ydata([threshold_dbfs, threshold_dbfs])
    line_thresh.set_label(f"Thresh ({threshold_dbfs:.1f} dBFS)")
    ax_dbfs.set_title(f"VAD Energy | [UP/DOWN] dBFS Thresh | [LEFT/RIGHT] Flatness Max: {threshold_flatness:.2f}")
    ax_dbfs.legend(loc="upper left")

fig.canvas.mpl_connect('key_press_event', on_key)

# --- 6. UPDATE LOOP & STATE MACHINE VAD ---
def update_plot(frame):
    global dbfs_hist, speech_dbfs_hist, is_speaking, speech_buffer, silence_counter
    
    while not audio_queue.empty():
        indata = audio_queue.get_nowait()
        sig, fft_mag, dbfs, dbfs_speech, flatness = compute_features(indata)
        
        # Logika Keputusan VAD (Kombinasi Energy Speech-Band & Spectral Flatness)
        frame_is_voice = (dbfs_speech > threshold_dbfs) and (flatness < threshold_flatness)
        
        if frame_is_voice:
            if not is_speaking:
                is_speaking = True
                # Masukkan pre-roll buffer agar awal kata tidak terpotong
                speech_buffer.extend(list(pre_roll_buffer))
            speech_buffer.append(sig)
            silence_counter = 0
        else:
            if is_speaking:
                speech_buffer.append(sig)
                silence_counter += 1
                # Jika sudah hening melebihi hangover time, kirim ke ASR
                if silence_counter >= HANGOVER_FRAMES:
                    if len(speech_buffer) >= MIN_SPEECH_FRAMES + PRE_ROLL_FRAMES:
                        full_audio = np.concatenate(speech_buffer)
                        asr_queue.put(full_audio)
                    # Reset state
                    is_speaking = False
                    speech_buffer = []
                    silence_counter = 0
            else:
                pre_roll_buffer.append(sig)
                
        # Update Plot Data
        line_wave.set_ydata(sig)
        line_spec.set_ydata(20 * np.log10(fft_mag))
        
        dbfs_hist = np.roll(dbfs_hist, -1)
        dbfs_hist[-1] = dbfs
        speech_dbfs_hist = np.roll(speech_dbfs_hist, -1)
        speech_dbfs_hist[-1] = dbfs_speech
        
        line_dbfs.set_ydata(dbfs_hist)
        line_speech_dbfs.set_ydata(speech_dbfs_hist)
        
        # Update Indikator Visual VAD
        if is_speaking:
            title_wave.set_text(f"VAD Status: [SPEECH DETECTED - RECORDING] (Flatness: {flatness:.2f})")
            title_wave.set_color("green")
            line_wave.set_color("#2ca02c")
        else:
            title_wave.set_text(f"VAD Status: SILENCE (Flatness: {flatness:.2f})")
            title_wave.set_color("gray")
            line_wave.set_color("#007acc")
            
    asr_text_box.set_text(f"ASR: {latest_transcript}\n{latest_asr_metrics}")
    return line_wave, line_spec, line_dbfs, line_speech_dbfs, line_thresh, title_wave, asr_text_box

# --- EKSEKUSI UTAMA ---
with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                    blocksize=BLOCK_SIZE, callback=audio_callback):
    ani = FuncAnimation(fig, update_plot, interval=30, blit=False, cache_frame_data=False)
    plt.show()
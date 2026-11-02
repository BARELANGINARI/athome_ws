import os
import glob
import time
import pandas as pd
from faster_whisper import WhisperModel
import jiwer

LOCAL_MODELS = {
    "base": os.path.expanduser("~/.cache/huggingface/hub/models--Systran--faster-whisper-base/snapshots/d01c3014881c9c6f3133c182f3d2887eb6ca1c789a7538c5c007196857a0a6a9"),
    "small": os.path.expanduser("~/.cache/huggingface/hub/models--Systran--faster-whisper-small/snapshots/f0525ff8f037671755ee14f9d45388656111f26a"),
}
# Original reference text played from external speaker (Ground Truth)
# Example:
GROUND_TRUTH_TEXT = "robotika politeknik negeri batam mengembangkan sistem navigasi otonom berbasis kecerdasan buatan"

MODELS_TO_TEST = ["tiny", "base", "small"]
CONFIGS = [
    {"device": "cpu", "compute_type": "int8"},
    {"device": "cuda", "compute_type": "float16"}
]

def clean_text(text):
    return jiwer.RemovePunctuation()(text.lower().strip())

def run_benchmarks(wav_folder="./"):
    wav_files = glob.glob(os.path.join(wav_folder, "*.wav"))
    results = []
    
    ref_clean = clean_text(GROUND_TRUTH_TEXT)

    for config in CONFIGS:
        dev = config["device"]
        comp = config["compute_type"]
        
        for model_size in MODELS_TO_TEST:
            print(f"\n[LOADING] Whisper {model_size} | Device: {dev} ({comp})...")
            try:
                # model = WhisperModel(model_size, device=dev, compute_type=comp)
                model = WhisperModel(model_size, device=dev, compute_type=comp)
            except Exception as e:
                print(f"[SKIP] Failed to load {dev}: {e}")
                continue

            for wav_path in wav_files:
                filename = os.path.basename(wav_path)
                # Estimate metadata from filename
                parts = filename.replace(".wav", "").split("_")
                mic = parts[1] if len(parts) > 1 else "Unknown"
                condition = parts[2] if len(parts) > 2 else "Unknown"
                distance = parts[3] if len(parts) > 3 else "Unknown"
                
                # Measure latency & RTF
                start_t = time.perf_counter()
                segments, info = model.transcribe(wav_path, beam_size=1, language="id")
                pred_text = " ".join([seg.text for seg in segments])
                latency = time.perf_counter() - start_t
                
                audio_dur = info.duration
                rtf = latency / audio_dur if audio_dur > 0 else 0
                
                pred_clean = clean_text(pred_text)
                wer = jiwer.wer(ref_clean, pred_clean)
                
                results.append({
                    "mic": mic,
                    "condition": condition,
                    "distance": distance,
                    "model": model_size,
                    "device": dev,
                    "compute": comp,
                    "wer": round(wer, 3),
                    "rtf": round(rtf, 3),
                    "latency_sec": round(latency, 2),
                    "transcription": pred_text
                })

    df = pd.DataFrame(results)
    df.to_csv("benchmark_stt_results.csv", index=False)
    print("\n--- BENCHMARK RESULTS COMPLETED ---")
    print(df[["mic", "model", "device", "wer", "rtf", "distance"]].to_markdown())

if __name__ == "__main__":
    run_benchmarks()
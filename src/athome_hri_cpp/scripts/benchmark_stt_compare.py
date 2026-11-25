#!/usr/bin/env python3
"""Benchmark and compare Python (athome_hri) vs C++ (athome_hri_cpp) STT nodes.

Usage:
  1) Record metrics while running Python or C++ STT:
     ./benchmark_stt_compare.py record --label python --out /tmp/bench_py.json --duration 60
     ./benchmark_stt_compare.py record --label cpp    --out /tmp/bench_cpp.json --duration 60
  2) Compare two recordings:
     ./benchmark_stt_compare.py compare /tmp/bench_py.json /tmp/bench_cpp.json
"""

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional


def find_stt_pid() -> Optional[int]:
    """Return PID of running stt_engine_node process, or None."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "stt_engine_node"], text=True, stderr=subprocess.DEVNULL
        )
        pids = [int(line.strip()) for line in out.splitlines() if line.strip().isdigit()]
        my_pid = os.getpid()
        pids = [p for p in pids if p != my_pid]
        return pids[0] if pids else None
    except Exception:
        return None


def sample_cpu_percent(pid: int) -> Optional[float]:
    try:
        out = subprocess.check_output(
            ["ps", "-o", "%cpu=", "-p", str(pid)], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return float(out) if out else None
    except Exception:
        return None


def sample_ram_mb(pid: int) -> Optional[float]:
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    kb = float(line.split()[1])
                    return kb / 1024.0
    except Exception:
        return None
    return None


def sample_vram_mb(pid: int) -> float:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        total = 0.0
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
                try:
                    total += float(parts[1])
                except ValueError:
                    pass
        return total
    except Exception:
        return 0.0


def has_consecutive_repetition(text: str, max_ngram: int = 4) -> bool:
    tokens = [re.sub(r"[^\w]+", "", t.lower()) for t in text.split()]
    tokens = [t for t in tokens if t]
    if len(tokens) < 2:
        return False
    for n in range(1, min(max_ngram, len(tokens) // 2) + 1):
        for i in range(len(tokens) - 2 * n + 1):
            if tokens[i : i + n] == tokens[i + n : i + 2 * n]:
                return True
    return False


class TranscriptListener:
    """Capture /hri/stt_transcript via rclpy if available, else ros2 topic echo."""

    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen] = None

    def _record_text(self, text: str) -> None:
        clean = text.strip()
        if not clean:
            return
        with self._lock:
            self.events.append(
                {
                    "t_rel_sec": round(time.monotonic() - self._t0, 2),
                    "text": clean,
                    "words": len(clean.split()),
                    "has_repetition": has_consecutive_repetition(clean),
                }
            )
        print(f"  [STT] {clean}", flush=True)

    def start(self) -> None:
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            import rclpy
            from rclpy.node import Node
            from std_msgs.msg import String

            rclpy.init(args=None)
            node = rclpy.create_node("stt_benchmark_listener")
            node.create_subscription(String, self.topic, lambda m: self._record_text(m.data), 10)
            while not self._stop.is_set() and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.2)
            node.destroy_node()
            rclpy.shutdown()
            return
        except Exception:
            pass

        # Fallback: ros2 topic echo subprocess
        try:
            self._proc = subprocess.Popen(
                ["ros2", "topic", "echo", self.topic, "--field", "data"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            if self._proc.stdout is not None:
                for line in self._proc.stdout:
                    if self._stop.is_set():
                        break
                    s = line.strip()
                    if s and s != "---":
                        self._record_text(s.strip("'\""))
        except Exception as exc:
            print(f"[WARN] Could not listen to {self.topic}: {exc}", file=sys.stderr)

    def stop(self) -> List[Dict[str, Any]]:
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._lock:
            return list(self.events)


def cmd_record(args: argparse.Namespace) -> int:
    stop_flag = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_flag.set())
    signal.signal(signal.SIGTERM, lambda *_: stop_flag.set())

    listener = TranscriptListener(args.topic)
    listener.start()

    cpu_samples: List[float] = []
    ram_samples: List[float] = []
    vram_samples: List[float] = []

    print(
        f"[BENCHMARK] Recording '{args.label}' on {args.topic} "
        f"(duration={'until Ctrl+C' if args.duration <= 0 else f'{args.duration}s'})...",
        flush=True,
    )
    t_start = time.monotonic()
    while not stop_flag.is_set():
        if args.duration > 0 and (time.monotonic() - t_start) >= args.duration:
            break
        pid = find_stt_pid()
        if pid is not None:
            cpu = sample_cpu_percent(pid)
            ram = sample_ram_mb(pid)
            vram = sample_vram_mb(pid)
            if cpu is not None:
                cpu_samples.append(cpu)
            if ram is not None:
                ram_samples.append(ram)
            vram_samples.append(vram)
        time.sleep(1.0)

    transcripts = listener.stop()
    elapsed = round(time.monotonic() - t_start, 2)

    def _stats(vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {"avg": 0.0, "peak": 0.0}
        return {"avg": round(statistics.mean(vals), 2), "peak": round(max(vals), 2)}

    rep_count = sum(1 for t in transcripts if t["has_repetition"])
    avg_words = round(statistics.mean([t["words"] for t in transcripts]), 2) if transcripts else 0.0

    payload = {
        "label": args.label,
        "duration_sec": elapsed,
        "samples_count": len(cpu_samples),
        "cpu_percent": _stats(cpu_samples),
        "ram_mb": _stats(ram_samples),
        "vram_mb": _stats(vram_samples),
        "total_transcripts": len(transcripts),
        "repetition_count": rep_count,
        "avg_words_per_utterance": avg_words,
        "transcripts": transcripts,
    }

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[BENCHMARK] Saved to {args.out} ({len(transcripts)} transcripts, {elapsed}s).")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    with open(args.python_json, "r", encoding="utf-8") as fh:
        py = json.load(fh)
    with open(args.cpp_json, "r", encoding="utf-8") as fh:
        cp = json.load(fh)

    print("\n### Perbandingan STT: Python (`athome_hri`) vs C++ (`athome_hri_cpp`)\n")
    print(f"| Metrik | {py.get('label', 'python')} | {cp.get('label', 'cpp')} |")
    print("| :--- | ---: | ---: |")
    print(f"| Durasi Uji (s) | {py['duration_sec']} | {cp['duration_sec']} |")
    print(
        f"| CPU % (Avg / Peak) | {py['cpu_percent']['avg']} / {py['cpu_percent']['peak']} | "
        f"{cp['cpu_percent']['avg']} / {cp['cpu_percent']['peak']} |"
    )
    print(
        f"| RAM MB (Avg / Peak) | {py['ram_mb']['avg']} / {py['ram_mb']['peak']} | "
        f"{cp['ram_mb']['avg']} / {cp['ram_mb']['peak']} |"
    )
    print(
        f"| VRAM GPU MB (Avg / Peak) | {py['vram_mb']['avg']} / {py['vram_mb']['peak']} | "
        f"{cp['vram_mb']['avg']} / {cp['vram_mb']['peak']} |"
    )
    print(f"| Total Transkrip | {py['total_transcripts']} | {cp['total_transcripts']} |")
    print(f"| Transkrip Berulang | {py['repetition_count']} | {cp['repetition_count']} |")
    print(
        f"| Rata-rata Kata/Ucapan | {py['avg_words_per_utterance']} | "
        f"{cp['avg_words_per_utterance']} |"
    )

    print("\n#### Daftar Transkrip")
    for title, data in ((py.get("label", "python"), py), (cp.get("label", "cpp"), cp)):
        print(f"\n**{title}:**")
        if not data["transcripts"]:
            print("- *(tidak ada transkrip terekam)*")
        for idx, t in enumerate(data["transcripts"], 1):
            rep_tag = " [REPETITION]" if t["has_repetition"] else ""
            print(f"{idx}. `[{t['t_rel_sec']}s]` {t['text']}{rep_tag}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Python vs C++ STT nodes.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record", help="Record CPU, RAM, VRAM, and /hri/stt_transcript")
    rec.add_argument("--label", required=True, help="Run label (e.g. python or cpp)")
    rec.add_argument("--out", required=True, help="Output JSON path")
    rec.add_argument("--duration", type=float, default=60.0, help="Duration in seconds (0 = until Ctrl+C)")
    rec.add_argument("--topic", default="/hri/stt_transcript", help="STT transcript topic")

    cmp_p = sub.add_parser("compare", help="Compare two recorded JSON files")
    cmp_p.add_argument("python_json", help="JSON recorded from Python STT")
    cmp_p.add_argument("cpp_json", help="JSON recorded from C++ STT")

    args = parser.parse_args()
    if args.cmd == "record":
        return cmd_record(args)
    return cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main())

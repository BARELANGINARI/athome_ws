"""Modul Text-to-Speech (TTS) untuk package ``athome_hri``.

Modul ini membungkus engine neural voice **Piper TTS (ONNX)** untuk mengubah
teks menjadi audio WAV mono 16-bit berformat Base64.

Fitur utama untuk sistem robotika real-time:
1. **Akselerasi GPU (CUDA) + Fallback CPU:** Mencoba memuat model di GPU jika
   ``use_cuda=True`` dan otomatis beralih ke CPU bila CUDA tidak tersedia.
2. **Disk Caching Berbasis Hash:** Menyimpan hasil sintesis ke file ``.wav``
   di direktori cache agar kalimat yang sering diucapkan robot cukup disintesis
   sekali dan langsung diputar ulang pada pemanggilan berikutnya.
3. **Trailing Silence 300 ms:** Menambahkan jeda senyap di akhir audio supaya
   suku kata terakhir tidak terpotong oleh hardware audio buffer.
"""

import base64
import hashlib
import io
import logging
import os
from pathlib import Path
from typing import Optional
import wave

logger = logging.getLogger("athome_hri.tts")

try:
    from piper.voice import PiperVoice

    HAS_PIPER = True
except ImportError:
    HAS_PIPER = False


class TextToSpeech:
    """Engine sintesis suara (TTS) berbasis Piper ONNX dengan cache WAV lokal.

    Alur kerja:
        teks -> cek cache ``.wav`` -> (jika belum ada) sintesis Piper -> simpan
        ke ``.wav`` -> kembalikan string Base64 siap putar.
    """

    def __init__(
        self,
        model_filename: str = "en_US-lessac-medium.onnx",
        use_cuda: bool = True,
        cache_dir: Optional[str] = None,
    ) -> None:
        """Inisialisasi model Piper TTS dan direktori cache audio.

        Args:
            model_filename: Nama file model ``.onnx`` Piper (harus disertai
                file konfigurasi ``.onnx.json`` pada folder yang sama).
            use_cuda: Jika ``True``, coba muat model di GPU NVIDIA (CUDA)
                terlebih dahulu sebelum fallback ke CPU.
            cache_dir: Path folder cache WAV kustom. Jika ``None``, otomatis
                menggunakan ``~/.cache/athome_hri/tts_cache``.
        """
        self.piper: Optional["PiperVoice"] = None
        self.model_filename = model_filename
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir
            else Path.home() / ".cache" / "athome_hri" / "tts_cache"
        )
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning("Failed to create TTS cache dir %s: %s", self.cache_dir, exc)

        if not HAS_PIPER:
            logger.warning("piper-tts is not installed. TTS disabled.")
            return

        model_path = self._find_model(model_filename)
        if not model_path:
            logger.warning(
                "Model '%s' not found in search paths.", model_filename
            )
            return

        try:
            logger.info("Loading Piper TTS model from: %s (use_cuda=%s)", model_path, use_cuda)
            if use_cuda:
                try:
                    self.piper = PiperVoice.load(model_path, use_cuda=True)
                    logger.info("Piper TTS ready on GPU (CUDA).")
                    return
                except Exception as cuda_exc:
                    # Fallback ke CPU agar node TTS tetap jalan meski onnxruntime-gpu / VRAM bermasalah
                    logger.warning(
                        "Failed to initialize Piper TTS on CUDA (%s). Falling back to CPU.",
                        cuda_exc,
                    )
            self.piper = PiperVoice.load(model_path, use_cuda=False)
            logger.info("Piper TTS ready on CPU.")
        except Exception as exc:
            logger.error("Failed to load Piper model: %s", exc)

    @staticmethod
    def _find_model(filename: str) -> Optional[str]:
        """Cari lokasi file model ONNX di folder instalasi ROS 2 maupun source.

        Urutan pencarian:
            1. ``<share>/athome_hri/model/<filename>`` (via ``ament_index``)
            2. Folder ``model/`` relatif terhadap lokasi modul Python ini
            3. ``./src/athome_hri/model/<filename>`` dari working directory

        Args:
            filename: Nama file model ``.onnx``.

        Returns:
            Path absolut file model jika ditemukan, atau ``None`` jika tidak ada.
        """
        candidates: list[Path] = []
        try:
            from ament_index_python.packages import get_package_share_directory

            share_dir = Path(get_package_share_directory("athome_hri"))
            candidates.append(share_dir / "model" / filename)
        except Exception:
            pass

        module_dir = Path(__file__).resolve().parent
        candidates.extend(
            [
                module_dir.parent / "model" / filename,
                module_dir / "model" / filename,
                Path.cwd() / "src" / "athome_hri" / "model" / filename,
            ]
        )

        for path in candidates:
            if path.is_file():
                return str(path)
        return None

    def _get_cache_path(self, normalized_text: str) -> Path:
        """Bangun path file cache ``.wav`` unik dari nama model dan teks.

        Menyertakan ``model_filename`` pada key hash memastikan cache tidak
        tertukar jika suara/model TTS diganti.
        """
        key = f"{self.model_filename}:{normalized_text}".encode("utf-8")
        digest = hashlib.sha256(key).hexdigest()[:24]
        return self.cache_dir / f"{digest}.wav"

    def synthesize_wav_base64(self, text: str) -> str:
        """Ubah teks menjadi audio WAV mono 16-bit dalam format Base64.

        Jika teks sudah pernah disintesis sebelumnya, audio langsung dibaca
        dari file ``.wav`` di cache tanpa menjalankan inferensi Piper.

        Args:
            text: Kalimat yang akan diucapkan robot.

        Returns:
            String Base64 berisi file WAV lengkap, atau string kosong ``""``
            jika teks kosong atau proses sintesis gagal.
        """
        if not text or not text.strip():
            return ""

        clean_text = text.strip()
        cache_path = self._get_cache_path(clean_text)

        # 1. Cek cache disk (> 44 byte = ukuran minimal header WAV valid)
        try:
            if cache_path.is_file() and cache_path.stat().st_size > 44:
                wav_bytes = cache_path.read_bytes()
                logger.debug("TTS cache hit: %s", cache_path.name)
                return base64.b64encode(wav_bytes).decode("utf-8")
        except Exception as exc:
            logger.warning("Failed reading TTS cache %s: %s", cache_path, exc)

        if not self.piper:
            return ""

        # 2. Cache miss -> sintesis audio baru lewat Piper TTS
        try:
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.piper.config.sample_rate)
                for chunk in self.piper.synthesize(clean_text):
                    wav_file.writeframes(chunk.audio_int16_bytes)
                # Tambahkan trailing silence (300 ms) agar suku kata terakhir tidak terpotong hardware buffer
                trailing_silence = b"\x00" * int(self.piper.config.sample_rate * 2 * 0.3)
                wav_file.writeframes(trailing_silence)

            wav_bytes = buffer.getvalue()
            # Tulis ke file sementara lalu replace agar aman dari file korup jika node dimatikan mendadak
            try:
                tmp_path = cache_path.with_suffix(".tmp")
                tmp_path.write_bytes(wav_bytes)
                tmp_path.replace(cache_path)
            except Exception as cache_exc:
                logger.warning("Failed writing TTS cache %s: %s", cache_path, cache_exc)

            return base64.b64encode(wav_bytes).decode("utf-8")
        except Exception as exc:
            logger.error("Audio synthesis failed for '%s': %s", text, exc)
            return ""

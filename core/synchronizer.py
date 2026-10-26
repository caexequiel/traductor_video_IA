import math
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import soundfile as sf
import numpy as np

class AudioSynchronizer:
    """Ajusta la duración y velocidad de cada frase doblada para que encaje con el timing original."""

    @staticmethod
    def _stretch_audio_ffmpeg(input_wav: str, output_wav: str, speed_factor: float) -> str:
        """Acelera o ralentiza el audio preservando el tono mediante el filtro atempo de FFmpeg."""
        speed_factor = max(0.5, min(2.0, speed_factor))
        filters = []
        
        # atempo solo soporta entre 0.5 y 2.0 por filtro
        if 0.5 <= speed_factor <= 2.0:
            filters.append(f"atempo={speed_factor:.3f}")
        elif speed_factor > 2.0:
            filters.append("atempo=2.0")
            filters.append(f"atempo={speed_factor / 2.0:.3f}")
        else:
            filters.append("atempo=0.5")
            filters.append(f"atempo={speed_factor / 0.5:.3f}")

        filter_str = ",".join(filters)
        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_wav,
            "-filter:a", filter_str,
            "-vn",
            output_wav
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Error al sincronizar audio con FFmpeg: {result.stderr}")
        return output_wav

    @classmethod
    def synchronize_segments(
        cls,
        segments: List[Dict[str, Any]],
        total_duration: float,
        output_master_wav: str,
        sample_rate: int = 24000,
        progress_cb: Optional[Callable[[float, str], None]] = None
    ) -> str:
        """Crea una pista continua de audio sincronizada al milisegundo con la pista original."""
        total_samples = int(math.ceil(total_duration * sample_rate))
        master_buffer = np.zeros(total_samples, dtype=np.float32)

        total_segs = len(segments)
        temp_dir = Path(output_master_wav).parent / "temp_sync"
        temp_dir.mkdir(parents=True, exist_ok=True)

        for i, seg in enumerate(segments):
            cloned_path = seg.get("cloned_audio_path")
            if not cloned_path or not Path(cloned_path).exists():
                continue

            orig_start = float(seg["start"])
            orig_end = float(seg["end"])
            target_dur = max(0.2, orig_end - orig_start)

            # Duración del audio sintetizado
            cloned_info = sf.info(cloned_path)
            cloned_dur = cloned_info.duration

            if cloned_dur <= 0:
                continue

            speed_factor = cloned_dur / target_dur

            # Limitar la aceleración a un rango comprensible y natural (0.85x a 1.35x)
            effective_speed = max(0.85, min(1.35, speed_factor))

            stretched_wav = temp_dir / f"sync_{i:04d}.wav"
            if abs(effective_speed - 1.0) > 0.03:
                cls._stretch_audio_ffmpeg(cloned_path, str(stretched_wav), effective_speed)
                data, sr = sf.read(str(stretched_wav))
            else:
                data, sr = sf.read(cloned_path)

            if len(data.shape) > 1:
                data = data.mean(axis=1)

            # Resample si difiere de sample_rate
            if sr != sample_rate:
                num_new_samples = int(round(len(data) * float(sample_rate) / sr))
                from scipy import signal
                data = signal.resample(data, num_new_samples)

            # Insertar en master_buffer en orig_start
            start_idx = int(round(orig_start * sample_rate))
            end_idx = start_idx + len(data)

            if start_idx < total_samples:
                actual_end = min(total_samples, end_idx)
                slice_len = actual_end - start_idx
                # Sumar con mezcla suave (suavizado de bordes)
                master_buffer[start_idx:actual_end] += data[:slice_len]

            if progress_cb:
                pct = (i + 1) / total_segs
                progress_cb(round(pct, 2), f"Sincronizando frase {i + 1}/{total_segs} ({orig_start:.1f}s -> {orig_end:.1f}s)")

        # Normalizar para evitar clipping
        max_val = np.max(np.abs(master_buffer))
        if max_val > 0.95:
            master_buffer = master_buffer * (0.95 / max_val)

        Path(output_master_wav).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_master_wav, master_buffer, sample_rate)
        return output_master_wav

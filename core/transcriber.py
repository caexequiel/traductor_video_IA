import os
import gc
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import torch

class SubtitleParser:
    """Parsea archivos .srt y .vtt a una lista uniforme de segmentos."""

    @staticmethod
    def _time_to_seconds(hours: int, minutes: int, seconds: int, milliseconds: int) -> float:
        return hours * 3600.0 + minutes * 60.0 + seconds + milliseconds / 1000.0

    @classmethod
    def parse_srt(cls, srt_path: str) -> List[Dict[str, Any]]:
        segments = []
        pattern = re.compile(
            r"(\d+)\r?\n"
            r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\r?\n"
            r"([\s\S]*?)(?=\r?\n\r?\n\d+|\r?\n*$)"
        )
        with open(srt_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        for match in pattern.finditer(content):
            _, sh, sm, ss, sms, eh, em, es, ems, text = match.groups()
            start = cls._time_to_seconds(int(sh), int(sm), int(ss), int(sms))
            end = cls._time_to_seconds(int(eh), int(em), int(es), int(ems))
            clean_text = " ".join(line.strip() for line in text.strip().splitlines() if line.strip())
            # Quitar tags HTML como <i></i>, <font>
            clean_text = re.sub(r"<[^>]+>", "", clean_text)
            if clean_text:
                segments.append({
                    "id": len(segments),
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "text": clean_text
                })
        return segments

    @classmethod
    def parse_file(cls, filepath: str) -> List[Dict[str, Any]]:
        ext = Path(filepath).suffix.lower()
        if ext in [".srt", ".vtt"]:
            return cls.parse_srt(filepath)
        raise ValueError(f"Formato de subtítulo no soportado: {ext}")


class WhisperTranscriber:
    """Transcriptor automático con faster-whisper optimizado para GPU."""

    def __init__(self, model_size: str = "medium", device: str = "cuda", compute_type: str = "float16"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None
    ) -> List[Dict[str, Any]]:
        from faster_whisper import WhisperModel

        if progress_cb:
            progress_cb(0.1, f"Cargando modelo faster-whisper ({self.model_size})...")

        device_to_use = self.device if torch.cuda.is_available() else "cpu"
        comp_type = self.compute_type if device_to_use == "cuda" else "int8"

        model = WhisperModel(
            self.model_size,
            device=device_to_use,
            compute_type=comp_type,
            download_root=os.path.expanduser("~/.cache/faster_whisper")
        )

        if progress_cb:
            progress_cb(0.3, "Transcribiendo y detectando marcas de tiempo...")

        segments_generator, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400)
        )

        results = []
        total_duration = info.duration or 1.0

        for segment in segments_generator:
            text = segment.text.strip()
            if text:
                results.append({
                    "id": len(results),
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                    "text": text,
                    "confidence": getattr(segment, "avg_logprob", 0.0)
                })
                if progress_cb and total_duration > 0:
                    pct = 0.3 + 0.6 * min(1.0, segment.end / total_duration)
                    progress_cb(round(pct, 2), f"Transcribiendo: {segment.start:.1f}s / {total_duration:.1f}s")

        # Liberar memoria de GPU inmediatamente
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if progress_cb:
            progress_cb(1.0, f"Transcripción completada ({len(results)} segmentos). Idioma detectado: {info.language}")

        return results

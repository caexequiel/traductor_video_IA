import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

class AudioExtractor:
    """Extrae pistas de audio y metadatos de archivos de video mediante FFmpeg/FFprobe."""

    @staticmethod
    def get_media_info(video_path: str) -> Dict[str, Any]:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Error al inspeccionar video con ffprobe: {result.stderr}")
        
        data = json.loads(result.stdout)
        format_info = data.get("format", {})
        streams = data.get("streams", [])

        has_audio = any(s.get("codec_type") == "audio" for s in streams)
        has_video = any(s.get("codec_type") == "video" for s in streams)
        subtitle_streams = [
            {
                "index": s.get("index"),
                "codec": s.get("codec_name"),
                "lang": s.get("tags", {}).get("language", "und"),
                "title": s.get("tags", {}).get("title", f"Sub {s.get('index')}")
            }
            for s in streams if s.get("codec_type") == "subtitle"
        ]

        duration = float(format_info.get("duration", 0.0))

        return {
            "duration": duration,
            "has_audio": has_audio,
            "has_video": has_video,
            "subtitle_streams": subtitle_streams,
            "size_bytes": int(format_info.get("size", 0)),
            "filename": Path(video_path).name
        }

    @staticmethod
    def extract_audio(video_path: str, output_wav_path: str, sample_rate: int = 24000) -> str:
        """Extrae el audio del video en formato WAV mono/estéreo con sample rate especificado."""
        Path(output_wav_path).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", "1",  # Mono para modelos de voz
            output_wav_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg falló al extraer audio: {result.stderr}")
        return output_wav_path

    @staticmethod
    def extract_embedded_subtitle(video_path: str, sub_stream_index: int, output_srt_path: str) -> str:
        """Extrae una pista de subtítulos embebida del contenedor MKV/MP4 a formato SRT."""
        Path(output_srt_path).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-map", f"0:{sub_stream_index}",
            output_srt_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg falló al extraer subtítulo: {result.stderr}")
        return output_srt_path

    @classmethod
    def find_available_subtitles(cls, video_path: str):
        """Busca todas las pistas de subtítulos disponibles (embebidas en el contenedor o archivos en la carpeta)."""
        subs = []
        vpath = Path(video_path).resolve()
        
        # 1. Subtítulos embebidos en el contenedor
        try:
            info = cls.get_media_info(str(vpath))
            for st in info.get("subtitle_streams", []):
                lang = (st.get("lang") or "und").lower()
                codec = st.get("codec") or "sub"
                title = st.get("title") or f"Pista {st.get('index')}"
                is_es = any(x in lang or x in title.lower() for x in ["es", "spa", "esp", "lat", "castellano"])
                subs.append({
                    "source": "embedded",
                    "stream_index": st.get("index"),
                    "codec": codec,
                    "language": lang,
                    "title": f"Pista {st.get('index')}: {title} ({lang.upper()})",
                    "is_spanish": is_es,
                    "path": None
                })
        except Exception:
            pass

        # 2. Subtítulos externos en el mismo directorio
        parent_dir = vpath.parent
        if parent_dir.exists() and parent_dir.is_dir():
            stem_lower = vpath.stem.lower()
            valid_exts = {".srt", ".vtt", ".ass", ".ssa"}
            for f in parent_dir.iterdir():
                if f.is_file() and f.suffix.lower() in valid_exts:
                    fname_lower = f.name.lower()
                    is_match = (f.stem.lower() == stem_lower) or (f.stem.lower().startswith(stem_lower)) or any(x in fname_lower for x in ["sub", "esp", "spanish"])
                    if is_match or len(list(parent_dir.glob("*.srt"))) + len(list(parent_dir.glob("*.ass"))) <= 5:
                        is_es = any(x in fname_lower for x in ["es", "spa", "esp", "lat", "subespañol", "subespanol"])
                        lang = "es" if is_es else "und"
                        subs.append({
                            "source": "external_file",
                            "stream_index": None,
                            "codec": f.suffix.lower().replace(".", ""),
                            "language": lang,
                            "title": f"Archivo: {f.name}",
                            "is_spanish": is_es,
                            "path": str(f.resolve())
                        })

        return subs


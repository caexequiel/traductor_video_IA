import subprocess
from pathlib import Path
from typing import Optional

class VideoMuxer:
    """Múltiplexa el video original con la nueva pista de audio en castellano sin recodificar video."""

    @staticmethod
    def mux(
        video_path: str,
        dialogue_audio_path: str,
        output_video_path: str,
        background_audio_path: Optional[str] = None
    ) -> str:
        Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)

        if background_audio_path and Path(background_audio_path).exists():
            # Mezclar audio de diálogo doblado + fondo original con ducking leve
            cmd = [
                "ffmpeg",
                "-y",
                "-i", video_path,
                "-i", dialogue_audio_path,
                "-i", background_audio_path,
                "-filter_complex",
                "[1:a]volume=1.05[vocal];[2:a]volume=0.75[bg];[vocal][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]",
                "-map", "0:v:0",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                output_video_path
            ]
        else:
            # Solo la pista de diálogo doblada
            cmd = [
                "ffmpeg",
                "-y",
                "-i", video_path,
                "-i", dialogue_audio_path,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                output_video_path
            ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            # Fallback en caso de incompatibilidad con -c:v copy
            cmd_fallback = [
                "ffmpeg",
                "-y",
                "-i", video_path,
                "-i", dialogue_audio_path,
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "22",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                output_video_path
            ]
            res_fb = subprocess.run(cmd_fallback, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res_fb.returncode != 0:
                raise RuntimeError(f"FFmpeg falló al generar el video final: {result.stderr}\nFallback: {res_fb.stderr}")

        return output_video_path

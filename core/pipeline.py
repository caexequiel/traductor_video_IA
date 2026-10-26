import os
import gc
import uuid
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import torch

from core.audio_extractor import AudioExtractor
from core.separator import AudioSeparator
from core.transcriber import SubtitleParser, WhisperTranscriber
from core.translator import TextTranslator
from core.cloner import VoiceCloner
from core.synchronizer import AudioSynchronizer
from core.muxer import VideoMuxer

class DubbingPipeline:
    """Orquestador completo del flujo de doblaje inteligente."""

    def __init__(self, work_dir: str = "temp", output_dir: str = "output"):
        self.work_dir = Path(work_dir)
        self.output_dir = Path(output_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        video_path: str,
        subtitle_path: Optional[str] = None,
        whisper_model: str = "medium",
        preserve_bg: bool = True,
        source_language: Optional[str] = None,
        progress_cb: Optional[Callable[[str, float, str, Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo.
        progress_cb(stage, percentage, message, extra_data)
        """
        run_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        job_dir = self.work_dir / run_id
        job_dir.mkdir(parents=True, exist_ok=True)

        def emit(stage: str, pct: float, msg: str, extra: Optional[Dict[str, Any]] = None):
            if progress_cb:
                progress_cb(stage, round(pct, 2), msg, extra or {})

        try:
            # 1. Análisis de video
            emit("extract", 0.05, "Inspeccionando formato y pistas del video...")
            media_info = AudioExtractor.get_media_info(video_path)
            duration = media_info["duration"]

            # Extraer audio
            emit("extract", 0.15, "Extrayendo pista de audio con FFmpeg...")
            extracted_wav = str(job_dir / "raw_audio.wav")
            AudioExtractor.extract_audio(video_path, extracted_wav, sample_rate=24000)
            emit("extract", 1.0, f"Audio extraído correctamente ({duration:.1f}s).")

            # 2. Separación vocal / fondo (opcional)
            vocals_wav = extracted_wav
            bg_wav = None
            if preserve_bg:
                emit("separation", 0.1, "Separando voz de música y efectos de fondo...")
                vocals_wav, bg_wav = AudioSeparator.separate(
                    extracted_wav,
                    str(job_dir / "separation"),
                    progress_cb=lambda p, m: emit("separation", p, m)
                )

            # 3. Transcripción / Subtítulos
            emit("transcription", 0.05, "Obteniendo marcas de tiempo y diálogos...")
            segments = []
            if subtitle_path and Path(subtitle_path).exists():
                emit("transcription", 0.5, f"Leyendo subtítulos proporcionados: {Path(subtitle_path).name}...")
                segments = SubtitleParser.parse_file(subtitle_path)
                emit("transcription", 1.0, f"{len(segments)} frases cargadas desde subtítulos.")
            else:
                emit("transcription", 0.1, f"Transcribiendo con faster-whisper ({whisper_model})...")
                transcriber = WhisperTranscriber(model_size=whisper_model)
                segments = transcriber.transcribe(
                    vocals_wav,
                    language=source_language,
                    progress_cb=lambda p, m: emit("transcription", p, m)
                )

            if not segments:
                raise ValueError("No se encontraron diálogos ni frases para doblar en el video.")

            emit("transcription", 1.0, f"Transcripción lista ({len(segments)} segmentos encontrados).", {"segments": segments})

            # 4. Traducción al castellano
            emit("translation", 0.1, "Traduciendo frases al castellano...")
            translator = TextTranslator(target_lang="es")
            segments = translator.translate_segments(
                segments,
                source_lang=source_language or "en",
                progress_cb=lambda p, m: emit("translation", p, m)
            )
            emit("translation", 1.0, "Traducción completada con éxito.", {"segments": segments})

            # 5. Extracción de muestra de voz del interlocutor original
            emit("voice_cloning", 0.05, "Extrayendo muestra del timbre de voz original...")
            speaker_sample = str(job_dir / "speaker_sample.wav")
            cloner = VoiceCloner(output_dir=str(job_dir / "cloned_audio"))
            cloner.extract_voice_sample(vocals_wav, segments, speaker_sample)

            # 6. Clonación de voz (XTTS-v2)
            emit("voice_cloning", 0.1, "Sintetizando voz en castellano con clonación...")
            segments = cloner.clone_segments(
                segments,
                speaker_wav_path=speaker_sample,
                target_language="es",
                progress_cb=lambda p, m: emit("voice_cloning", p, m)
            )
            emit("voice_cloning", 1.0, "Clonación de voces finalizada.")

            # 7. Sincronización temporal con preservación de tono
            emit("synchronization", 0.1, "Sincronizando tiempos y pausas con el video original...")
            master_dubbed_wav = str(job_dir / "master_dubbed.wav")
            AudioSynchronizer.synchronize_segments(
                segments,
                total_duration=duration,
                output_master_wav=master_dubbed_wav,
                sample_rate=24000,
                progress_cb=lambda p, m: emit("synchronization", p, m)
            )
            emit("synchronization", 1.0, "Pista de audio en castellano perfectamente sincronizada.")

            # 8. Muxing final de video
            emit("muxing", 0.2, "Generando el nuevo archivo de video con FFmpeg...")
            video_stem = Path(video_path).stem
            final_video_name = f"{video_stem}_es_{run_id}.mp4"
            final_video_path = str(self.output_dir / final_video_name)

            VideoMuxer.mux(
                video_path=video_path,
                dialogue_audio_path=master_dubbed_wav,
                output_video_path=final_video_path,
                background_audio_path=bg_wav
            )
            emit("muxing", 1.0, "¡Video completado con éxito!", {
                "output_video_path": final_video_path,
                "filename": final_video_name
            })

            return {
                "status": "success",
                "output_video": final_video_path,
                "filename": final_video_name,
                "duration": duration,
                "total_segments": len(segments),
                "segments": segments
            }

        except Exception as e:
            emit("error", 0.0, f"Error en el proceso: {str(e)}")
            raise e
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

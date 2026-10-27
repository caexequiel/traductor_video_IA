import os
import gc
import re
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
        embedded_sub_index: Optional[int] = None,
        sub_already_translated: bool = False,
        whisper_model: str = "large-v3",
        preserve_bg: bool = True,
        source_language: Optional[str] = None,
        dialect: str = "es-latam",
        progress_cb: Optional[Callable[[str, float, str, Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo de doblaje inteligente.
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
            detected_lang = source_language

            # Si se especificó una pista embebida, extraerla primero
            if embedded_sub_index is not None:
                emit("transcription", 0.1, f"Extrayendo pista de subtítulos incrustada #{embedded_sub_index}...")
                extracted_sub = str(job_dir / "embedded_sub.srt")
                try:
                    AudioExtractor.extract_embedded_subtitle(video_path, embedded_sub_index, extracted_sub)
                    if Path(extracted_sub).exists():
                        subtitle_path = extracted_sub
                except Exception as ex:
                    print(f"[Aviso] No se pudo extraer pista embebida #{embedded_sub_index}: {ex}")

            is_already_spanish = False

            if subtitle_path == "hardcoded_ocr":
                emit("transcription", 0.1, f"Detectando momentos de habla con faster-whisper ({whisper_model})...")
                transcriber = WhisperTranscriber(model_size=whisper_model)
                segments, detected_lang = transcriber.transcribe(
                    vocals_wav,
                    language=source_language,
                    progress_cb=lambda p, m: emit("transcription", p * 0.7, m)
                )
                emit("transcription", 0.75, "Leyendo subtítulos directamente de la pantalla del video (OCR)...")
                from core.ocr_subtitles import VideoSubtitleOCR
                segments = VideoSubtitleOCR.extract_subtitles_for_segments(
                    video_path,
                    segments,
                    dialect=dialect,
                    progress_cb=lambda p, m: emit("transcription", 0.75 + 0.25 * p, m)
                )
                is_already_spanish = True
                effective_source = "es"
            elif subtitle_path and Path(subtitle_path).exists():
                emit("transcription", 0.5, f"Leyendo subtítulos: {Path(subtitle_path).name}...")
                segments = SubtitleParser.parse_file(subtitle_path)
                emit("transcription", 1.0, f"{len(segments)} frases cargadas desde subtítulos.")
                sample_words = set(re.findall(r'\b\w+\b', " ".join(s.get("text", "") for s in segments[:15]).lower()))
                es_common = {"que", "de", "no", "a", "la", "el", "es", "y", "en", "lo", "un", "por", "me", "una", "te", "los", "se"}
                is_already_spanish = sub_already_translated or (len(sample_words.intersection(es_common)) >= 3)
                effective_source = "es" if is_already_spanish else (source_language or "en")
            else:
                emit("transcription", 0.1, f"Transcribiendo con faster-whisper ({whisper_model})...")
                transcriber = WhisperTranscriber(model_size=whisper_model)
                segments, detected_lang = transcriber.transcribe(
                    vocals_wav,
                    language=source_language,
                    progress_cb=lambda p, m: emit("transcription", p, m)
                )
                effective_source = source_language or detected_lang or "ja"

            if not segments:
                raise ValueError("No se encontraron diálogos ni frases para doblar en el video.")

            emit("transcription", 1.0, f"Diálogos listos ({len(segments)} frases). Idioma: {effective_source.upper()}", {"segments": segments, "language": effective_source})

            # 4. Traducción al español (Latino o Castellano)
            dialect_label = "Español Latino" if "latam" in dialect else "Castellano"
            translator = TextTranslator(target_lang="es", dialect=dialect)

            def sanitize_repetitions(txt: str) -> str:
                # Evitar bucles como "no, no, no, no..."
                t = re.sub(r'\b(\w+)(?:[,\s]+\1\b){2,}', r'\1', txt, flags=re.IGNORECASE)
                return t.strip()

            if is_already_spanish:
                emit("translation", 0.5, f"Subtítulos en español listos. Aplicando localización a {dialect_label}...")
                for s in segments:
                    orig_txt = s.get("translated_text") or s.get("text", "")
                    s["translated_text"] = sanitize_repetitions(translator.to_latin_american_spanish(orig_txt) if "latam" in dialect else orig_txt)
                emit("translation", 1.0, f"Localización a {dialect_label} completada desde subtítulos.", {"segments": segments})
            else:
                emit("translation", 0.1, f"Traduciendo frases ({effective_source.upper()} -> {dialect_label})...")
                # Filtrar gritos antes de traducir
                for s in segments:
                    if re.match(r'^[あいうえおぁぃぅぇぉアイウエオァィゥェォー!！?？\s~〜・]+$', s.get("text", "").strip()):
                        s["is_scream"] = True
                        s["translated_text"] = ""

                trans_candidates = [s for s in segments if not s.get("is_scream")]
                if trans_candidates:
                    translator.translate_segments(
                        trans_candidates,
                        source_lang=effective_source,
                        progress_cb=lambda p, m: emit("translation", p, m)
                    )
                for s in segments:
                    if s.get("translated_text"):
                        s["translated_text"] = sanitize_repetitions(s["translated_text"])
                emit("translation", 1.0, f"Traducción a {dialect_label} completada con éxito.", {"segments": segments})

            # 5. Extracción de muestra de voz del interlocutor original para fallback
            emit("voice_cloning", 0.05, "Extrayendo muestra del timbre de voz original...")
            speaker_sample = str(job_dir / "speaker_sample.wav")
            cloner = VoiceCloner(output_dir=str(job_dir / "cloned_audio"))
            cloner.extract_voice_sample(vocals_wav, segments, speaker_sample)

            # 6. Clonación de voz emotiva y dinámica por escena (XTTS-v2)
            emit("voice_cloning", 0.1, "Sintetizando voces con entonación emotiva y timbre por escena...")
            segments = cloner.clone_segments(
                segments,
                speaker_wav_path=speaker_sample,
                vocals_wav_path=vocals_wav,
                target_language="es",
                progress_cb=lambda p, m: emit("voice_cloning", p, m)
            )
            emit("voice_cloning", 1.0, "Clonación emotiva de voces finalizada.")

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

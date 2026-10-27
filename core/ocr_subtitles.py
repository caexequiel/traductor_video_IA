import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import cv2
import numpy as np

class VideoSubtitleOCR:
    """Detecta y extrae subtítulos pegados en la imagen del video mediante OCR de alta velocidad."""

    _ocr_engine = None

    @classmethod
    def get_engine(cls):
        if cls._ocr_engine is None:
            from rapidocr_onnxruntime import RapidOCR
            cls._ocr_engine = RapidOCR()
        return cls._ocr_engine

    @classmethod
    def detect_hardcoded_subtitles(cls, video_path: str) -> bool:
        """Determina rápidamente si el video tiene subtítulos en español impresos en pantalla."""
        vpath = Path(video_path).resolve()
        if not vpath.exists():
            return False

        try:
            cap = cv2.VideoCapture(str(vpath))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            if duration < 3.0:
                cap.release()
                return False

            engine = cls.get_engine()
            es_stopwords = {'que', 'de', 'no', 'a', 'la', 'el', 'es', 'y', 'en', 'lo', 'un', 'por', 'me', 'una', 'te', 'los', 'se', 'mi', 'mis', 'para', 'con', 'del', 'al', 'las'}

            sample_checkpoints = [0.08, 0.15, 0.25, 0.40, 0.55, 0.70, 0.85]
            spanish_hits = 0

            for pct in sample_checkpoints:
                sec = duration * pct
                cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
                ret, frame = cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    # Subtítulos casi siempre en el 25% inferior
                    crop = frame[int(h * 0.72):h, :]
                    res, _ = engine(crop)
                    if res:
                        txt = " ".join(r[1] for r in res).lower()
                        words = set(re.findall(r'\b\w+\b', txt))
                        if len(words.intersection(es_stopwords)) >= 1:
                            spanish_hits += 1
                            if spanish_hits >= 2:
                                cap.release()
                                return True

            cap.release()
            return spanish_hits >= 2
        except Exception as e:
            print(f"[Aviso] Error en detect_hardcoded_subtitles: {e}")
            return False

    @classmethod
    def clean_subtitle_text(cls, text: str) -> str:
        """Limpia imperfecciones de OCR en subtítulos."""
        # Unir líneas y espacios dobles
        t = " ".join(text.split()).strip()
        # Eliminar caracteres aislados de ruido al inicio o final
        t = re.sub(r'^[^\w¿¡]+', '', t)
        # Corregir letras pegadas frecuentes
        t = re.sub(r'\b([a-zA-Z])\s+([a-zA-Z]{3,})\b', r'\1 \2', t)
        # Eliminar repeticiones aberrantes (e.g. no no no no)
        t = re.sub(r'\b(\w+)(?:[,\s]+\1\b){2,}', r'\1', t, flags=re.IGNORECASE)
        return t.strip()

    @classmethod
    def is_screaming_or_noise(cls, original_text: str) -> bool:
        """Detecta si un diálogo japonés o segmento es solo un grito o ruido sin contenido textual."""
        clean = original_text.strip()
        # Repetición de vocales o katakana de grito (e.g. ああああ, うおおお, ギャアア)
        if re.match(r'^[あいうえおぁぃぅぇぉアイウエオァィゥェォー!！?？\s~〜・]+$', clean):
            return True
        if len(clean) <= 1:
            return True
        return False

    @classmethod
    def extract_subtitles_for_segments(
        cls,
        video_path: str,
        segments: List[Dict[str, Any]],
        dialect: str = "es-latam",
        progress_cb: Optional[Callable[[float, str], None]] = None
    ) -> List[Dict[str, Any]]:
        """Lee el texto de los subtítulos en pantalla en los momentos exactos en que los personajes hablan."""
        from core.translator import TextTranslator
        translator = TextTranslator(target_lang="es", dialect=dialect)

        vpath = Path(video_path).resolve()
        cap = cv2.VideoCapture(str(vpath))
        engine = cls.get_engine()

        total = len(segments)
        ocr_count = 0

        for idx, seg in enumerate(segments):
            start = float(seg["start"])
            end = float(seg["end"])
            orig_text = seg.get("text", "")

            # Si es solo un grito sin texto y no hay subtítulos, omitir síntesis
            if cls.is_screaming_or_noise(orig_text):
                seg["is_scream"] = True

            # Muestrear en el 40% y 60% de la duración del diálogo para capturar el subtítulo activo
            best_ocr = ""
            for pos_ratio in [0.4, 0.6, 0.2]:
                timestamp_sec = start + (end - start) * pos_ratio
                cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
                ret, frame = cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    crop = frame[int(h * 0.72):h, :]
                    res, _ = engine(crop)
                    if res:
                        raw_ocr = " ".join(r[1] for r in res)
                        cleaned = cls.clean_subtitle_text(raw_ocr)
                        if len(cleaned) > len(best_ocr):
                            best_ocr = cleaned

            if best_ocr and len(best_ocr) >= 3:
                # Localizar al español latino neutro
                loc_text = translator.to_latin_american_spanish(best_ocr) if "latam" in dialect else best_ocr
                seg["translated_text"] = loc_text
                seg["source"] = "ocr_screen"
                seg["is_scream"] = False
                ocr_count += 1
            else:
                # Si no hubo subtítulo en pantalla y era un grito, no forzar texto inventado
                if seg.get("is_scream"):
                    seg["translated_text"] = ""
                # Si había diálogo hablado real pero sin subtítulo, traducir con el modelo
                elif not seg.get("translated_text"):
                    tr = translator.translate_segments([seg], source_lang="ja")
                    if tr and tr[0].get("translated_text"):
                        seg["translated_text"] = tr[0]["translated_text"]

            if progress_cb:
                pct = (idx + 1) / total
                progress_cb(pct, f"Leyendo subtítulos en video: frase {idx + 1}/{total} ('{seg.get('translated_text', '')[:25]}...')")

        cap.release()
        print(f"[VideoSubtitleOCR] Extraídos {ocr_count} subtítulos directamente de pantalla de {total} segmentos.")
        return segments

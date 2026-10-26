import gc
import torch
from typing import List, Dict, Any, Optional, Callable

class TextTranslator:
    """Traductor de texto a español/castellano optimizado para ejecución local."""

    def __init__(self, target_lang: str = "es", device: Optional[str] = None):
        self.target_lang = target_lang
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def translate_segments(
        self,
        segments: List[Dict[str, Any]],
        source_lang: str = "en",
        progress_cb: Optional[Callable[[float, str], None]] = None
    ) -> List[Dict[str, Any]]:
        """Traduce una lista de segmentos con texto al español, preservando timestamps y metadatos."""
        if not segments:
            return []

        if progress_cb:
            progress_cb(0.1, f"Inicializando modelo de traducción ({source_lang} -> {self.target_lang})...")

        # Seleccionar modelo según par de idiomas
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-es"
        # Si el idioma origen no es inglés ni tiene modelo directo, usar modelo multilingüe o fallback
        try:
            from transformers import MarianMTModel, MarianTokenizer
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name).to(self.device)
        except Exception:
            # Fallback a NLLB-200 distilled si el par no existe en opus-mt
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                model_name = "facebook/nllb-200-distilled-600M"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            except Exception as e:
                # Si no hay conexión o falla la descarga, devolver texto con advertencia
                if progress_cb:
                    progress_cb(1.0, f"Aviso: Modelo de traducción no disponible ({e}). Usando texto original.")
                for seg in segments:
                    seg["translated_text"] = seg.get("text", "")
                return segments

        total = len(segments)
        batch_size = 16

        for i in range(0, total, batch_size):
            batch = segments[i:i + batch_size]
            texts = [s.get("text", "") for s in batch]

            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                translated_tokens = model.generate(**inputs, max_length=512)
            
            translated_texts = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

            for seg, trans in zip(batch, translated_texts):
                seg["translated_text"] = trans.strip()

            if progress_cb:
                pct = 0.2 + 0.7 * min(1.0, (i + len(batch)) / total)
                progress_cb(round(pct, 2), f"Traduciendo segmento {min(i + len(batch), total)} / {total}...")

        # Liberar memoria de GPU
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if progress_cb:
            progress_cb(1.0, f"Traducción completada ({total} segmentos traducidos al castellano).")

        return segments

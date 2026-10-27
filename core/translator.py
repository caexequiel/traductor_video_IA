import re
import gc
import torch
from typing import List, Dict, Any, Optional, Callable

# Mapeo de términos de España a Español Latino Neutro
LATAM_REPLACEMENTS = [
    # Pronombres y posesivos
    (r"\bvosotros\b", "ustedes"),
    (r"\bvosotras\b", "ustedes"),
    (r"\bos\b", "les"),
    (r"\bvuestro\b", "su"),
    (r"\bvuestra\b", "su"),
    (r"\bvuestros\b", "sus"),
    (r"\bvuestras\b", "sus"),

    # Verbos en segunda persona plural (vosotros -> ustedes)
    (r"\btenéis\b", "tienen"),
    (r"\btened\b", "tengan"),
    (r"\bsois\b", "son"),
    (r"\bsed\b", "sean"),
    (r"\bvais\b", "van"),
    (r"\bid\b", "vayan"),
    (r"\berais\b", "eran"),
    (r"\bhacéis\b", "hacen"),
    (r"\bhaced\b", "hagan"),
    (r"\bsabéis\b", "saben"),
    (r"\bpodéis\b", "pueden"),
    (r"\bqueréis\b", "quieren"),
    (r"\bdecís\b", "dicen"),
    (r"\bdecid\b", "digan"),
    (r"\bestáis\b", "están"),
    (r"\bestad\b", "estén"),
    (r"\bhabéis\b", "han"),
    (r"\bibais\b", "iban"),
    (r"\bveíais\b", "veían"),
    (r"\bvenís\b", "vienen"),
    (r"\bvenid\b", "vengan"),
    (r"\bmirad\b", "miren"),
    (r"\bescuchad\b", "escuchen"),
    (r"\bcallaos\b", "cállense"),

    # Vocabulario cotidiano España -> Latinoamérica
    (r"\bordenador\b", "computadora"),
    (r"\bordenadores\b", "computadoras"),
    (r"\bcoche\b", "auto"),
    (r"\bcoches\b", "autos"),
    (r"\baparcar\b", "estacionar"),
    (r"\baparcado\b", "estacionado"),
    (r"\baparcamiento\b", "estacionamiento"),
    (r"\bvídeo\b", "video"),
    (r"\bvídeos\b", "videos"),
    (r"\bmóvil\b", "celular"),
    (r"\bmóviles\b", "celulares"),
    (r"\bpiso\b", "departamento"),
    (r"\bpisos\b", "departamentos"),
    (r"\bgafas\b", "lentes"),
    (r"\bzumo\b", "jugo"),
    (r"\bzumos\b", "jugos"),
    (r"\bpatatas\b", "papas"),
    (r"\benfadado\b", "enojado"),
    (r"\benfadada\b", "enojada"),
    (r"\benfadados\b", "enojados"),
    (r"\bchaval\b", "chico"),
    (r"\bchavales\b", "chicos"),
    (r"\bvale\b(?=[,.\s!?]|$)", "de acuerdo"),
    (r"\bcoger\b", "tomar"),
    (r"\bcogí\b", "tomé"),
    (r"\bcogió\b", "tomó"),
    (r"\bcogieron\b", "tomaron"),
    (r"\bcogemos\b", "tomamos"),
]

def to_latin_american_spanish(text: str) -> str:
    """Adapta el texto en español a expresiones y pronombres de Español Latino Neutro."""
    if not text:
        return text

    res = text
    for pattern, repl in LATAM_REPLACEMENTS:
        # Reemplazo preservando mayúscula inicial si existe
        def replace_match(m):
            match_str = m.group(0)
            if match_str[0].isupper():
                return repl.capitalize()
            return repl

        res = re.sub(pattern, replace_match, res, flags=re.IGNORECASE)
    return res


class TextTranslator:
    """Traductor neuronal a español (Castellano o Latinoamericano) optimizado para ejecución local."""

    def __init__(self, target_lang: str = "es", dialect: str = "es-latam", device: Optional[str] = None):
        self.target_lang = target_lang
        self.dialect = dialect
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def translate_segments(
        self,
        segments: List[Dict[str, Any]],
        source_lang: str = "en",
        progress_cb: Optional[Callable[[float, str], None]] = None
    ) -> List[Dict[str, Any]]:
        """Traduce los segmentos de texto al español preservando el timing."""
        if not segments:
            return []

        # Normalizar código de idioma
        norm_source = source_lang.lower().strip()
        if norm_source in ["jpn", "japanese"]:
            norm_source = "ja"
        elif norm_source in ["eng", "english"]:
            norm_source = "en"

        if progress_cb:
            dialect_label = "Español Latino" if "latam" in self.dialect else "Castellano"
            progress_cb(0.1, f"Inicializando modelo de traducción ({norm_source.upper()} -> {dialect_label})...")

        from transformers import MarianMTModel, MarianTokenizer

        model_name = f"Helsinki-NLP/opus-mt-{norm_source}-es"

        # Cargar modelo directo o fallback a NLLB
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name).to(self.device)
        except Exception:
            # Si no existe modelo directo, intentar pivote vía inglés si el idioma no es inglés
            if norm_source != "en":
                try:
                    pivot_name = f"Helsinki-NLP/opus-mt-{norm_source}-en"
                    tokenizer = MarianTokenizer.from_pretrained(pivot_name)
                    model = MarianMTModel.from_pretrained(pivot_name).to(self.device)
                except Exception:
                    # Último recurso: usar NLLB-200
                    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                    model_name = "facebook/nllb-200-distilled-600M"
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        total = len(segments)
        batch_size = 16

        for i in range(0, total, batch_size):
            batch = segments[i:i + batch_size]
            texts = [s.get("text", "").strip() for s in batch]

            # Reemplazar cadenas vacías por espacio
            safe_texts = [t if t else "." for t in texts]

            inputs = tokenizer(safe_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                translated_tokens = model.generate(**inputs, max_length=512)
            
            translated_texts = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

            for seg, trans in zip(batch, translated_texts):
                clean_trans = trans.strip()
                if clean_trans in [".", ""]:
                    clean_trans = seg.get("text", "")

                # Aplicar adaptación a Español Latino si corresponde
                if "latam" in self.dialect:
                    clean_trans = to_latin_american_spanish(clean_trans)

                seg["translated_text"] = clean_trans

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
            dialect_label = "Español Latino" if "latam" in self.dialect else "Castellano"
            progress_cb(1.0, f"Traducción completada ({total} segmentos traducidos a {dialect_label}).")

        return segments

import os
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

# Aceptar automáticamente términos de licencia no interactivos de Coqui
os.environ["COQUI_TOS_AGREED"] = "1"

import torch

# Compatibilidad con PyTorch 2.6 (weights_only=False para checkpoints de Coqui)
_orig_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

import soundfile as sf
import numpy as np
import torchaudio

# Reemplazar torchaudio.load con SoundFile para evitar errores de segmentación de libsox en Linux
def _soundfile_load(p, *args, **kwargs):
    data, sr = sf.read(p, dtype='float32')
    t = torch.from_numpy(data)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    else:
        t = t.t()
    return t, sr
torchaudio.load = _soundfile_load

class VoiceCloner:
    """Clonador de voz zero-shot utilizando XTTS-v2 en castellano."""

    def __init__(self, output_dir: str, device: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def extract_voice_sample(self, audio_path: str, segments: List[Dict[str, Any]], sample_out_path: str) -> str:
        """Extrae un fragmento de voz limpia de entre 4 y 10 segundos del audio original para usar como referencia."""
        data, sr = sf.read(audio_path)
        
        # Buscar el mejor segmento que dure entre 3 y 8 segundos
        best_seg = None
        for s in segments:
            dur = s["end"] - s["start"]
            if 3.0 <= dur <= 8.0:
                best_seg = s
                break
        
        if not best_seg and segments:
            # Si no hay uno de esa duración, tomar el primero que dure más de 1.5s
            for s in segments:
                if (s["end"] - s["start"]) >= 1.5:
                    best_seg = s
                    break

        if best_seg:
            start_sample = int(best_seg["start"] * sr)
            end_sample = int(min(len(data), int(best_seg["end"] * sr)))
            clip = data[start_sample:end_sample]
        else:
            # Tomar los primeros 5 segundos de audio
            clip = data[:int(sr * min(5.0, len(data) / sr))]

        Path(sample_out_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(sample_out_path, clip, sr)
        return sample_out_path

    def _synthesize_safe(self, tts, text: str, speaker_wav_path: str, lang: str, out_wav: Path) -> bool:
        clean_text = " ".join(text.split()).strip()
        if not clean_text:
            return False

        if len(clean_text) <= 200:
            tts.tts_to_file(text=clean_text, speaker_wav=speaker_wav_path, language=lang, file_path=str(out_wav))
            return True

        # Dividir textos extensos en frases de máximo 150 caracteres
        import re
        parts = [p.strip() for p in re.split(r'(?<=[.,;!?])\s+', clean_text) if p.strip()]
        grouped = []
        curr = ""
        for p in parts:
            if len(curr) + len(p) < 150:
                curr = f"{curr} {p}".strip()
            else:
                if curr:
                    grouped.append(curr)
                curr = p
        if curr:
            grouped.append(curr)

        audio_chunks = []
        sr_target = 24000
        for i, chunk_txt in enumerate(grouped):
            temp_sub = out_wav.parent / f"chunk_{out_wav.stem}_{i}.wav"
            try:
                tts.tts_to_file(text=chunk_txt, speaker_wav=speaker_wav_path, language=lang, file_path=str(temp_sub))
                if temp_sub.exists():
                    d, sr_target = sf.read(str(temp_sub))
                    audio_chunks.append(d)
                    temp_sub.unlink(missing_ok=True)
            except Exception as ex:
                print(f"[Error en subfragmento]: {ex}")

        if audio_chunks:
            combined = np.concatenate(audio_chunks)
            sf.write(str(out_wav), combined, sr_target)
            return True

        return False

    def clone_segments(
        self,
        segments: List[Dict[str, Any]],
        speaker_wav_path: str,
        target_language: str = "es",
        progress_cb: Optional[Callable[[float, str], None]] = None
    ) -> List[Dict[str, Any]]:
        """Genera audio clonado para cada segmento traducido."""
        if not segments:
            return []

        if progress_cb:
            progress_cb(0.05, "Cargando modelo XTTS-v2 para clonación de voz...")

        from TTS.api import TTS

        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=(self.device == "cuda"))

        total = len(segments)
        for idx, seg in enumerate(segments):
            text = seg.get("translated_text", "").strip()
            if not text:
                continue

            seg_out_wav = self.output_dir / f"cloned_seg_{seg['id']:04d}.wav"
            
            try:
                success = self._synthesize_safe(
                    tts=tts,
                    text=text,
                    speaker_wav_path=speaker_wav_path,
                    lang=target_language,
                    out_wav=seg_out_wav
                )
                if success and seg_out_wav.exists():
                    seg["cloned_audio_path"] = str(seg_out_wav)
                else:
                    seg["cloned_audio_path"] = None
            except Exception as e:
                print(f"[Error en síntesis del segmento {seg['id']}]: {e}")
                seg["cloned_audio_path"] = None

            if progress_cb:
                pct = 0.1 + 0.85 * ((idx + 1) / total)
                progress_cb(round(pct, 2), f"Clonando voz: frase {idx + 1}/{total} ('{text[:30]}...')")

        # Liberar VRAM
        del tts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if progress_cb:
            progress_cb(1.0, f"Clonación de voces finalizada para {total} frases.")

        return segments

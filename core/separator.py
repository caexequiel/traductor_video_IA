import os
import gc
import subprocess
from pathlib import Path
from typing import Tuple, Optional, Callable
import torch

class AudioSeparator:
    """Separa voz y música/efectos de fondo utilizando Demucs para preservar la banda sonora original."""

    @staticmethod
    def separate(
        audio_path: str,
        output_dir: str,
        progress_cb: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Separa el audio en voces y pista instrumental/fondo.
        Retorna (vocals_path, background_path).
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        vocals_path = str(out_path / "vocals.wav")
        bg_path = str(out_path / "background.wav")

        if progress_cb:
            progress_cb(0.1, "Iniciando separación de audio (conservando música y efectos de fondo)...")

        # Intentar ejecutar Demucs si está disponible
        try:
            import sys
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cmd = [
                sys.executable,
                "-m", "demucs.separate",
                "--two-stems=vocals",
                "-n", "htdemucs",
                "-d", device,
                "-o", str(out_path),
                audio_path
            ]
            if progress_cb:
                progress_cb(0.3, "Ejecutando red neuronal Demucs (htdemucs)...")

            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                audio_stem = Path(audio_path).stem
                demucs_out = out_path / "htdemucs" / audio_stem
                found_vocals = demucs_out / "vocals.wav"
                found_bg = demucs_out / "no_vocals.wav"

                if found_vocals.exists() and found_bg.exists():
                    # Mover o copiar a nombres estándar
                    import shutil
                    shutil.copy(str(found_vocals), vocals_path)
                    shutil.copy(str(found_bg), bg_path)

                    if progress_cb:
                        progress_cb(1.0, "Separación de voz y fondo completada.")
                    return vocals_path, bg_path
        except Exception as e:
            print(f"[Aviso Demucs]: {e}. Continuando con audio unificado.")

        # Si Demucs falla o no está instalado, usar el audio original como voz y None como fondo
        if progress_cb:
            progress_cb(1.0, "Uso directo de pista original para clonación.")
        return audio_path, None

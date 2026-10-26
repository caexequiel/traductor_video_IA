#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
from pathlib import Path

os.environ["COQUI_TOS_AGREED"] = "1"

try:
    import torch
    _orig_torch_load = torch.load
    def _safe_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _safe_torch_load

    import soundfile as sf
    import torchaudio
    def _soundfile_load(p, *args, **kwargs):
        data, sr = sf.read(p, dtype='float32')
        t = torch.from_numpy(data)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        else:
            t = t.t()
        return t, sr
    torchaudio.load = _soundfile_load
except ImportError:
    pass

def check_environment():
    print("=" * 60)
    print("  🎬 Traductor Inteligente de Video - Verificación de Sistema")
    print("=" * 60)

    # 1. Verificar FFmpeg
    if not shutil.which("ffmpeg"):
        print("❌ Error: 'ffmpeg' no está instalado en el sistema.")
        print("   Por favor instala ffmpeg con tu gestor de paquetes.")
        sys.exit(1)
    else:
        print("✅ FFmpeg detectado correctamente.")

    # 2. Verificar PyTorch y CUDA
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            print(f"✅ GPU Acelerada detectada: {device_name} ({vram_mb / 1024:.1f} GB VRAM)")
        else:
            print("⚠️ Advertencia: CUDA no detectado, se ejecutará en modo CPU.")
    except ImportError:
        print("⚠️ PyTorch no está importable en este intérprete.")

    # 3. Carpetas de trabajo
    Path("temp").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    print("✅ Carpetas temporales y de salida verificadas.")
    print("=" * 60)

def main():
    check_environment()
    import uvicorn
    host = "0.0.0.0"
    port = 8000
    print(f"\n🚀 Servidor web iniciado exitosamente.")
    print(f"👉 Abre en tu navegador: http://localhost:{port}\n")
    uvicorn.run("app.main:app", host=host, port=port, reload=False, log_level="info")

if __name__ == "__main__":
    main()

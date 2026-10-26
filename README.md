# traductor_video_IA

Aplicación local con interfaz web para traducir y doblar videos al castellano manteniendo la voz original de los interlocutores mediante **clonación de voz zero-shot** e **IA optimizada para GPUs con 6 GB de VRAM** (como la NVIDIA GeForce RTX 3050).

---

## ✨ Características Principales

1. **Explorador de Documentos y Videos Integrado:**
   - Navega por cualquier carpeta de tu disco (`/home/exequiel`, `Descargas`, `Vídeos`, `Documentos`, etc.) de forma visual.
   - Detecta videos (`.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`) y subtítulos (`.srt`, `.vtt`).
2. **Doble Modo de Detección:**
   - **Auto-transcripción con IA:** Usa `faster-whisper` (CTranslate2) con marcas de tiempo precisas en milisegundos.
   - **Subtítulos Externos:** Carga directamente tus archivos `.srt` o `.vtt`.
3. **Clonación de Voz Zero-Shot en Castellano:**
   - Extrae automáticamente una muestra de 3 a 6 segundos de la voz del interlocutor original.
   - Genera el audio en castellano con el modelo **XTTS-v2**, reproduciendo el timbre, tono y expresividad original.
4. **Sincronización Temporal Milimétrica:**
   - Algoritmo de *pitch-preserved time-stretching* (ajuste de duración sin deformar el tono) para que las frases en español encajen perfectamente en los intervalos de silencio y movimiento labial del video original.
5. **Conservación de Pista de Fondo:**
   - Separa la música y efectos sonoros de fondo (con `Demucs`) para que la banda sonora permanezca intacta y solo se sustituya la voz.
6. **Optimizado para 6 GB de VRAM:**
   - Pipeline por etapas con liberación y vaciado dinámico de memoria VRAM (`torch.cuda.empty_cache()`) entre cada paso.

---

## 🚀 Cómo Iniciar la Aplicación

Desde la terminal en esta carpeta:

```bash
./run.sh
```

O también con:

```bash
.venv/bin/python run.py
```

Luego abre tu navegador web en:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## 🖥️ Uso de la Aplicación

1. Haz clic en **"📂 Abrir Explorador de Archivos"** y selecciona tu video.
2. Elige si prefieres transcribir automáticamente con Whisper o cargar subtítulos existentes.
3. Activa o desactiva la opción de conservar música y efectos de fondo.
4. Presiona **"✨ Iniciar Doblaje y Clonación"**.
5. Podrás ver en tiempo real el progreso etapa por etapa, la tabla de frases traducidas, y al finalizar podrás reproducir y descargar el nuevo video doblado.

from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.routes.fs import router as fs_router
from app.routes.process import router as process_router
from app.routes.media import router as media_router
from app.state import app_state

app = FastAPI(title="Traductor Inteligente de Video", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar routers de la API
app.include_router(fs_router)
app.include_router(process_router)
app.include_router(media_router)

# WebSocket para progreso en vivo
@app.websocket("/ws/progress")
async def websocket_progress(websocket: WebSocket):
    await app_state.register_ws(websocket)
    try:
        while True:
            # Mantener conexión viva recibiendo pings si los hay
            await websocket.receive_text()
    except WebSocketDisconnect:
        app_state.unregister_ws(websocket)
    except Exception:
        app_state.unregister_ws(websocket)

# Montar carpeta estática
static_dir = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
def serve_index():
    index_file = static_dir / "index.html"
    return FileResponse(str(index_file))

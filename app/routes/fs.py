import os
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from core.audio_extractor import AudioExtractor

router = APIRouter(prefix="/api/fs", tags=["filesystem"])

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".ts"}
SUBTITLE_EXTENSIONS = {".srt", ".vtt", ".sub", ".ass"}

@router.get("/bookmarks")
def get_bookmarks():
    home = Path.home()
    bookmarks = [
        {"name": "Carpeta del Proyecto", "path": str(Path.cwd().resolve()), "icon": "folder-star"},
        {"name": "Inicio (~)", "path": str(home), "icon": "home"}
    ]
    
    # Rutas comunes en español e inglés
    candidates = [
        ("Descargas", home / "Descargas"),
        ("Downloads", home / "Downloads"),
        ("Vídeos", home / "Vídeos"),
        ("Videos", home / "Videos"),
        ("Documentos", home / "Documentos"),
        ("Documents", home / "Documents"),
        ("Escritorio", home / "Escritorio"),
        ("Desktop", home / "Desktop")
    ]
    seen = set()
    for name, p in candidates:
        if p.exists() and str(p) not in seen:
            bookmarks.append({"name": name, "path": str(p), "icon": "folder"})
            seen.add(str(p))

    return {"bookmarks": bookmarks}

@router.get("/browse")
def browse(path: Optional[str] = Query(None)):
    target_path = Path(path).resolve() if path else Path.cwd().resolve()

    if not target_path.exists():
        target_path = Path.home()

    if not target_path.is_dir():
        target_path = target_path.parent

    entries = []
    try:
        with os.scandir(target_path) as it:
            for entry in it:
                try:
                    # Ocultar archivos o carpetas ocultas por defecto excepto si se requiere
                    if entry.name.startswith("."):
                        continue
                    
                    p = Path(entry.path)
                    is_dir = entry.is_dir(follow_symlinks=True)
                    ext = p.suffix.lower() if not is_dir else ""
                    is_video = ext in VIDEO_EXTENSIONS
                    is_sub = ext in SUBTITLE_EXTENSIONS

                    # Incluir carpetas, videos y subtitulos
                    if is_dir or is_video or is_sub:
                        size = entry.stat().st_size if not is_dir else 0
                        entries.append({
                            "name": entry.name,
                            "path": str(p.resolve()),
                            "is_dir": is_dir,
                            "ext": ext,
                            "is_video": is_video,
                            "is_subtitle": is_sub,
                            "size": size
                        })
                except (PermissionError, FileNotFoundError):
                    continue
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permiso denegado para explorar este directorio.")

    # Ordenar: primero carpetas (alfabéticamente), luego archivos
    entries.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))

    # Información del directorio padre para botón 'Subir nivel'
    parent_path = str(target_path.parent) if target_path.parent != target_path else None

    return {
        "current_path": str(target_path),
        "parent_path": parent_path,
        "entries": entries
    }

@router.get("/inspect")
def inspect_file(path: str = Query(...)):
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="El archivo no existe.")

    ext = p.suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        try:
            info = AudioExtractor.get_media_info(str(p))
            return {"type": "video", "info": info}
        except Exception as e:
            return {"type": "video", "error": str(e)}

    return {"type": "file", "name": p.name, "size": p.stat().st_size}

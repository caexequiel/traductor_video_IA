import os
import mimetypes
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api/media", tags=["media"])

def range_stream(file_path: str, start: int, end: int, chunk_size: int = 1024 * 1024):
    with open(file_path, "rb") as f:
        f.seek(start)
        bytes_to_read = (end - start) + 1
        while bytes_to_read > 0:
            read_size = min(chunk_size, bytes_to_read)
            data = f.read(read_size)
            if not data:
                break
            bytes_to_read -= len(data)
            yield data

@router.api_route("/stream", methods=["GET", "HEAD"])
def stream_media(path: str, request: Request):
    """Permite reproducir videos locales en el navegador mediante streaming por rangos HTTP 206."""
    file_path = Path(path).resolve()
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Archivo no encontrado.")

    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")

    content_type, _ = mimetypes.guess_type(str(file_path))
    content_type = content_type or "application/octet-stream"

    if request.method == "HEAD":
        from fastapi import Response
        return Response(
            status_code=200,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
                "Content-Type": content_type
            }
        )

    if range_header:
        # Formato: "bytes=0-1000"
        bytes_range = range_header.strip().split("=")[-1]
        start_str, end_str = bytes_range.split("-")
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1
        end = min(end, file_size - 1)
        content_length = (end - start) + 1

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
            "Content-Type": content_type,
        }
        return StreamingResponse(
            range_stream(str(file_path), start, end),
            status_code=status.HTTP_206_PARTIAL_CONTENT,
            headers=headers
        )

    # Respuesta completa normal
    return StreamingResponse(
        open(str(file_path), "rb"),
        media_type=content_type,
        headers={"Content-Length": str(file_size), "Accept-Ranges": "bytes"}
    )

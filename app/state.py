import asyncio
from typing import Dict, Any, List, Set
from fastapi import WebSocket

class AppState:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.active_websockets: Set[WebSocket] = set()

    async def register_ws(self, ws: WebSocket):
        await ws.accept()
        self.active_websockets.add(ws)

    def unregister_ws(self, ws: WebSocket):
        self.active_websockets.discard(ws)

    async def broadcast(self, message: Dict[str, Any]):
        dead = []
        for ws in self.active_websockets:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.unregister_ws(ws)

app_state = AppState()

import asyncio
import base64
import http.server
import json
import logging
from pathlib import Path
import socketserver
import ssl
import threading
from typing import Any, Awaitable, Callable, Dict, Optional, Set

import websockets
from websockets.server import WebSocketServerProtocol

logger = logging.getLogger("athome_hri.web")

AudioCallback = Callable[[bytes], Awaitable[Any]]
TextCallback = Callable[[str], Awaitable[Any]]


class _ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


class WebServer:
    """HTTP and WebSocket web server module for smartphone/client interface."""

    def __init__(
        self,
        http_port: int = 8000,
        ws_port: int = 8765,
        on_audio_received: Optional[AudioCallback] = None,
        on_text_received: Optional[TextCallback] = None,
    ) -> None:
        self.http_port = http_port
        self.ws_port = ws_port
        self.on_audio_received = on_audio_received
        self.on_text_received = on_text_received
        self.active_ws: Set[WebSocketServerProtocol] = set()
        self._httpd: Optional[socketserver.TCPServer] = None

    def start_http(self) -> None:
        """Run local HTTP server in a background daemon thread."""
        thread = threading.Thread(
            target=self._run_http, name="HriHttpServer", daemon=True
        )
        thread.start()

    def stop_http(self) -> None:
        """Stop HTTP server if running."""
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
            except Exception:
                pass

    def _run_http(self) -> None:
        web_dir = self._find_web_dir()

        class QuietHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, directory=web_dir, **kwargs)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        try:
            with _ReusableTCPServer(("0.0.0.0", self.http_port), QuietHandler) as httpd:
                self._httpd = httpd
                cert_file = Path(web_dir) / "cert.pem"
                key_file = Path(web_dir) / "key.pem"
                proto = "http"

                if cert_file.is_file() and key_file.is_file():
                    try:
                        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                        ctx.load_cert_chain(str(cert_file), str(key_file))
                        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
                        proto = "https"
                    except Exception as ssl_exc:
                        logger.warning("SSL fallback to HTTP: %s", ssl_exc)

                logger.info(
                    "Web UI active at %s://0.0.0.0:%d (Dir: %s)",
                    proto,
                    self.http_port,
                    web_dir,
                )
                httpd.serve_forever()
        except Exception as exc:
            logger.error(
                "HTTP port %d unavailable or failed: %s", self.http_port, exc
            )

    @staticmethod
    def _find_web_dir() -> str:
        """Search Web UI static directory (index.html) via ament_index or local path."""
        candidates: list[Path] = []
        try:
            from ament_index_python.packages import get_package_share_directory

            share_dir = Path(get_package_share_directory("athome_hri"))
            candidates.append(share_dir / "src")
        except Exception:
            pass

        module_dir = Path(__file__).resolve().parent
        candidates.extend(
            [
                module_dir.parent / "src",
                module_dir / "src",
                Path.cwd() / "src" / "athome_hri" / "src",
            ]
        )

        for directory in candidates:
            if (directory / "index.html").is_file():
                return str(directory)
        return str(module_dir)

    async def broadcast_ui(self, payload: Dict[str, Any]) -> None:
        """Send JSON message to all connected WebSocket clients."""
        if not self.active_ws:
            return
        msg = json.dumps(payload)
        clients = [ws for ws in tuple(self.active_ws) if ws.open]
        if clients:
            await asyncio.gather(
                *(ws.send(msg) for ws in clients),
                return_exceptions=True,
            )

    async def ws_handler(self, websocket: WebSocketServerProtocol) -> None:
        """Handle client WebSocket connection lifecycle and incoming messages."""
        self.active_ws.add(websocket)
        logger.info("WebSocket client connected from %s", websocket.remote_address)

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    if self.on_audio_received:
                        await self.on_audio_received(message)
                    continue

                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    logger.warning("Invalid WebSocket JSON payload.")
                    continue

                msg_type = data.get("type", "")
                if msg_type == "AUDIO_BASE64":
                    audio_b64 = data.get("audio", "")
                    if audio_b64 and self.on_audio_received:
                        audio_bytes = base64.b64decode(audio_b64)
                        await self.on_audio_received(audio_bytes)
                elif msg_type == "SIM_TEXT":
                    text = data.get("text", "")
                    if text and self.on_text_received:
                        await self.on_text_received(text)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.active_ws.discard(websocket)
            logger.info("WebSocket client disconnected.")

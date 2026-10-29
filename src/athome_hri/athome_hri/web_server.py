import asyncio
import base64
import http.server
import json
import os
import socketserver
import ssl
import threading
from typing import Callable, Set
import websockets


class WebServer:
    """HTTP and WebSocket web server module for smartphone/client interface."""

    def __init__(
        self,
        http_port: int = 8000,
        ws_port: int = 8765,
        on_audio_received: Callable[[bytes], any] = None,
        on_text_received: Callable[[str], any] = None,
    ):
        self.http_port = http_port
        self.ws_port = ws_port
        self.on_audio_received = on_audio_received
        self.on_text_received = on_text_received
        self.active_ws: Set[websockets.WebSocketServerProtocol] = set()

    def start_http(self):
        """Run local HTTP server in a background thread."""
        thread = threading.Thread(target=self._run_http, daemon=True)
        thread.start()

    def _run_http(self):
        web_dir = self._find_web_dir()

        class QuietHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=web_dir, **kwargs)

            def log_message(self, format, *args):
                pass

        socketserver.TCPServer.allow_reuse_address = True
        try:
            with socketserver.TCPServer(("0.0.0.0", self.http_port), QuietHandler) as httpd:
                cert_file = os.path.join(web_dir, "cert.pem")
                key_file = os.path.join(web_dir, "key.pem")
                proto = "http"

                if os.path.exists(cert_file) and os.path.exists(key_file):
                    try:
                        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                        ctx.load_cert_chain(cert_file, key_file)
                        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
                        proto = "https"
                    except Exception as e:
                        print(f"[HTTP] SSL fallback to HTTP: {e}")

                print(f"[HTTP] Web UI active at {proto}://0.0.0.0:{self.http_port} (Dir: {web_dir})")
                httpd.serve_forever()
        except Exception as e:
            print(f"[HTTP] Port {self.http_port} unavailable or failed: {e}")

    def _find_web_dir(self) -> str:
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "src"),
            os.path.join(os.path.dirname(__file__), "src"),
            os.path.join(os.getcwd(), "src", "athome_hri", "src"),
        ]
        try:
            from ament_index_python.packages import get_package_share_directory
            candidates.append(os.path.join(get_package_share_directory("athome_hri"), "src"))
        except Exception:
            pass

        for d in candidates:
            if os.path.exists(os.path.join(d, "index.html")):
                return d
        return os.path.dirname(__file__)

    async def broadcast_ui(self, payload: dict):
        """Send JSON message to all connected clients."""
        if not self.active_ws:
            return
        msg = json.dumps(payload)
        await asyncio.gather(
            *[ws.send(msg) for ws in self.active_ws if ws.open],
            return_exceptions=True,
        )

    async def ws_handler(self, websocket):
        """Handle client WebSocket connection lifecycle."""
        self.active_ws.add(websocket)
        print(f"[WS] Client connected from {websocket.remote_address}")

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    if self.on_audio_received:
                        await self.on_audio_received(message)
                else:
                    data = json.loads(message)
                    msg_type = data.get("type", "")

                    if msg_type == "AUDIO_BASE64":
                        audio_bytes = base64.b64decode(data.get("audio", ""))
                        if self.on_audio_received:
                            await self.on_audio_received(audio_bytes)
                    elif msg_type == "SIM_TEXT":
                        text = data.get("text", "")
                        if self.on_text_received:
                            await self.on_text_received(text)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.active_ws.discard(websocket)
            print("[WS] Klien terputus.")


#!/usr/bin/env python3
"""
Barelang Inari - Standalone Robot Face Node
Serves the 7-inch LCD web interface and connects ROS 2 expression topics to WebSocket.
"""

import asyncio
import http.server
import json
import logging
from pathlib import Path
import socketserver
import subprocess
import threading
from typing import Set

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import websockets
from websockets.server import WebSocketServerProtocol

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [athome_face]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("athome_face")


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


class RobotFaceNode(Node):
    """ROS 2 Node that bridges robot emotions and states to the 7-inch HDMI face interface."""

    def __init__(self, async_loop: asyncio.AbstractEventLoop) -> None:
        super().__init__("robot_face_node")
        self.loop = async_loop
        self.active_clients: Set[WebSocketServerProtocol] = set()

        # Parameter Declarations
        self.declare_parameter("http_port", 8080)
        self.declare_parameter("ws_port", 8766)
        self.declare_parameter("launch_browser", False)
        self.declare_parameter("kiosk_mode", True)

        self.http_port: int = int(self.get_parameter("http_port").value)
        self.ws_port: int = int(self.get_parameter("ws_port").value)
        self.launch_browser: bool = bool(self.get_parameter("launch_browser").value)
        self.kiosk_mode: bool = bool(self.get_parameter("kiosk_mode").value)

        # Web Directory Discovery
        self.web_dir: Path = self._find_web_dir()
        logger.info("Serving Web Face GUI from: %s", self.web_dir)

        # ROS 2 Subscriptions
        # 1. Main Direct Expression Topic (String)
        self.create_subscription(
            String, "/robot/expression", self._on_expression_received, 10
        )

        # 2. Optional HRI Event Topic
        self.create_subscription(
            String, "/hri/event", self._on_hri_event_received, 10
        )

        # 3. Optional TTS Subtitle Topic
        self.create_subscription(
            String, "/hri/tts_say", self._on_tts_say_received, 10
        )

        logger.info("RobotFaceNode ready. Listening on /robot/expression")

    def _find_web_dir(self) -> Path:
        """Find the static web directory containing index.html."""
        # Check source tree relative to this file
        current_file = Path(__file__).resolve()
        source_web = current_file.parent.parent / "web"
        if (source_web / "index.html").is_file():
            return source_web

        # Check install space
        try:
            from ament_index_python.packages import get_package_share_directory
            share_web = Path(get_package_share_directory("athome_face")) / "web"
            if (share_web / "index.html").is_file():
                return share_web
        except Exception:
            pass

        return source_web

    def start_http_server(self) -> None:
        """Start simple HTTP server in a background thread."""
        web_path = str(self.web_dir)

        class QuietHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=web_path, **kwargs)

            def log_message(self, format, *args):
                pass  # Suppress noisy HTTP request logs

        def _serve():
            try:
                with ReusableTCPServer(("0.0.0.0", self.http_port), QuietHandler) as httpd:
                    logger.info("Face Web UI active at http://0.0.0.0:%d", self.http_port)
                    httpd.serve_forever()
            except Exception as exc:
                logger.error("HTTP Server error on port %d: %s", self.http_port, exc)

        t = threading.Thread(target=_serve, name="FaceHttpServer", daemon=True)
        t.start()

    async def ws_handler(self, websocket: WebSocketServerProtocol) -> None:
        """Handle incoming WebSocket client connections."""
        self.active_clients.add(websocket)
        logger.info(
            "Face UI client connected (%s). Total clients: %d",
            websocket.remote_address,
            len(self.active_clients),
        )
        try:
            async for raw_msg in websocket:
                # Can process incoming client events here if needed
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.active_clients.discard(websocket)
            logger.info("Face UI client disconnected. Remaining: %d", len(self.active_clients))

    def broadcast(self, payload: dict) -> None:
        """Broadcast payload to all active WebSocket clients thread-safely."""
        if not self.active_clients:
            return

        msg_str = json.dumps(payload)

        async def _send():
            dead_clients = set()
            for ws in self.active_clients:
                try:
                    await ws.send(msg_str)
                except Exception:
                    dead_clients.add(ws)
            self.active_clients.difference_update(dead_clients)

        self.loop.call_soon_threadsafe(lambda: asyncio.create_task(_send()))

    def _on_expression_received(self, msg: String) -> None:
        expr = msg.data.strip().upper()
        logger.info("[Expression] Received: %s", expr)
        self.broadcast({"type": "EXPRESSION", "expression": expr})

    def _on_hri_event_received(self, msg: String) -> None:
        evt = msg.data.strip().upper()
        mapping = {
            "WAKEWORD_DETECTED": "GREETING",
            "FOLLOW_COMMAND": "FOLLOWING",
            "STOP_COMMAND": "IDLE",
            "ARRIVED_COMMAND": "CURIOUS",
            "OPERATOR_LOST": "LOST",
            "OPERATOR_FOUND": "GREETING",
        }
        if evt in mapping:
            self.broadcast({"type": "EXPRESSION", "expression": mapping[evt]})

    def _on_tts_say_received(self, msg: String) -> None:
        text = msg.data.strip()
        if text:
            self.broadcast({"type": "TTS_PLAY", "text": text})

    def launch_kiosk_browser(self) -> None:
        """Launch Chromium browser in full-screen kiosk mode for 7-inch HDMI LCD."""
        if not self.launch_browser:
            return

        url = f"http://localhost:{self.http_port}"
        cmd = [
            "chromium",
            "--kiosk" if self.kiosk_mode else "--app=" + url,
            "--noerrdialogs",
            "--disable-infobars",
            "--check-for-update-interval=31536000",
            url,
        ]
        try:
            subprocess.Popen(cmd)
            logger.info("Launched Kiosk Browser pointing to %s", url)
        except FileNotFoundError:
            try:
                # Try google-chrome if chromium not found
                cmd[0] = "google-chrome"
                subprocess.Popen(cmd)
                logger.info("Launched Chrome Kiosk pointing to %s", url)
            except Exception as e:
                logger.warning("Could not automatically launch browser: %s", e)


async def main_async() -> None:
    rclpy.init()
    loop = asyncio.get_running_loop()
    node = RobotFaceNode(loop)

    # Start ROS 2 executor thread
    ros_thread = threading.Thread(
        target=rclpy.spin, args=(node,), name="FaceRosSpin", daemon=True
    )
    ros_thread.start()

    # Start Web Servers
    node.start_http_server()
    if node.launch_browser:
        node.launch_kiosk_browser()

    logger.info("=================================================================")
    logger.info("Barelang Inari Face Server Active!")
    logger.info("Web UI Local URL : http://localhost:%d", node.http_port)
    logger.info("WebSocket Bridge : ws://localhost:%d", node.ws_port)
    logger.info("CLI Expression   : ros2 topic pub /robot/expression std_msgs/msg/String \"{data: 'GREETING'}\"")
    logger.info("=================================================================")

    # Run WebSocket Server indefinitely
    async with websockets.serve(node.ws_handler, "0.0.0.0", node.ws_port):
        await asyncio.Future()


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()


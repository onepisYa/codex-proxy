import json
import logging
import socket
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Dict, Any, Type, Optional, cast

try:
    import orjson
except ImportError:
    import json as orjson

from .config import config
from .providers.base import BaseProvider
from .providers.gemini import GeminiProvider
from .providers.zai import ZAIProvider
from .normalizer import RequestNormalizer

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Registry for AI model providers."""

    _providers: Dict[str, BaseProvider] = {}

    @classmethod
    def register(cls, prefix: str, provider: BaseProvider):
        cls._providers[prefix] = provider

    @classmethod
    def get_provider(cls, model_name: str) -> BaseProvider:
        for prefix, provider in cls._providers.items():
            if model_name.startswith(prefix):
                return provider
        # Default to ZAI if no match
        return cast(BaseProvider, cls._providers.get("zai"))


# Initialize registry
ProviderRegistry.register("gemini", GeminiProvider())
ProviderRegistry.register("zai", ZAIProvider())
# Default / Catch-all
ProviderRegistry.register("glm", ZAIProvider())


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server with performance optimizations."""

    daemon_threads = True
    allow_reuse_address = True

    def server_bind(self):
        super().server_bind()
        try:
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception as e:
            logger.warning(f"Failed to set TCP_NODELAY: {e}")


class ProxyRequestHandler(BaseHTTPRequestHandler):
    """Handles incoming Codex requests and routes them to appropriate providers."""

    def do_POST(self):
        try:
            self._handle_post()
        except Exception as e:
            logger.error(f"Handler Error: {e}", exc_info=True)
            try:
                self.send_error(500, str(e))
            except Exception:
                pass

    def _handle_post(self):
        logger.info(f"POST {self.path}")

        if self.path not in (
            "/v1/responses",
            "/responses",
            "/v1/responses/compact",
            "/responses/compact",
        ):
            self.send_error(404, f"Endpoint {self.path} not supported.")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self.send_error(400, "Empty body")
            return

        body = self.rfile.read(content_length)
        if config.debug_mode:
            logger.debug(f"RAW REQUEST: {body.decode('utf-8', errors='replace')}")

        try:
            data = orjson.loads(body)
        except Exception:
            data = json.loads(body)

        # Attach context headers
        data["_headers"] = {
            "session_id": self.headers.get("session_id"),
            "x-openai-subagent": self.headers.get("x-openai-subagent"),
            "x-codex-turn-state": self.headers.get("x-codex-turn-state"),
            "x-codex-personality": self.headers.get("x-codex-personality"),
        }

        # Normalize request if it's a standard responses call
        is_compact = "/compact" in self.path
        if not is_compact:
            data = RequestNormalizer.normalize(data)
            data["_is_responses_api"] = True

        model = data.get("model", "")
        provider = ProviderRegistry.get_provider(model)

        if is_compact:
            provider.handle_compact(data, self)
        else:
            provider.handle_request(data, self)

        self.close_connection = True

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def log_message(self, format, *args):
        """Override to suppress default logging to stdout."""
        pass


def run_server():
    server_address = (config.host, config.port)
    httpd = ThreadedHTTPServer(server_address, ProxyRequestHandler)
    logger.info(f"Codex Proxy listening on {config.host}:{config.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        httpd.server_close()


if __name__ == "__main__":
    run_server()

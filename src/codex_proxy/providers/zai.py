import time
import logging
import requests
import json
from typing import Dict, Any, List, Optional
from .base import BaseProvider
from ..utils import create_session
from ..config import config
from .zai_stream import stream_responses_loop

try:
    import orjson

    def json_dumps(data: Any) -> bytes:
        return orjson.dumps(data)

    def json_loads(data: bytes | str) -> Any:
        return orjson.loads(data)
except ImportError:

    def json_dumps(data: Any) -> bytes:
        return json.dumps(data).encode("utf-8")

    def json_loads(data: bytes | str) -> Any:
        return json.loads(data)


logger = logging.getLogger(__name__)


class ZAIProvider(BaseProvider):
    """Provider for Z.AI GLM models."""

    def __init__(self):
        self.session = create_session()

    def handle_request(self, data: Dict[str, Any], handler: Any) -> None:
        """Entry point for Z.AI requests."""
        payload = self._prepare_payload(data)
        self._transform_payload(payload)
        self._execute_request(payload, data, handler)

    def _prepare_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a clean payload with only supported parameters."""
        payload = {
            "model": data.get("model"),
            "messages": data.get("messages", []),
            "stream": data.get("stream", False),
        }
        if "tools" in data:
            payload["tools"] = data["tools"]
        if "tool_choice" in data:
            payload["tool_choice"] = data["tool_choice"]
        if "temperature" in data:
            payload["temperature"] = data["temperature"]
        if "top_p" in data:
            payload["top_p"] = data["top_p"]
        return payload

    def _transform_payload(self, payload: Dict[str, Any]) -> None:
        """Apply Z.AI specific transformations."""
        # 1. Fix Roles
        for m in payload.get("messages", []):
            if m.get("role") == "developer":
                m["role"] = "system"

        # 2. Filter and Clean Tools
        if "tools" in payload:
            payload["tools"] = [
                tool for tool in payload["tools"] if tool.get("type") == "function"
            ]
            for tool in payload["tools"]:
                if "strict" in tool:
                    del tool["strict"]

    def _execute_request(
        self, payload: Dict[str, Any], original_data: Dict[str, Any], handler: Any
    ) -> None:
        """Perform the actual API call and handle response."""
        auth_header = handler.headers.get("Authorization", "")
        stream = payload.get("stream", False)

        try:
            with self.session.post(
                config.z_ai_url,
                json=payload,
                headers={"Authorization": auth_header},
                stream=stream,
                timeout=(10, 600),
            ) as resp:
                if stream:
                    self._handle_stream_response(resp, payload, handler)
                else:
                    self._handle_sync_response(resp, original_data, handler)
        except Exception as e:
            logger.error(f"ZAI Request failed: {e}")
            raise e

    def _handle_stream_response(
        self, resp: requests.Response, payload: Dict[str, Any], handler: Any
    ) -> None:
        """Manage streaming response forwarding."""
        handler.send_response(resp.status_code)
        handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
        handler.send_header("Connection", "keep-alive")
        handler.end_headers()

        created_ts = int(time.time())
        stream_responses_loop(resp, handler, payload["model"], created_ts, payload)

    def _handle_sync_response(
        self, resp: requests.Response, original_data: Dict[str, Any], handler: Any
    ) -> None:
        """Manage synchronous response forwarding."""
        handler.send_response(resp.status_code)
        handler.send_header("Content-Type", "application/json")
        handler.end_headers()

        if original_data.get("_is_responses_api") and resp.status_code == 200:
            try:
                self._write_mapped_response(resp, handler)
                return
            except Exception as e:
                logger.warning(f"Failed to map ZAI response: {e}")

        handler.wfile.write(resp.content)

    def _write_mapped_response(self, resp: requests.Response, handler: Any) -> None:
        """Map ZAI standard chat response to Codex Responses API format."""
        z_data = resp.json()
        choice = z_data["choices"][0]
        message = choice["message"]
        usage = z_data.get("usage", {})

        resp_obj = {
            "id": f"zai_{z_data.get('id')}",
            "object": "response",
            "created": z_data.get("created"),
            "model": z_data.get("model"),
            "status": "completed",
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": message.get("content", "")}],
                }
            ],
        }
        handler.wfile.write(json_dumps(resp_obj))

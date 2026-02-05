import time
import logging
import requests
import json
from typing import Dict, Any, Optional, List

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


class ZAIStreamHandler:
    """Manages the mapping of ZAI stream to Codex Responses API events."""

    def __init__(
        self,
        handler: Any,
        model: str,
        created_ts: int,
        request_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.handler = handler
        self.model = model
        self.created_ts = created_ts
        self.request_metadata = request_metadata or {}
        self.resp_id = f"resp_{created_ts}"
        self.seq_num = 0

        self.full_content = ""
        self.message: Optional[Dict[str, Any]] = None
        self.item_id: Optional[str] = None
        self.idx: int = 0

    def _send_event(self, evt_type: str, data: Dict[str, Any]) -> None:
        """Serialize and send a single SSE event."""
        self.seq_num += 1
        event = {
            "id": f"evt_{int(time.time() * 1000)}_{self.seq_num}",
            "object": "response.event",
            "type": evt_type,
            "created_at": int(time.time()),
            "sequence_number": self.seq_num,
            **data,
        }
        payload = (
            b"event: " + evt_type.encode() + b"\ndata: " + json_dumps(event) + b"\n\n"
        )
        self.handler.wfile.write(payload)
        self.handler.wfile.flush()

    def process_stream(self, resp: requests.Response) -> None:
        """Main loop for processing the ZAI stream."""
        # Rich response object for created event
        response_obj = {
            "id": self.resp_id,
            "object": "response",
            "created_at": self.created_ts,
            "model": self.model,
            "status": "in_progress",
            "temperature": self.request_metadata.get("temperature", 1.0),
            "top_p": self.request_metadata.get("top_p", 1.0),
            "tool_choice": self.request_metadata.get("tool_choice", "auto"),
            "tools": self.request_metadata.get("tools", []),
            "parallel_tool_calls": True,
            "store": self.request_metadata.get("store", True),
            "metadata": self.request_metadata.get("metadata", {}),
            "output": [],
        }

        self._send_event("response.created", {"response": response_obj})

        try:
            for line in resp.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue

                if line == b"data: [DONE]":
                    break

                self._handle_line(line[6:])
        except Exception as e:
            logger.error(f"Error in ZAI stream processing: {e}")
        finally:
            self._finalize(response_obj)

    def _handle_line(self, json_data: bytes) -> None:
        """Parse a single ZAI stream line and emit corresponding Codex events."""
        try:
            data = json_loads(json_data)
            choice = data.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            content = delta.get("content", "")

            if content:
                self.full_content += content

                if self.message is None:
                    self._init_message()

                self._send_event(
                    "response.output_text.delta",
                    {
                        "response_id": self.resp_id,
                        "item_id": self.item_id,
                        "output_index": self.idx,
                        "content_index": 0,
                        "delta": content,
                    },
                )
                # Keep local state updated
                if self.message:
                    self.message["content"][0]["text"] = self.full_content
        except Exception as e:
            logger.debug(f"Failed to parse ZAI stream line: {e}")

    def _init_message(self) -> None:
        """Initialize the assistant message item."""
        self.item_id = f"msg_{int(time.time() * 1000)}_{self.idx}"
        self.message = {
            "id": self.item_id,
            "type": "message",
            "role": "assistant",
            "status": "in_progress",
            "content": [{"type": "output_text", "text": ""}],
        }
        self._send_event(
            "response.output_item.added",
            {
                "response_id": self.resp_id,
                "output_index": self.idx,
                "item": self.message,
            },
        )

    def _finalize(self, response_obj: Dict[str, Any]) -> None:
        """Emit completion events and close the response."""
        if self.message is not None:
            self.message["status"] = "completed"
            self._send_event(
                "response.output_item.done",
                {
                    "response_id": self.resp_id,
                    "output_index": self.idx,
                    "item": self.message,
                },
            )

        # Update response object for completion
        response_obj.update(
            {
                "status": "completed",
                "completed_at": int(time.time()),
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "output": [self.message] if self.message is not None else [],
            }
        )

        self._send_event("response.completed", {"response": response_obj})


def stream_responses_loop(
    resp: requests.Response,
    handler: Any,
    model: str,
    created_ts: int,
    request_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Wrapper function matching the existing interface."""
    stream_handler = ZAIStreamHandler(handler, model, created_ts, request_metadata)
    stream_handler.process_stream(resp)

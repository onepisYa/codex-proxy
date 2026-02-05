# codex-proxy

The `codex-proxy` is a high-performance bridge for the Codex CLI. It maps the specialized "Responses API" to AI providers like Z.AI and Gemini, ensuring 1:1 parity with native GPT models. It handles the low-level translation so your agentic workflows just work.

## What it actually does

- **Deep Parity with Responses API**: It implements the full lifecycle of OpenAI's newest API. The proxy emits rich SSE events (`response.created`, `response.output_item.added`, `response.completed`) containing all the metadata Codex expects—temperature, tool definitions, and reasoning blocks.
- **Role Correction**: Automatically maps the `developer` role to `system`. This fixes immediate crashes on providers like Z.AI that don't support the newer role naming yet.
- **Advanced Gemini Support**: Unlocks Gemini 2.0 and 3.0 features. It supports "thinking" blocks, strict JSON schemas, and automatically switches to a fallback model (like Flash Lite) if you hit a rate limit.
- **History Auto-Compaction**: Background history management. When your conversation gets too long, it uses a fast Flash model to summarize the context, keeping your token usage efficient without losing the thread.
- **Fast and Robust**: A multi-threaded Python server using `orjson` for minimal overhead and low latency.

## Supported Providers

- **Gemini**: Full integration with Google's internal APIs. It manages the OAuth2 flow, discovers your Project ID, and maps reasoning effort to the correct thinking budget.
- **Z.AI (GLM)**: Robust support for GLM models. Includes spec-compliant streaming, tool call sanitization, and automatic pruning of unsupported parameters.

## The Control Script

Management is handled through a single, modular tool: `scripts/control.sh`.

```bash
# Start or stop the container
./scripts/control.sh start
./scripts/control.sh stop

# Monitor logs (standard Unix format)
./scripts/control.sh logs

# Run a quick test against a specific Codex profile
./scripts/control.sh run -p glm -- "Why is the sky blue?"

# Run the full integration test suite
./scripts/control.sh test
```

## Setup

The proxy runs in Docker and expects Gemini credentials at `~/.gemini/oauth_creds.json`.

1. **Configuration**:
   Manage settings via environment variables or `~/.gemini/proxy_config.json`:
   - `Z_AI_API_KEY`: Your Z.AI key.
   - `GEMINI_CLIENT_ID` / `SECRET`: Credentials for Gemini internal APIs.
   - `PORT`: Defaults to `8765`.

2. **Connecting Codex**:
   Update your `~/.codex/config.toml` to use the proxy:

   ```toml
   [model_providers.z_ai]
   name = "ZAI Proxy"
   base_url = "http://localhost:8765"
   env_key = "Z_AI_API_KEY"
   wire_api = "responses" # Crucial: Must be responses API

   [profiles.glm]
   model = "glm-4.6"
   model_provider = "z_ai"
   ```

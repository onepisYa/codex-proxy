# codex-proxy

A high-performance bridge that maps the OpenAI Responses API used by Codex to other providers like Google Gemini and Z.AI (GLM). It handles the low-level translation, role mapping, and stream formatting so you can use these models as drop-in replacements for GPT-4/5 in your agentic workflows.

## Features

- **Responses API Parity**: Implements the full lifecycle of the Responses API, including metadata-rich events (`response.created`, `response.output_item.added`, etc.).
- **Provider Compatibility**: Fixes breaking issues, such as mapping the `developer` role back to `system` and sanitizing tool parameters for older APIs.
- **Advanced Gemini Integration**: Supports Gemini "thinking" blocks, JSON schemas, and automatic fallback to Flash models under rate limits.
- **Unified Authentication**: Supports every `gemini-cli` authentication type (OAuth, AI Studio API Keys, Vertex AI, ADC).
- **Context Management**: Dedicated compaction endpoint using fast Flash models to summarize history.

## Getting Started

Manage the proxy using the `scripts/control.sh` script.

```bash
# Start the proxy in Docker
./scripts/control.sh start

# Monitor logs (Standard Unix format)
./scripts/control.sh logs

# Run a test command through Codex
./scripts/control.sh run -p glm -- "Why is the sky blue?"

# Stop the proxy
./scripts/control.sh stop
```

## Configuration

The proxy loads configuration from environment variables or a JSON file at `~/.config/codex-proxy/config.json`.

| Variable | Config Key | Description |
| :--- | :--- | :--- |
| `PORT` | `port` | Server listen port (default: 8765). |
| `GEMINI_API_KEY` | `gemini_api_key` | Google AI Studio API Key. |
| `Z_AI_API_KEY` | `z_ai_api_key` | Z.AI (GLM) API Key. |
| `GEMINI_MODELS` | `gemini_models` | Comma-separated list of supported Gemini models. |
| `GOOGLE_CLOUD_PROJECT` | - | Manual override for Google Cloud Project ID. |

### Configuration Example (`~/.config/codex-proxy/config.json`)

```json
{
  "port": 8765,
  "gemini_api_key": "your-ai-studio-key",
  "z_ai_api_key": "your-z-ai-key",
  "gemini_models": ["gemini-3-flash-preview", "gemini-3-pro-preview"]
}
```

## Authentication (Gemini)

The proxy is designed to be zero-config. It automatically supports all [gemini-cli authentication methods](https://github.com/google-gemini/gemini-cli/blob/main/docs/get-started/authentication.md):

1.  **Standard Google Login**: Run `gemini login` on your host. The proxy automatically discovers these credentials and handles the OAuth2 refresh flow using the official Gemini CLI client identity.
2.  **AI Studio (API Key)**: Set `GEMINI_API_KEY`. The proxy will automatically switch to the Google AI public API.
3.  **Service Accounts**: Set `GOOGLE_APPLICATION_CREDENTIALS` to your JSON key path.
4.  **Application Default Credentials (ADC)**: Works automatically in GCP environments (GCE, Cloud Shell) or via `gcloud auth application-default login`.
5.  **Vertex AI**: Works natively if you provide a `GOOGLE_CLOUD_PROJECT` and have authenticated via ADC or Service Account.

## Connecting to Codex

Update your `~/.codex/config.toml` to point to the proxy.

### Z.AI (GLM)
```toml
[model_providers.z_ai]
name = "ZAI Proxy"
base_url = "http://localhost:8765"
env_key = "Z_AI_API_KEY"
wire_api = "responses"

[profiles.glm]
model = "glm-4.6"
model_provider = "z_ai"
```

### Gemini
```toml
[model_providers.gemini_proxy]
name = "Gemini Proxy"
base_url = "http://localhost:8765"
wire_api = "responses"

[profiles.gemini]
model = "gemini-3-pro-preview"
model_provider = "gemini_proxy"
```

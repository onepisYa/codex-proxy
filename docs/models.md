# Models endpoint and discovery

codex-proxy can periodically discover models from upstream providers and expose them via the OpenAI-compatible models API.

## Discovery loop

When `model_discovery.enabled` is true, codex-proxy periodically calls provider model listing endpoints when implemented.
When a provider does not support discovery, codex-proxy falls back to `providers.<name>.models` in config.

## `GET /v1/models`

codex-proxy serves an OpenAI-style models list on:

- `GET /v1/models`
- `GET /models`

The returned list is configurable:

- `models_endpoint.source = served` returns `models.served` only.
- `models_endpoint.source = discovered` returns discovered upstream model ids.
- `models_endpoint.source = both` merges both sources.

## Metadata and pricing

Some providers do not expose context window sizes or pricing via APIs. For internal use, codex-proxy supports
configuration of per-provider/per-model metadata:

```json
{
  "model_metadata": {
    "tabcode": {
      "gpt-4.1": {
        "context_window": 128000,
        "max_output_tokens": 16384,
        "pricing": {
          "input_per_mtoken": 10.0,
          "output_per_mtoken": 30.0
        }
      }
    }
  }
}
```


# Sessions and sticky routing

codex-proxy uses sticky routing to keep repeated requests on the same upstream account/provider in order to maximize
cache reuse and reduce tail latency.

By default, the sticky key is derived from request content. This is fragile when the proxy rewrites inputs (for example
auto-compaction) because the message hash changes.

This document describes the provider-independent **session key** mechanism used to stabilize sticky routing across
rewrites.

## Session key

codex-proxy accepts an opaque session identifier and derives a stable routing key from it.

- Request header: `x-codex-proxy-session: <opaque>`
- Request metadata (OpenAI-compatible): `metadata.codex_proxy_session = "<opaque>"`

The proxy echoes the session key back in the same response header so stateless clients can persist it.

### Precedence

When multiple sources are present, codex-proxy resolves the session key in the following order:

1. `x-codex-proxy-session` header
2. `metadata.codex_proxy_session`
3. `previous_response_id` (proxy-local mapping, best effort)
4. Generate a new session key and return it in the response header

## `previous_response_id` support (proxy-local)

OpenAI Responses supports `previous_response_id` for linking turns. Upstream providers may or may not implement it.

codex-proxy supports `previous_response_id` as a **proxy-local hint**:

- When the proxy can observe the `response.id` of a completed response, it records:
  - `response.id -> session_key` in memory (TTL controlled by `session.response_id_ttl_seconds`)
- When a client later sends `previous_response_id`, codex-proxy restores the same session key if the mapping exists.

### Limitations

- For streaming responses (`text/event-stream`), codex-proxy does not currently parse the stream to extract `response.id`
  and therefore cannot reliably populate the mapping.
- For large JSON responses, codex-proxy may skip parsing to avoid buffering large bodies.

If your client needs robust session persistence in all cases, use `x-codex-proxy-session`.

## Config

Relevant config section:

```json
{
  "session": {
    "header_name": "x-codex-proxy-session",
    "metadata_key": "codex_proxy_session",
    "response_id_ttl_seconds": 86400
  }
}
```


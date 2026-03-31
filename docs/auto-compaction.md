# Auto-compaction

Some clients use an automatic compaction threshold to keep prompts under a model context limit. When the threshold is
misconfigured, the client may never call `/v1/responses/compact` and can get stuck on repeated "context too long" errors.

codex-proxy can mitigate this by performing an **auto-compaction retry** on context-length failures.

## Behavior

When enabled, codex-proxy:

1. Sends the original `/v1/responses` request upstream.
2. If the upstream returns an error that looks like "context too long", codex-proxy:
   - compacts the *prefix* of `input.items[]`
   - keeps the last `N` items as the "tail"
   - retries the request with `input.items = [<compacted-prefix>, <tail...>]`
3. Returns the successful response to the client and sets:
   - `x-codex-proxy-auto-compacted: true`

The session key mechanism (see `sessions.md`) is used so sticky routing remains stable even though the input has been
rewritten.

## Provider compatibility

Auto-compaction is provider-independent by using two strategies:

- If the resolved request provider is OpenAI-compatible (`provider.type = open_ai`), codex-proxy uses **native**
  compaction (`/responses/compact`) and retries with the returned `encrypted_content` artifact.
- Otherwise, codex-proxy performs a **summary compaction** via the configured compaction target and retries with a
  synthetic system message containing a summary.

Summary compaction is best-effort and may lose some tool-call fidelity.

## Config

```json
{
  "auto_compaction": {
    "enabled": true,
    "max_attempts_per_request": 1,
    "tail_items_to_keep": 8,
    "compact_instructions": "Compact the conversation history for continued use...",
    "summary_instructions": "Summarize the conversation history so far..."
  }
}
```

Notes:

- `max_attempts_per_request` controls how many auto-compaction retries happen for a single request.
- `tail_items_to_keep` controls how much recent context is kept verbatim.


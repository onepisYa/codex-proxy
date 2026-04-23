# MiniMax Phase 0 - Namespace Hotfix Plan

## Context

新版 Codex 发 `type=namespace` 容器工具，fp-agent validate 直接 400 拒绝。
Phase 0 是 hotfix，独立交付。

## Current State (2026-04-22)

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1 | ✅ Done | TranslationSession, thread_local removed |
| Phase 2 | ✅ Done | tool_choice, BTreeMap, auth, max_output_tokens |
| Phase 3 | ✅ Done | wiremock deps, E2E done, default_max_tokens done |
| Phase 4 | ✅ Done | tracing spans, check_base_resp model param |
| **Phase 0** | ✅ **Done** | **namespace hotfix - fp-agent validate + schema + wire flatten** |

## Phase 0 Tasks

### T1: fp-agent validate_tools whitelist namespace
**File**: `third_party/FerroPhase/crates/fp-agent/src/validate.rs:69`

```rust
// Before
"function" | "web_search" | "retrieval" | "custom" => {}

// After
"function" | "web_search" | "retrieval" | "custom" | "namespace" => {}
```

### T2: fp-agent Tool struct add tools field
**File**: `third_party/FerroPhase/crates/fp-agent/src/schema/openai.rs:167`

Add field to `pub struct Tool`:
```rust
/// Inner tools for namespace container type
#[serde(default)]
pub tools: Option<Vec<Tool>>,
```

### T3: minimax_wire translate_tools namespace flatten
**File**: `src/providers/minimax_wire.rs:573` (`translate_tools`)

Add branch in filter_map:
```rust
if tool.tool_type == "namespace" {
    // Flatten: parent.name + "__" + inner.name
    // e.g. mcp__codex_apps__calendar + "__" + create_event
    if let Some(inner_tools) = &tool.tools {
        return inner_tools.iter()
            .filter(|inner| inner.tool_type == "function")
            .map(|inner| {
                let qualified_name = format!("{}__{}", tool.name.as_ref().unwrap_or(&String::new()), inner.name.as_ref().unwrap_or(&String::new()));
                AnthropicTool {
                    name: qualified_name,
                    description: inner.description.clone().unwrap_or_default(),
                    input_schema: inner.parameters.clone().unwrap_or(serde_json::json!({})),
                }
            })
            .collect();
    }
    return None;
}
```

### T4: Unit tests for namespace handling
**Files**: `src/providers/minimax_wire.rs` tests + fp-agent tests

Three tests:
1. `validate_namespace_type_passes` - fp-agent validate allows namespace
2. `tool_struct_parses_inner_tools` - Tool serde roundtrip with tools field
3. `translate_namespace_flattens_inner_tools` - namespace → N AnthropicTools with qualified names

### T5: Followup tracking
Track p5_zai_namespace and p5_openai_namespace as separate followup work.

## Verification

```bash
cargo fmt && cargo check && cargo test
```

## Files Touched

- `third_party/FerroPhase/crates/fp-agent/src/validate.rs` (T1)
- `third_party/FerroPhase/crates/fp-agent/src/schema/openai.rs` (T2)
- `src/providers/minimax_wire.rs` (T3)
- `src/providers/minimax_stream.rs` (possibly T3 if stream needs update)
- `tests/minimax_e2e.rs` (T4 - add namespace smoke test)

## Notes

- This is a hotfix - minimal change, no refactoring
- No breaking changes to existing providers
- Follows Linus principle: fix the actual problem, not the imaginary ones

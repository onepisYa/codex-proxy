# MiniMax Provider 优化计划

## 最终状态

| Phase | 状态 | 关键交付 |
|-------|------|----------|
| Phase 1 | ✅ 完成 | TranslationSession 替换 thread_local |
| Phase 2 | ✅ 完成 | 4 个 bug 修复 |
| Phase 3 | ✅ 完成 | wiremock deps 已添加，E2E placeholder 就绪，default_max_tokens 实现 |
| Phase 4 | ✅ 完成 | Tracing spans + check_base_resp model 参数 |
| Phase 0 | ✅ 完成 | fp-agent validate + schema + minimax_wire flatten (namespace 容器支持) |

## 已完成详情

### Phase 1: TranslationSession 重构
- `src/providers/minimax_session.rs` 新建
- `minimax_wire.rs` 删除所有 thread_local
- `minimax_stream.rs` StreamState 持有 TranslationSession
- Thinking signature 存取与回填

### Phase 2: Bug 修复
- tool_choice 具名: `{type: "tool", name: "get_weather"}`
- BTreeMap<usize, BlockState> 替换 HashMap
- 401/403 → ProxyError::Auth
- max_output_tokens 优先于 max_tokens

### Phase 3
- ✅ reasoning 事件名: `response.reasoning_text.delta`
- ✅ web_search warn 文案修正
- ✅ reasoning fallback: 16384 → 8192
- ✅ image_url drop warn
- ✅ wiremock/tracing-test 已添加为 dev-deps
- ✅ p3_default_max_tokens: `resolve_default_max_tokens()` 替换所有硬编码 4096

### Phase 4
- ✅ Tracing spans: `info_span!("minimax_request", model, stream, account_id)` + `.instrument(span)`
- ✅ check_base_resp: model 参数 + 错误消息含 model 上下文
- ✅ E2E smoke tests: 4 个 `#[ignore]` placeholder

## 验证

```bash
cargo fmt && cargo check && cargo test  # 78 passed, 5 ignored
```

## Phase 0 详情 (v5 namespace hotfix)

### p0_namespace_validate
- ✅ `third_party/FerroPhase/crates/fp-agent/src/validate.rs:69` — 白名单加 `"namespace"`

### p0_namespace_schema
- ✅ `third_party/FerroPhase/crates/fp-agent/src/schema/openai.rs` — Tool struct 加 `pub tools: Option<Vec<Tool>>` 字段

### p0_namespace_flatten_minimax
- ✅ `src/providers/minimax_wire.rs` — `translate_tools` 用 `flat_map` 处理 namespace 容器，展平为 qualified names（`mcp__server__tool`）

### p0_namespace_tests
- ✅ `third_party/FerroPhase/crates/fp-agent/src/validate.rs` — `test_validate_namespace_tool_type_passes`
- ✅ `src/providers/minimax_wire.rs` — `test_translate_namespace_flattens_inner_tools`
- ✅ `tests/minimax_e2e.rs` — namespace E2E smoke test

## Phase 2 详情

### p2_mcp_namespace_preserve
- ✅ `src/schema/sse.rs` — `FunctionCallItem` 新增 `namespace: Option<String>` 字段
- ✅ `src/providers/minimax_wire.rs` — 新增 `pub fn extract_namespace()` 辅助函数
- ✅ `src/providers/minimax_stream.rs` — 两处 `FunctionCallItem` 构造添加 namespace 提取
- ✅ `src/providers/zai.rs` / `zai_stream.rs` / `gemini_stream.rs` — 添加 `namespace: None`
- ✅ `src/providers/minimax_wire.rs` — 4 个 `extract_namespace` 单元测试

### p2_mcp_tool_name_length
- ✅ `src/providers/minimax_wire.rs` — 64字符工具名硬限制，`> 64` 丢弃
- ✅ `test_translate_tool_name_exceeds_64_chars_is_dropped`
- ✅ `test_translate_tool_name_exactly_64_chars_passes`

## Phase 3 详情

### p3_wiremock
- ✅ `tests/minimax_integration.rs` 新建（9 个 wiremock 场景）
  - sync_text_returns_output
  - streaming_delta_yields_sse
  - streaming_tool_call_emits_tool_events
  - tool_result_roundtrip
  - base_resp_error_maps_to_400
  - upstream_401_maps_to_auth_error
  - upstream_500_is_error_response
  - stream_idle_timeout_yields_error_event
  - concurrent_requests_isolated

## Phase 5 详情

### p5_zai_namespace
- ✅ `third_party/FerroPhase/crates/fp-agent/src/providers/zai.rs` — `transform_tools` 用 `flat_map` 处理 namespace 容器，展平为 qualified names
- ✅ 2 个单元测试: `transform_tools_namespace_flattens_inner_tools`, `transform_tools_non_namespace_passes_through`

### p5_openai_namespace
- ✅ `third_party/FerroPhase/crates/fp-agent/src/providers/openai.rs` — `flatten_namespace_tools` 用 `flat_map` 处理 namespace 容器
- ✅ 3 个单元测试: `flatten_namespace_tools_expands_inner_tools`, `flatten_namespace_tools_passes_non_namespace`, `flatten_namespace_tools_multiple_inner`

## 待完成

（全部完成）

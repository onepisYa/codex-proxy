# Minimax Provider 实施计划

## 概述

在 codex-proxy 中新增 Minimax provider，打通 OpenAI Responses API ↔ MiniMax Anthropic-wire 双向转换。

## 四阶段流水线

| Phase | 任务 | 依赖 | 状态 |
|-------|------|------|------|
| 1 | Config 层 + Provider 骨架 | - | 🚧 进行中 |
| 2 | Wire 层 Anthropic 编解码 | Phase 1 | ⏳ 待开始 |
| 3 | Stream 层 + 同步执行 | Phase 2 | ⏳ 待开始 |
| 4 | Compact 原生支持 | Phase 3 | ⏳ 待开始 |

## 架构图

```
Client (OpenAI Responses)
    │
    ▼
codex-proxy (Minimax Provider)
    │  translate_to_anthropic_request()
    ▼
Minimax API (Anthropic Messages)
    │
    ▼
minimax_stream (SSE 状态机) 或 minimax_wire (同步 JSON)
    │
    ▼
Client (OpenAI Responses)
```

## 关键设计决策

1. **Anthropic wire 自包含**：所有 Anthropic 类型定义在 minimax_wire.rs，不污染 fp-agent
2. **主数据源**：`ResponsesRequest.input`（InputItem 格式）
3. **Fallback**：无 input 时用 ChatRequest.messages
4. **错误处理**：HTTP 200 下 `base_resp.status_code != 0` 识别为业务错误

## 文件清单

### 新建
- `src/providers/minimax.rs` - Provider 主实现
- `src/providers/minimax_wire.rs` - Anthropic wire 类型 + 编解码
- `src/providers/minimax_stream.rs` - SSE 状态机

### 修改
- `src/providers/mod.rs` - 注册模块和 Provider
- `src/config/mod.rs` - ProviderType/ProviderConfig 新增 Minimax
- `src/server.rs` - compact_handler 放宽检查（Phase 4）

## Phase 1 任务清单

- [ ] ProviderType 增加 Minimax
- [ ] ProviderConfig 增加 Minimax 变体
- [ ] 新增 MinimaxProviderConfig 结构体
- [ ] 更新所有 exhaustive match
- [ ] accounts validate 支持 Minimax+ApiKey
- [ ] 新增 as_minimax() accessor
- [ ] ProviderRegistry 注册 Minimax
- [ ] 新建 minimax.rs（空壳，handle_request 返回 NotImplemented）
- [ ] 新建占位模块 minimax_wire.rs, minimax_stream.rs

## Phase 2 任务清单

- [ ] AnthropicRequest/Response/ContentBlock/Usage/BaseResp 类型
- [ ] translate_to_anthropic_request()
- [ ] translate_to_responses_response()
- [ ] check_base_resp()
- [ ] 6+ 单元测试

## Phase 3 任务清单

- [ ] Anthropic SSE → Responses SSE 状态机
- [ ] execute_request/handle_sync/handle_stream
- [ ] resolve_minimax_auth/build_messages_url
- [ ] 4+ stream 单元测试

## Phase 4 任务清单

- [ ] execute_compact 实现
- [ ] server.rs compact_handler 修改
- [ ] 示例配置文档

## 验证命令

```bash
cargo fmt
cargo check --all-targets
cargo test
```

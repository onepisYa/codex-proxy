# MCP 工具调用研究 & codex-proxy 扩展方向

## 1. MCP 工具调用失败根因分析

### 问题现象
```
"name":"__searchGitHub","namespace":"mcp__grep_app"
"output":"unsupported call: mcp__grep_app__searchGitHub"
```

### 根因
- MCP 服务器 `grep_app` 已注册并可被发现（`/mcp` 可见）
- 但工具未正确加载到会话的工具注册表中
- `unsupported call` 错误来自工具调度层找不到处理器
- 沙箱网络限制加剧问题（proxy 7890 端口被阻止）

**结论**：这是 Codex CLI 会话配置问题，非 codex-proxy 代码问题

---

## 2. TS 项目架构（icebear0828-codex-proxy）

### 目录结构
```
src/
├── translation/           # 协议转换层
│   ├── tool-format.ts     # 工具定义转换（核心）
│   ├── openai-to-codex.ts # OpenAI → Codex
│   ├── codex-to-openai.ts # Codex → OpenAI
│   ├── codex-to-anthropic.ts
│   ├── anthropic-to-codex.ts
│   └── codex-event-extractor.ts  # SSE 事件解析
├── proxy/                 # 上游连接处理
└── routes/
    └── responses.ts       # POST /v1/responses 端点
```

### 工具格式转换（tool-format.ts）

**CodexToolDefinition**:
```typescript
interface CodexToolDefinition {
  type: "function";
  name: string;
  description?: string;
  parameters?: Record<string, unknown>;
  strict?: boolean;
}
```

**转换模式**:
- OpenAI `tool_calls` → Codex `function_call` items
- OpenAI `tools` 数组 → Codex `tools` 数组
- `tool_choice` → Codex `tool_choice`

### SSE 事件提取（codex-event-extractor.ts）

处理流式工具调用：
- `response.output_item.added` + `function_call` → 工具调用开始
- `response.function_call_arguments.delta` → 参数增量
- `response.function_call_arguments.done` → 参数完成

---

## 3. OpenAI Responses API 工具调用

### 基础格式
```python
response = client.responses.create(
    model="gpt-4o",
    input="What's the news?",
    tools=[{"type": "web_search"}]
)
```

### 工具输出
```json
{
  "id": "ws_67bd64fe91f081919bec069ad65797f1",
  "status": "completed",
  "type": "web_search_call"
}
```

**关键特性**：
- 工具在请求创建时定义在 `tools` 数组
- 工具调用作为 output items 返回
- 支持多种工具类型：`web_search`, `file_search`, `computer_use`

---

## 4. Claude Code MCP 集成

### 工具命名规范
```
mcp__plugin_<plugin-name>_<server-name>__<tool-name>
```

示例：
- `mcp__plugin_asana_asana__asana_create_task`
- `mcp__plugin_myplug_database__query`

### Codex MCP 文件
- `codex-rs/core/src/mcp_tool_call.rs` - 工具调用分发
- `codex-rs/core/src/tools/handlers/mcp.rs` - MCP 处理器
- `codex-mcp/src/mcp_tool_names.rs` - 工具名限定

### 工具调用流程
1. `handle_mcp_tool_call()` 解析参数并调用 MCP 服务器
2. `McpHandler` 分发到 MCP 协议
3. 结果以 `McpToolOutput` 返回

---

## 5. codex-proxy 当前架构

### 核心流程
```
responses_handler
    ↓
normalize_responses_request (ResponsesRequest → ChatRequest)
    ↓
resolve_response_route (provider + account)
    ↓
provider.handle_request(raw, chat, headers, context)
    ↓
minimax_wire: translate_to_anthropic_request (ChatRequest → Minimax格式)
    ↓
stream_responses_sse (Minimaxi响应 → SSE)
```

### 现有工具处理（minimax_wire.rs）
```rust
fn translate_tools(tools: &[Tool]) -> Vec<AnthropicTool> {
    tools.iter().flat_map(|tool| {
        // namespace 展平: mcp__server__tool
        if tool.tool_type == "namespace" { ... }
        // 64字符工具名限制
        // web_search 跳过
    })
}
```

---

## 6. 扩展方向建议

### 短期（当前架构内）
1. **完善工具调用测试** - 增加更多 E2E 场景
2. **工具名长度优化** - 考虑 hash 缩短超长工具名
3. **错误处理增强** - tool_call 失败的细粒度错误码

### 中期（新增功能）
1. **MCP 服务器注册表** - 支持动态 MCP 服务器发现
2. **MCP 工具调用代理** - 转发 MCP 工具调用到外部服务器
3. **WebSocket 传输** - 支持 MCP WebSocket 传输（当前仅 HTTP/SSE）

### 长期（架构演进）
1. **统一工具格式层** - 类似 TS 项目的 `tool-format.ts` 集中管理
2. **多提供商工具映射** - 工具能力发现与映射
3. **工具调用结果缓存** - 相同工具调用的结果复用

---

## 7. 参考资料

### TS 项目
- `/Users/onepisya/github-knowledge/icebear0828-codex-proxy/src/translation/tool-format.ts`
- `/Users/onepisya/github-knowledge/icebear0828-codex-proxy/src/translation/openai-to-codex.ts`
- `/Users/onepisya/github-knowledge/icebear0828-codex-proxy/src/routes/responses.ts`

### Codex MCP
- `/Users/onepisya/github-knowledge/codex/codex-rs/core/src/mcp_tool_call.rs`
- `/Users/onepisya/github-knowledge/codex/codex-rs/core/src/tools/handlers/mcp.rs`

### Rust codex-proxy
- `/Users/onepisya/github-knowledge/JakkuSakura-codex-proxy/src/providers/minimax_wire.rs`
- `/Users/onepisya/github-knowledge/JakkuSakura-codex-proxy/src/providers/minimax_stream.rs`
- `/Users/onepisya/github-knowledge/JakkuSakura-codex-proxy/src/schema/sse.rs`

---

## 8. 下一步行动

1. [ ] 分析当前工具调用的瓶颈（是否有测试覆盖？）
2. [ ] 考虑是否需要 MCP 服务器代理功能
3. [ ] 评估是否需要 WebSocket 传输支持
4. [ ] 制定具体的工具名 hash 方案（如需要）

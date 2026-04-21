# 工具调用协议研究

## 概述

本文档整理 Claude Code、Codex、icebear Codex Proxy 三处代码库中工具调用的数据结构与执行流程，为 MiniMax Provider 工具调用调试提供参考。

---

## 一、Claude Code 工具调用架构

### 1.1 Hook 事件体系

Claude Code 通过 PreToolUse / PostToolUse 钩子拦截工具调用。

**PreToolUse Hook 输入**:
```json
{
  "session_id": "abc123",
  "hook_event_name": "PreToolUse",
  "tool_name": "Bash|Write|Edit|Read|...",
  "tool_input": { ... }
}
```

**PostToolUse Hook 输入**:
```json
{
  "hook_event_name": "PostToolUse",
  "tool_name": "...",
  "tool_input": { ... },
  "tool_result": { ... }
}
```

### 1.2 tool_use / tool_result 配对

```json
// 请求
{ "tool_use_id": "tool_123", "type": "tool_use", "name": "Bash", "input": { "command": "ls" } }

// 响应
{ "tool_use_id": "tool_123", "type": "tool_result", "content": "total 8 ..." }
```

关键：`tool_use_id` 用于配对请求与响应。

---

## 二、Codex RS 工具调用源码

### 2.1 核心数据结构

**文件**: `codex/codex-rs/protocol/src/models.rs`

```rust
pub enum ResponseItem {
    FunctionCall {
        id: Option<String>,      // tool_use id，可选
        name: String,            // 工具名
        call_id: String,         // 用于关联 output
        arguments: String,       // JSON string
    },
    FunctionCallOutput {
        call_id: String,         // 关联到 FunctionCall.call_id
        output: FunctionCallOutputPayload,
    },
}
```

**Tool 定义** (`models.rs`):
```rust
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}
```

**CallToolResult** (`mcp.rs`):
```rust
pub struct CallToolResult {
    pub content: Vec<serde_json::Value>,  // MCP content blocks
    pub structured_content: Option<serde_json::Value>,
    pub is_error: Option<bool>,
}
```

### 2.2 执行流程

**文件**: `codex/codex-rs/core/src/mcp_tool_call.rs`

```rust
pub(crate) async fn handle_mcp_tool_call(
    sess: Arc<Session>,
    call_id: String,
    server: String,
    tool_name: String,
    arguments: String,
) -> CallToolResult {
    // 1. 解析 arguments 为 JSON
    let arguments_value = serde_json::from_str(&arguments).ok();

    // 2. 发送 McpToolCallBeginEvent

    // 3. 执行工具调用
    let result = execute_mcp_tool_call(...).await;

    // 4. 发送 McpToolCallEndEvent，携带 CallToolResult
}
```

### 2.3 结果转换

`CallToolResult` → `FunctionCallOutputPayload`:

```rust
impl CallToolResult {
    pub fn as_function_call_output_payload(&self) -> FunctionCallOutputPayload {
        if let Some(structured_content) = &self.structured_content {
            return FunctionCallOutputPayload {
                body: FunctionCallOutputBody::Text(serialized_structured_content),
                success: Some(self.success()),
            };
        }
        // 否则转换 content 为 FunctionCallOutputContentItem
        let content_items = convert_mcp_content_to_items(&self.content);
        FunctionCallOutputPayload { body: ..., success: Some(self.success()) }
    }
}
```

---

## 三、icebear Codex Proxy 翻译实现

### 3.1 Codex 类型定义

**文件**: `icebear0828-codex-proxy/src/proxy/codex-types.ts`

```typescript
export type CodexInputItem =
  | { role: "user"; content: string | CodexContentPart[] }
  | { role: "assistant"; content: string }
  | { role: "system"; content: string }
  | { type: "function_call"; id?: string; call_id: string; name: string; arguments: string }
  | { type: "function_call_output"; call_id: string; output: string };
```

注意：`function_call` 的 `call_id` 是**必需字段**，用于关联 output。

### 3.2 Anthropic 请求转换

**文件**: `icebear0828-codex-proxy/src/translation/codex-request-to-anthropic.ts`

```typescript
type AnthropicContentBlock =
  | { type: "text"; text: string }
  | { type: "image"; source: { type: "url"; url: string } | { type: "base64"; ... } }
  | { type: "tool_use"; id: string; name: string; input: unknown }
  | { type: "tool_result"; tool_use_id: string; content: string };

export function translateCodexToAnthropicRequest(
  req: CodexResponsesRequest,
  modelId: string,
): AnthropicMessageRequest {
  const messages = inputItemsToAnthropicMessages(req.input);

  return {
    model: modelId,
    messages,
    max_tokens: 8192,
    stream: req.stream,
    system: req.instructions,
    thinking: req.reasoning?.effort
      ? { type: "enabled", budget_tokens: REASONING_EFFORT_BUDGET[req.reasoning.effort] }
      : undefined,
    tools: req.tools,
    tool_choice: req.tool_choice,
  };
}
```

### 3.3 inputItemsToAnthropicMessages 逻辑

```typescript
function inputItemsToAnthropicMessages(input: CodexInputItem[]): AnthropicMessage[] {
  for (const item of input) {
    if (item.type === "function_call") {
      // 转为 tool_use block，合并到前一条 assistant message
      const toolUse = {
        type: "tool_use" as const,
        id: item.call_id,       // 用 call_id 作为 id
        name: item.name,
        input: JSON.parse(item.arguments),
      };
      // 合并到 last assistant message
    }
    if (item.type === "function_call_output") {
      // 转为 tool_result block，必须在 user message 中
      const toolResult = {
        type: "tool_result" as const,
        tool_use_id: item.call_id,  // 用上游 call_id
        content: item.output,
      };
    }
  }
}
```

**关键**: `tool_use_id` 必须与上游 `tool_use.id` 匹配。

---

## 四、当前 MiniMax Provider 实现

### 4.1 Anthropic 类型定义

**文件**: `src/providers/minimax_wire.rs`

```rust
pub enum AnthropicContentBlock {
    Text { text: String },
    ToolUse { id: String, name: String, input: serde_json::Value },
    ToolResult { tool_use_id: String, content: String },
    Thinking { thinking: String, signature: String },
}
```

### 4.2 翻译逻辑

**function_call 处理** (minimax_wire.rs:235-281):
```rust
"function_call" => {
    let name = item.name.clone().unwrap_or_default();
    let arguments = item.arguments.as_ref().map(|v| v.to_string()).unwrap_or_else(|| "{}".to_string());

    // 解析 arguments，必须为 object
    let input: serde_json::Value = match serde_json::from_str::<serde_json::Value>(&arguments) {
        Ok(serde_json::Value::Object(obj)) => serde_json::Value::Object(obj),
        _ => serde_json::json!({}),  // Bug 1 修复：空字符串 fallback
    };

    let id = item.id.clone().unwrap_or_else(|| generate_tool_id());

    let tool_block = AnthropicContentBlock::ToolUse { id, name, input };
    // 合并到前一条 assistant message
}
```

**function_call_output 处理** (minimax_wire.rs:282-308):
```rust
"function_call_output" => {
    // Bug 2 修复：call_id 缺失则跳过
    let Some(tool_use_id) = item.call_id.clone() else {
        i += 1;
        continue;
    };

    let output_content = item.output.as_ref().map(...).unwrap_or_default();

    let tool_result = AnthropicContentBlock::ToolResult {
        tool_use_id,  // 使用上游 call_id
        content: output_content,
    };

    // 创建 user message 承载 tool_result
    anthropic_messages.push(AnthropicMessage {
        role: "user".to_string(),
        content: AnthropicMessageContent::Blocks(vec![tool_result]),
    });
}
```

---

## 五、关键发现总结

| 发现点 | 说明 |
|--------|------|
| `call_id` 是关联桥梁 | `function_call.call_id` → `function_call_output.call_id` |
| `tool_use_id` 须匹配 | `ToolResult.tool_use_id` 必须等于 `ToolUse.id` |
| arguments 须为 object | 空字符串或非对象时 fallback 到 `{}` |
| tool_result 须在 user message | Anthropic 要求 tool_result 在 role="user" 的消息中 |

---

## 六、调试建议

1. **确认服务已重新构建**: `cargo build --release` 后重启
2. **检查 SSE 流中 id 字段**: MiniMax 返回的 `tool_use.id` 是否正确解析
3. **验证 call_id 传递链**: 上游 `function_call.call_id` → 转换为 `ToolUse.id` → 下游 `ToolResult.tool_use_id`

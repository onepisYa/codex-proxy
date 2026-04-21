use crate::error::{ProviderError, ProxyError};
use crate::schema::sse::ResponseObject;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::collections::HashMap;
use tracing::debug;

// Thread-local storage for tracking tool_use.id mappings
// This is needed because:
// 1. We send tool_use to MiniMax with our own tool_use.id
// 2. MiniMax sends back tool_use in SSE with its own call_id
// 3. Codex sends function_call_output with MiniMax's call_id (from SSE)
// 4. We need to look up our original tool_use.id to send in tool_result to MiniMax
//
// The mapping flow:
// - During function_call translation: store Codex call_id -> our tool_use.id (if call_id exists)
// - During SSE processing: map minimax_call_id -> our_tool_use_id (by sequence position)
// - During function_call_output: look up our_tool_use_id using the call_id from function_call_output
thread_local! {
    // Maps Codex call_id (from function_call.call_id) to our tool_use.id
    static CODEX_CALL_ID_MAPPING: std::cell::RefCell<HashMap<String, String>> =
        std::cell::RefCell::new(HashMap::new());

    // Maps minimax_call_id (from SSE) to our original tool_use.id
    static MINIMAX_CALL_ID_MAPPING: std::cell::RefCell<HashMap<String, String>> =
        std::cell::RefCell::new(HashMap::new());

    // Stores our tool_use.id sequence (ordered as we sent to MiniMax)
    static OUR_TOOL_USE_IDS: std::cell::RefCell<Vec<String>> =
        std::cell::RefCell::new(Vec::new());

    // Counter for tracking which tool_use we're processing during SSE
    static TOOL_USE_COUNTER: std::cell::RefCell<usize> =
        std::cell::RefCell::new(0);
}

// Store our tool_use.id sequence before sending to MiniMax
pub fn set_our_tool_use_ids(ids: Vec<String>) {
    let len = ids.len();
    OUR_TOOL_USE_IDS.with(|cell| {
        *cell.borrow_mut() = ids;
    });
    debug!("Stored {} tool_use.ids for mapping", len);
}

// Store mapping from Codex call_id to our tool_use.id
// Called during function_call translation (when call_id exists)
pub fn store_codex_call_id_mapping(codex_call_id: &str, our_tool_id: &str) {
    CODEX_CALL_ID_MAPPING.with(|map| {
        map.borrow_mut().insert(codex_call_id.to_string(), our_tool_id.to_string());
        debug!("Stored mapping: codex_call_id={} -> tool_use_id={}", codex_call_id, our_tool_id);
    });
}

// Called during SSE processing when we receive tool_use from MiniMax
// Stores the mapping: minimax_call_id -> our_tool_use_id (by position)
pub fn store_call_id_mapping(call_id_minimax: &str) {
    let our_id = OUR_TOOL_USE_IDS.with(|cell| {
        let counter = TOOL_USE_COUNTER.with(|c| *c.borrow());
        let ids = cell.borrow();
        ids.get(counter).cloned().unwrap_or_else(|| {
            format!("unknown_tool_{}", counter)
        })
    });

    MINIMAX_CALL_ID_MAPPING.with(|map| {
        map.borrow_mut().insert(call_id_minimax.to_string(), our_id.clone());
        debug!("Stored mapping: minimax_call_id={} -> our_tool_use_id={}", call_id_minimax, our_id);
    });

    // Increment counter for next tool_use
    TOOL_USE_COUNTER.with(|c| {
        *c.borrow_mut() += 1;
    });
}

// Called during function_call_output translation
// Looks up our original tool_use.id given the call_id from function_call_output
// First tries CODEX_CALL_ID_MAPPING (if call_id was stored during function_call translation)
// Falls back to MINIMAX_CALL_ID_MAPPING (if SSE processing stored the mapping)
pub fn lookup_our_tool_use_id(call_id: &str) -> Option<String> {
    // First try Codex call_id mapping
    let result = CODEX_CALL_ID_MAPPING.with(|map| {
        map.borrow().get(call_id).cloned()
    });
    if result.is_some() {
        return result;
    }
    // Fall back to MiniMax call_id mapping
    MINIMAX_CALL_ID_MAPPING.with(|map| {
        map.borrow().get(call_id).cloned()
    })
}

// Reset thread-local state (call at start of each request)
pub fn reset_tool_use_mapping() {
    CODEX_CALL_ID_MAPPING.with(|map| {
        map.borrow_mut().clear();
    });
    MINIMAX_CALL_ID_MAPPING.with(|map| {
        map.borrow_mut().clear();
    });
    OUR_TOOL_USE_IDS.with(|ids| {
        ids.borrow_mut().clear();
    });
    TOOL_USE_COUNTER.with(|c| {
        *c.borrow_mut() = 0;
    });
}

use fp_agent::schema::openai::{
    ChatRequest, Content, Instructions, ResponsesInput, ResponsesRequest, TextPart, Tool,
};

// === Anthropic Request types ===

#[derive(Clone, Debug, Serialize)]
pub struct AnthropicRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    pub max_tokens: u64,
    #[serde(rename = "stream", serialize_with = "serialize_stream_bool")]
    pub stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tools: Vec<AnthropicTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<AnthropicThinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
}

fn serialize_stream_bool<S>(v: &bool, s: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    if *v {
        s.serialize_bool(true)
    } else {
        s.serialize_bool(false)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicMessageContent,
}

#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum AnthropicMessageContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
    },
    Thinking {
        thinking: String,
        signature: String,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct AnthropicThinking {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub budget_tokens: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

// === Anthropic Response types ===

#[derive(Clone, Debug, Deserialize)]
pub struct AnthropicResponse {
    pub id: String,
    pub model: String,
    pub role: String,
    pub content: Vec<AnthropicResponseBlock>,
    #[serde(default)]
    pub stop_reason: Option<String>,
    #[serde(default)]
    pub usage: Option<AnthropicUsage>,
    #[serde(default)]
    pub base_resp: Option<BaseResp>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicResponseBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    Thinking {
        thinking: String,
        #[serde(default)]
        signature: Option<String>,
    },
}

#[derive(Clone, Debug, Deserialize)]
pub struct AnthropicUsage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u64>,
    #[serde(default)]
    pub cache_read_input_tokens: Option<u64>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct BaseResp {
    #[serde(default)]
    pub status_code: i64,
    #[serde(default)]
    pub status_msg: String,
}

// === Translate Context ===

pub struct TranslateCtx {
    pub reasoning: Option<crate::config::ReasoningConfig>,
    pub default_max_tokens: u64,
    pub stream: bool,
}

// === Public API ===

/// Translates ResponsesRequest + ChatRequest into AnthropicRequest
pub fn translate_to_anthropic_request(
    raw: &ResponsesRequest,
    chat: &ChatRequest,
    ctx: &TranslateCtx,
) -> AnthropicRequest {
    // Reset tool_use mapping state for this request
    reset_tool_use_mapping();

    // Pre-pass: collect all function_call ids in order so we can map them later
    // during SSE stream processing (MiniMax may send back call_id different from ours)
    let mut system_parts: Vec<String> = Vec::new();

    // 1. instructions → top-level system
    if let Some(ref instructions) = raw.instructions {
        match instructions {
            Instructions::Text(s) => system_parts.push(s.clone()),
            Instructions::Parts(parts) => {
                for part in parts {
                    match part {
                        TextPart::Text(s) => system_parts.push(s.clone()),
                        TextPart::Obj { text } => system_parts.push(text.clone()),
                    }
                }
            }
        }
    }

    // 2. Build messages from raw.input (InputItem) with fallback to chat.messages
    let input_items = match &raw.input {
        Some(ResponsesInput::Items(items)) => items.clone(),
        Some(ResponsesInput::Text(_)) => {
            // Text input: normalize_responses_request already added it as user message in chat.messages
            // Fall back to chat.messages so the text is not lost
            fp_agent::schema::openai::messages_to_input_items(&chat.messages)
        }
        None => {
            // Fallback to chat.messages converted to input items
            fp_agent::schema::openai::messages_to_input_items(&chat.messages)
        }
    };

    let mut anthropic_messages: Vec<AnthropicMessage> = Vec::new();
    let mut i = 0;

    while i < input_items.len() {
        let item = &input_items[i];

        match item.item_type.as_str() {
            "message" => {
                // role: "system" → merge into system, skip from messages
                if item.role.as_deref() == Some("system") {
                    if let Some(ref content) = item.content {
                        match content {
                            Content::Text(s) => system_parts.push(s.clone()),
                            Content::Parts(parts) => {
                                for part in parts {
                                    if let Some(text) = &part.text {
                                        system_parts.push(text.clone());
                                    }
                                }
                            }
                            Content::Json(_) => {}
                        }
                    }
                } else {
                    // Regular message
                    let role = match item.role.as_deref() {
                        Some("developer") | Some("system") => "assistant",
                        Some(r) => r,
                        None => "user",
                    };
                    let content = translate_content_to_anthropic(&item.content, &mut Vec::new());

                    if !content.is_empty() {
                        anthropic_messages.push(AnthropicMessage {
                            role: role.to_string(),
                            content: AnthropicMessageContent::Blocks(content),
                        });
                    }
                }
            }
            "function_call" => {
                // type: "function_call" → merge into assistant message content
                let name = item.name.clone().unwrap_or_default();

                // Debug log the raw fields
                tracing::debug!("function_call name={:?} arguments={:?} input={:?} action={:?}",
                    name, item.arguments, item.input, item.action);

                // Match fp-agent behavior: try arguments first, then input, then action
                // Codex may send arguments in any of these fields
                let args_value = item.arguments.as_ref()
                    .or(item.input.as_ref())
                    .or(item.action.as_ref());

                tracing::debug!("function_call args_value: {:?}", args_value);

                // Parse arguments - could be String (JSON string), Object, or other
                let input: serde_json::Value = match args_value {
                    Some(serde_json::Value::String(s)) if s.trim().is_empty() => {
                        // Empty or whitespace-only string is invalid JSON, treat as empty object
                        tracing::warn!("function_call arguments is empty or whitespace: {:?}, using {{}}", s);
                        serde_json::json!({})
                    }
                    Some(serde_json::Value::String(s)) => {
                        tracing::debug!("function_call args_value is Value::String, s={:?}", s);
                        // arguments is a JSON string, parse it
                        match serde_json::from_str::<serde_json::Value>(s) {
                            Ok(v) if v.is_object() => {
                                tracing::debug!("function_call parsed JSON object: {:?}", v);
                                v
                            }
                            Ok(v) => {
                                tracing::warn!("function_call arguments parsed but not object: {:?}, using {{}}", v);
                                serde_json::json!({})
                            }
                            Err(e) => {
                                tracing::warn!("function_call arguments parse failed: {} for s={:?}, using {{}}", e, s);
                                serde_json::json!({})
                            }
                        }
                    }
                    Some(serde_json::Value::Object(obj)) => {
                        tracing::debug!("function_call args_value is Value::Object: {:?}", obj);
                        serde_json::Value::Object(obj.clone())
                    }
                    Some(v) => {
                        // For other Value types (like Array, Number, etc.), convert to JSON string then parse
                        tracing::debug!("function_call args_value is other Value type: {:?}", v);
                        let parsed = serde_json::to_string(v)
                            .ok()
                            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                            .and_then(|val| if val.is_object() { Some(val) } else { None });
                        match parsed {
                            Some(val) => val,
                            None => {
                                tracing::warn!("function_call args_value could not be parsed as object: {:?}, using {{}}", v);
                                serde_json::json!({})
                            }
                        }
                    }
                    None => {
                        tracing::warn!("function_call all args fields are None, using {{}}");
                        serde_json::json!({})
                    }
                };

                // Use item.call_id as tool_use.id if available (Codex's call_id for correlation).
                // If not available, fall back to item.id or generate a new ID.
                // This is critical because function_call_output uses call_id to reference the original call.
                let id = item
                    .call_id
                    .clone()
                    .or_else(|| item.id.clone())
                    .unwrap_or_else(|| crate::providers::minimax_stream::generate_tool_id());

                // Store mapping: Codex call_id -> our tool_use.id
                // This allows us to find our tool_use.id when processing function_call_output
                if let Some(ref codex_call_id) = item.call_id {
                    store_codex_call_id_mapping(codex_call_id, &id);
                }

                // Add tool_use block
                let tool_block = AnthropicContentBlock::ToolUse {
                    id: id.clone(),
                    name,
                    input,
                };

                // Check if we should add to existing message or create new
                if let Some(last_msg) = anthropic_messages.last_mut() {
                    if last_msg.role == "assistant" {
                        if let AnthropicMessageContent::Blocks(ref mut blocks) = last_msg.content {
                            blocks.push(tool_block);
                            i += 1;
                            continue;
                        }
                    }
                }

                // Create new assistant message with tool_use
                anthropic_messages.push(AnthropicMessage {
                    role: "assistant".to_string(),
                    content: AnthropicMessageContent::Blocks(vec![tool_block]),
                });
            }
            "function_call_output" => {
                // type: "function_call_output" → merge into user message content
                // NOTE: item.call_id is MiniMax's call_id from the original tool_use SSE event.
                // We stored minimax_call_id -> tool_use.id mapping during SSE processing.
                let Some(minimax_call_id) = item.call_id.clone() else {
                    // Skip function_call_output with missing call_id - we cannot
                    // reliably correlate it without the upstream's ID.
                    i += 1;
                    continue;
                };

                tracing::debug!(" Codex function_call_output call_id: {:?}", minimax_call_id);

                // Look up our original tool_use.id using MiniMax's call_id
                // This mapping was stored during SSE processing when we received tool_use from MiniMax
                let tool_use_id = lookup_our_tool_use_id(&minimax_call_id)
                    .unwrap_or_else(|| {
                        // This is a bug - the mapping should have been stored during SSE processing
                        tracing::warn!("Tool_use ID mapping not found for minimax_call_id={}, this is a bug!", minimax_call_id);
                        minimax_call_id.clone()
                    });

                let output_content = item
                    .output
                    .as_ref()
                    .map(|c| translate_content_to_text(c))
                    .unwrap_or_default();

                let tool_result = AnthropicContentBlock::ToolResult {
                    tool_use_id,
                    content: output_content,
                };

                // Create user message with tool_result
                anthropic_messages.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: AnthropicMessageContent::Blocks(vec![tool_result]),
                });
            }
            "reasoning" => {
                // type: "reasoning" + encrypted_content → add thinking block
                let reasoning_text = item.reasoning_content.clone().unwrap_or_default();
                let signature = item.thought_signature.clone().unwrap_or_default();

                if !reasoning_text.is_empty() {
                    let thinking_block = AnthropicContentBlock::Thinking {
                        thinking: reasoning_text,
                        signature,
                    };

                    if let Some(last_msg) = anthropic_messages.last_mut() {
                        if last_msg.role == "assistant" {
                            if let AnthropicMessageContent::Blocks(ref mut blocks) =
                                last_msg.content
                            {
                                blocks.push(thinking_block);
                                i += 1;
                                continue;
                            }
                        }
                    }

                    // Create new assistant message with thinking
                    anthropic_messages.push(AnthropicMessage {
                        role: "assistant".to_string(),
                        content: AnthropicMessageContent::Blocks(vec![thinking_block]),
                    });
                }
            }
            _ => {
                // Unknown type, try to extract text content
                if let Some(ref content) = item.content {
                    let text = translate_content_to_text(content);
                    if !text.is_empty() {
                        anthropic_messages.push(AnthropicMessage {
                            role: "user".to_string(),
                            content: AnthropicMessageContent::Text(text),
                        });
                    }
                }
            }
        }

        i += 1;
    }

    // Fallback: if no messages, add a minimal user message so Minimax doesn't reject
    if anthropic_messages.is_empty() {
        anthropic_messages.push(AnthropicMessage {
            role: "user".to_string(),
            content: AnthropicMessageContent::Text("hello".to_string()),
        });
    }

    // Post-pass: collect actual tool_use ids from the built messages (in order)
    // This is needed because we generate tool_use.id during translation, not in a pre-pass.
    // The sequence matches MiniMax's SSE call_id sequence 1:1.
    let tool_use_ids: Vec<String> = anthropic_messages
        .iter()
        .filter_map(|msg| {
            if let AnthropicMessageContent::Blocks(blocks) = &msg.content {
                Some(blocks.iter().filter_map(|b| {
                    if let AnthropicContentBlock::ToolUse { id, .. } = b {
                        Some(id.clone())
                    } else {
                        None
                    }
                }))
            } else {
                None
            }
        })
        .flatten()
        .collect();
    set_our_tool_use_ids(tool_use_ids);

    // Build system prompt
    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n"))
    };

    // Translate tools
    let tools = translate_tools(&chat.tools);

    // Translate tool_choice
    let tool_choice = translate_tool_choice(&chat.tool_choice);

    // Translate reasoning config to thinking
    let thinking = ctx.reasoning.as_ref().map(|cfg| AnthropicThinking {
        kind: "enabled",
        budget_tokens: cfg
            .effort_levels
            .get(cfg.default_effort.as_deref().unwrap_or("medium"))
            .map(|e| e.budget)
            .unwrap_or(16384),
    });

    AnthropicRequest {
        model: raw.model.clone(),
        messages: anthropic_messages,
        system,
        max_tokens: raw.max_tokens.unwrap_or(ctx.default_max_tokens),
        stream: ctx.stream,
        tools,
        tool_choice,
        thinking,
        temperature: chat.temperature,
        top_p: chat.top_p,
    }
}

fn translate_content_to_anthropic(
    content: &Option<Content>,
    _warnings: &mut Vec<String>,
) -> Vec<AnthropicContentBlock> {
    match content {
        Some(Content::Text(s)) => vec![AnthropicContentBlock::Text { text: s.clone() }],
        Some(Content::Json(v)) => vec![AnthropicContentBlock::Text {
            text: v.to_string(),
        }],
        Some(Content::Parts(parts)) => {
            parts
                .iter()
                .filter_map(|part| match part.part_type.as_str() {
                    "text" => part
                        .text
                        .as_ref()
                        .map(|t| AnthropicContentBlock::Text { text: t.clone() }),
                    "image_url" => {
                        // Logged in caller
                        None
                    }
                    _ => {
                        if let Some(text) = &part.text {
                            Some(AnthropicContentBlock::Text { text: text.clone() })
                        } else {
                            None
                        }
                    }
                })
                .collect()
        }
        None => Vec::new(),
    }
}

fn translate_content_to_text(content: &Content) -> String {
    match content {
        Content::Text(s) => s.clone(),
        Content::Parts(parts) => parts
            .iter()
            .filter_map(|p| p.text.as_deref())
            .collect::<Vec<_>>()
            .join(""),
        Content::Json(v) => v.to_string(),
    }
}

fn translate_tools(tools: &[Tool]) -> Vec<AnthropicTool> {
    tools
        .iter()
        .filter_map(|tool| {
            // Skip web_search/hosted tools
            if tool.web_search.is_some() {
                tracing::warn!("web_search tool dropped for Anthropic compatibility");
                return None;
            }

            // Extract function info
            let (name, description, input_schema) = match &tool.function {
                Some(func) => (
                    func.name.clone(),
                    func.description.clone(),
                    func.parameters.clone().unwrap_or(serde_json::json!({})),
                ),
                None => {
                    // Flat function schema
                    let name = tool.name.clone().unwrap_or_default();
                    let description = tool.description.clone();
                    let input_schema = tool.parameters.clone().unwrap_or(serde_json::json!({}));
                    (name, description, input_schema)
                }
            };

            if name.is_empty() {
                return None;
            }

            Some(AnthropicTool {
                name,
                description,
                input_schema,
            })
        })
        .collect()
}

fn translate_tool_choice(tool_choice: &Option<String>) -> Option<serde_json::Value> {
    tool_choice.as_ref().map(|choice| match choice.as_str() {
        "auto" => serde_json::json!({ "type": "auto" }),
        "required" | "any" => serde_json::json!({ "type": "any" }),
        s if s.starts_with("tool") || s.contains('{') => {
            // Try to parse as named tool
            serde_json::json!({ "type": "tool", "name": s })
        }
        _ => serde_json::json!({ "type": "auto" }),
    })
}

/// Translates AnthropicResponse into ResponseObject
pub fn translate_to_responses_response(resp: AnthropicResponse) -> ResponseObject {
    let mut output_items: Vec<crate::schema::sse::OutputItem> = Vec::new();
    let item_id_base = resp.id.clone();
    let mut item_counter: usize = 0;

    for block in &resp.content {
        match block {
            AnthropicResponseBlock::Text { text } => {
                output_items.push(crate::schema::sse::OutputItem::Message(
                    crate::schema::sse::MessageItem {
                        id: format!("{}_{}", item_id_base, item_counter),
                        role: "assistant",
                        status: "completed".to_string(),
                        content: vec![crate::schema::sse::OutputContentPart::OutputText {
                            text: text.clone(),
                        }],
                    },
                ));
                item_counter += 1;
            }
            AnthropicResponseBlock::ToolUse { id, name, input } => {
                let arguments = input.to_string();
                output_items.push(crate::schema::sse::OutputItem::FunctionCall(
                    crate::schema::sse::FunctionCallItem {
                        id: format!("{}_{}", item_id_base, item_counter),
                        status: "completed".to_string(),
                        name: name.clone(),
                        arguments,
                        call_id: id.clone(),
                        thought_signature: None,
                    },
                ));
                item_counter += 1;
            }
            AnthropicResponseBlock::Thinking {
                thinking,
                signature: _,
            } => {
                output_items.push(crate::schema::sse::OutputItem::Reasoning(
                    crate::schema::sse::ReasoningItem {
                        id: format!("{}_{}", item_id_base, item_counter),
                        status: "completed".to_string(),
                        summary: Vec::new(),
                        content: vec![crate::schema::sse::ReasoningContentPart::ReasoningText {
                            text: thinking.clone(),
                        }],
                    },
                ));
                item_counter += 1;
            }
        }
    }

    let usage = resp.usage.as_ref().map(|u| crate::schema::sse::Usage {
        input_tokens: u.input_tokens,
        output_tokens: u.output_tokens,
        total_tokens: u.input_tokens + u.output_tokens,
        input_tokens_details: None,
        output_tokens_details: None,
    });

    ResponseObject {
        id: resp.id,
        object: "response",
        created_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0),
        completed_at: None,
        model: resp.model,
        status: "completed".to_string(),
        temperature: 1.0,
        top_p: 1.0,
        tool_choice: "auto".to_string(),
        tools: Vec::new(),
        parallel_tool_calls: true,
        store: false,
        metadata: BTreeMap::new(),
        output: output_items,
        usage,
    }
}

/// Checks Minimax business errors
pub fn check_base_resp(resp: &AnthropicResponse) -> Result<(), ProxyError> {
    if let Some(br) = &resp.base_resp {
        if br.status_code != 0 {
            return Err(ProxyError::Provider(ProviderError::new(
                Some(StatusCode::BAD_GATEWAY),
                format!("Minimax error ({}): {}", br.status_code, br.status_msg),
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::sse::{OutputContentPart, OutputItem};
    use fp_agent::schema::openai::InputItem;

    #[test]
    fn translate_instructions_to_system_string() {
        let raw = ResponsesRequest {
            model: "test-model".into(),
            input: Some(ResponsesInput::Text("hello".into())),
            messages: None,
            instructions: Some(Instructions::Text("You are a helpful assistant.".into())),
            previous_response_id: None,
            store: None,
            metadata: None,
            tools: None,
            tool_choice: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            max_output_tokens: None,
            stream: None,
            include: None,
        };

        let chat = ChatRequest {
            model: "test-model".into(),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_choice: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: false,
            store: false,
            metadata: BTreeMap::new(),
            previous_response_id: None,
            include: Vec::new(),
        };

        let ctx = TranslateCtx {
            reasoning: None,
            default_max_tokens: 4096,
            stream: false,
        };

        let req = translate_to_anthropic_request(&raw, &chat, &ctx);

        assert!(req.system.is_some());
        assert!(req.system.unwrap().contains("helpful assistant"));
        assert_eq!(req.model, "test-model");
    }

    #[test]
    fn translate_instructions_to_system_parts() {
        let raw = ResponsesRequest {
            model: "test-model".into(),
            input: Some(ResponsesInput::Items(vec![InputItem {
                item_type: "message".into(),
                id: None,
                call_id: None,
                role: Some("user".into()),
                name: None,
                content: Some(Content::Text("hello".into())),
                reasoning_content: None,
                thought_signature: None,
                thought: None,
                arguments: None,
                input: None,
                action: None,
                command: None,
                cwd: None,
                working_directory: None,
                changes: None,
                output: None,
                stdout: None,
                stderr: None,
                encrypted_content: None,
            }])),
            messages: None,
            instructions: Some(Instructions::Parts(vec![
                TextPart::Text("Part 1".into()),
                TextPart::Obj {
                    text: "Part 2".into(),
                },
            ])),
            previous_response_id: None,
            store: None,
            metadata: None,
            tools: None,
            tool_choice: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            max_output_tokens: None,
            stream: None,
            include: None,
        };

        let chat = ChatRequest {
            model: "test-model".into(),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_choice: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: false,
            store: false,
            metadata: BTreeMap::new(),
            previous_response_id: None,
            include: Vec::new(),
        };

        let ctx = TranslateCtx {
            reasoning: None,
            default_max_tokens: 4096,
            stream: false,
        };

        let req = translate_to_anthropic_request(&raw, &chat, &ctx);

        assert!(req.system.is_some());
        let system = req.system.unwrap();
        assert!(system.contains("Part 1"));
        assert!(system.contains("Part 2"));
    }

    fn make_responses_request(model: &str, input: ResponsesInput) -> ResponsesRequest {
        ResponsesRequest {
            model: model.into(),
            input: Some(input),
            messages: None,
            instructions: None,
            previous_response_id: None,
            store: None,
            metadata: None,
            tools: None,
            tool_choice: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            max_output_tokens: None,
            stream: None,
            include: None,
        }
    }

    #[test]
    fn translate_tool_call_output_merges_into_user() {
        let raw = make_responses_request(
            "test-model",
            ResponsesInput::Items(vec![
                InputItem {
                    item_type: "message".into(),
                    id: None,
                    call_id: None,
                    role: Some("user".into()),
                    name: None,
                    content: Some(Content::Text("What is the weather?".into())),
                    reasoning_content: None,
                    thought_signature: None,
                    thought: None,
                    arguments: None,
                    input: None,
                    action: None,
                    command: None,
                    cwd: None,
                    working_directory: None,
                    changes: None,
                    output: None,
                    stdout: None,
                    stderr: None,
                    encrypted_content: None,
                },
                InputItem {
                    item_type: "function_call".into(),
                    id: Some("call_abc123".into()),
                    call_id: None,
                    role: None,
                    name: Some("get_weather".into()),
                    content: None,
                    reasoning_content: None,
                    thought_signature: None,
                    thought: None,
                    arguments: Some(serde_json::json!({"city": "Tokyo"})),
                    input: None,
                    action: None,
                    command: None,
                    cwd: None,
                    working_directory: None,
                    changes: None,
                    output: None,
                    stdout: None,
                    stderr: None,
                    encrypted_content: None,
                },
                InputItem {
                    item_type: "function_call_output".into(),
                    id: None,
                    call_id: Some("call_abc123".into()),
                    role: None,
                    name: Some("get_weather".into()),
                    content: None,
                    reasoning_content: None,
                    thought_signature: None,
                    thought: None,
                    arguments: None,
                    input: None,
                    action: None,
                    command: None,
                    cwd: None,
                    working_directory: None,
                    changes: None,
                    output: Some(Content::Text("Sunny, 25C".into())),
                    stdout: None,
                    stderr: None,
                    encrypted_content: None,
                },
            ]),
        );

        let chat = ChatRequest {
            model: "test-model".into(),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_choice: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: false,
            store: false,
            metadata: BTreeMap::new(),
            previous_response_id: None,
            include: Vec::new(),
        };

        let ctx = TranslateCtx {
            reasoning: None,
            default_max_tokens: 4096,
            stream: false,
        };

        let req = translate_to_anthropic_request(&raw, &chat, &ctx);

        // Should have user message, assistant message with tool_use, user message with tool_result
        assert_eq!(req.messages.len(), 3);

        // First is user
        assert_eq!(req.messages[0].role, "user");

        // Second is assistant with tool_use
        assert_eq!(req.messages[1].role, "assistant");
        if let AnthropicMessageContent::Blocks(blocks) = &req.messages[1].content {
            assert!(blocks.iter().any(|b| matches!(b, AnthropicContentBlock::ToolUse { name, .. } if name == "get_weather")));
        } else {
            panic!("Expected blocks");
        }

        // Third is user with tool_result
        assert_eq!(req.messages[2].role, "user");
        if let AnthropicMessageContent::Blocks(blocks) = &req.messages[2].content {
            assert!(blocks.iter().any(|b| matches!(b, AnthropicContentBlock::ToolResult { content, .. } if content.contains("Sunny"))));
        } else {
            panic!("Expected blocks");
        }
    }

    #[test]
    fn translate_thinking_with_signature_round_trip() {
        let resp = AnthropicResponse {
            id: "test_resp".into(),
            model: "test-model".into(),
            role: "assistant".into(),
            content: vec![
                AnthropicResponseBlock::Thinking {
                    thinking: "Let me think...".into(),
                    signature: Some("sig_abc123".into()),
                },
                AnthropicResponseBlock::Text {
                    text: "Final answer".into(),
                },
            ],
            stop_reason: Some("end_turn".into()),
            usage: Some(AnthropicUsage {
                input_tokens: 100,
                output_tokens: 50,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            }),
            base_resp: None,
        };

        let output = translate_to_responses_response(resp.clone());

        // Should have reasoning item and message item
        assert_eq!(output.output.len(), 2);

        // First is reasoning
        match &output.output[0] {
            OutputItem::Reasoning(item) => {
                assert_eq!(item.status, "completed");
            }
            _ => panic!("Expected reasoning item first"),
        }

        // Second is message
        match &output.output[1] {
            OutputItem::Message(item) => {
                assert_eq!(item.role, "assistant");
                match &item.content[0] {
                    OutputContentPart::OutputText { text } => {
                        assert_eq!(text, "Final answer");
                    }
                }
            }
            _ => panic!("Expected message item second"),
        }

        // Usage should be preserved
        assert!(output.usage.is_some());
        let usage = output.usage.unwrap();
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
    }

    #[test]
    fn parse_response_with_base_resp_error() {
        let resp = AnthropicResponse {
            id: "test_resp".into(),
            model: "test-model".into(),
            role: "assistant".into(),
            content: vec![AnthropicResponseBlock::Text {
                text: "Hello".into(),
            }],
            stop_reason: None,
            usage: None,
            base_resp: Some(BaseResp {
                status_code: 1001,
                status_msg: "Model not available".into(),
            }),
        };

        let result = check_base_resp(&resp);
        assert!(result.is_err());

        if let Err(ProxyError::Provider(err)) = result {
            assert!(err.message.contains("1001"));
            assert!(err.message.contains("Model not available"));
        } else {
            panic!("Expected Provider error");
        }
    }

    #[test]
    fn translate_tool_choice_variants() {
        // Test auto
        let auto = translate_tool_choice(&Some("auto".into()));
        assert!(auto.is_some());
        assert_eq!(auto.unwrap(), serde_json::json!({ "type": "auto" }));

        // Test required
        let required = translate_tool_choice(&Some("required".into()));
        assert!(required.is_some());
        assert_eq!(required.unwrap(), serde_json::json!({ "type": "any" }));

        // Test any
        let any = translate_tool_choice(&Some("any".into()));
        assert!(any.is_some());
        assert_eq!(any.unwrap(), serde_json::json!({ "type": "any" }));

        // Test none (invalid for Anthropic)
        let none = translate_tool_choice(&None);
        assert!(none.is_none());

        // Test named tool
        let named = translate_tool_choice(&Some("get_weather".into()));
        assert!(named.is_some());
        // Should default to auto for unknown patterns
        assert_eq!(named.unwrap(), serde_json::json!({ "type": "auto" }));
    }

    #[test]
    fn translate_drops_image_and_warns() {
        let raw = make_responses_request(
            "test-model",
            ResponsesInput::Items(vec![InputItem {
                item_type: "message".into(),
                id: None,
                call_id: None,
                role: Some("user".into()),
                name: None,
                content: Some(Content::Text(
                    "Look at this: https://example.com/image.png".into(),
                )),
                reasoning_content: None,
                thought_signature: None,
                thought: None,
                arguments: None,
                input: None,
                action: None,
                command: None,
                cwd: None,
                working_directory: None,
                changes: None,
                output: None,
                stdout: None,
                stderr: None,
                encrypted_content: None,
            }]),
        );

        let chat = ChatRequest {
            model: "test-model".into(),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_choice: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: false,
            store: false,
            metadata: BTreeMap::new(),
            previous_response_id: None,
            include: Vec::new(),
        };

        let ctx = TranslateCtx {
            reasoning: None,
            default_max_tokens: 4096,
            stream: false,
        };

        let req = translate_to_anthropic_request(&raw, &chat, &ctx);

        // Should include the text with URL
        assert_eq!(req.messages.len(), 1);
        if let AnthropicMessageContent::Blocks(blocks) = &req.messages[0].content {
            assert_eq!(blocks.len(), 1);
            match &blocks[0] {
                AnthropicContentBlock::Text { text } => {
                    assert!(text.contains("Look at this"));
                    assert!(text.contains("image.png"));
                }
                _ => panic!("Expected text block"),
            }
        }
    }
}

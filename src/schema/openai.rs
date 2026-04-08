use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,

    #[serde(default)]
    pub input: Option<ResponsesInput>,
    #[serde(default)]
    pub messages: Option<Vec<ChatMessage>>,

    #[serde(default)]
    pub instructions: Option<Instructions>,

    #[serde(default)]
    pub previous_response_id: Option<String>,

    #[serde(default)]
    pub store: Option<bool>,

    #[serde(default)]
    pub metadata: Option<BTreeMap<String, Value>>,

    #[serde(default)]
    pub tools: Option<Vec<Tool>>,

    #[serde(default)]
    pub tool_choice: Option<String>,

    #[serde(default)]
    pub temperature: Option<f64>,

    #[serde(default)]
    pub top_p: Option<f64>,

    #[serde(default)]
    pub max_tokens: Option<u64>,
    #[serde(default)]
    pub max_output_tokens: Option<u64>,

    #[serde(default)]
    pub stream: Option<bool>,

    #[serde(default)]
    pub include: Option<Vec<String>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompactRequest {
    pub input: ResponsesInput,
    pub instructions: Instructions,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Instructions {
    Text(String),
    Parts(Vec<TextPart>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TextPart {
    Text(String),
    Obj { text: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<InputItem>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InputItem {
    #[serde(rename = "type", default)]
    pub item_type: String,

    #[serde(default)]
    pub id: Option<String>,

    #[serde(default)]
    pub call_id: Option<String>,

    #[serde(default)]
    pub role: Option<String>,

    #[serde(default)]
    pub name: Option<String>,

    #[serde(default)]
    pub content: Option<Content>,

    #[serde(default)]
    pub reasoning_content: Option<String>,

    #[serde(default)]
    pub thought_signature: Option<String>,

    #[serde(default)]
    pub thought: Option<String>,

    #[serde(default)]
    pub arguments: Option<Value>,

    #[serde(default)]
    pub input: Option<Value>,

    #[serde(default)]
    pub action: Option<Value>,

    #[serde(default)]
    pub command: Option<String>,

    #[serde(default)]
    pub cwd: Option<String>,

    #[serde(default)]
    pub working_directory: Option<String>,

    #[serde(default)]
    pub changes: Option<Vec<FileChange>>,

    #[serde(default)]
    pub output: Option<Content>,

    #[serde(default)]
    pub stdout: Option<Content>,

    #[serde(default)]
    pub stderr: Option<String>,

    // OpenAI compaction artifact
    #[serde(default)]
    pub encrypted_content: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileChange {
    pub path: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
    Json(Value),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type", default)]
    pub part_type: String,

    #[serde(default)]
    pub text: Option<String>,

    #[serde(default)]
    pub image_url: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,

    #[serde(default)]
    pub function: Option<FunctionDef>,

    // Flat function schema (non-wrapped)
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<Value>,
    #[serde(default)]
    pub strict: Option<bool>,

    // web_search config (OpenAI style)
    #[serde(default)]
    pub web_search: Option<WebSearchConfig>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WebSearchConfig {
    #[serde(default)]
    pub enable: Option<bool>,
    #[serde(default)]
    pub search_engine: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub tools: Vec<Tool>,
    #[serde(default)]
    pub tool_choice: Option<String>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<u64>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub store: bool,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
    #[serde(default)]
    pub previous_response_id: Option<String>,

    #[serde(default)]
    pub include: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<ChatContent>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub thought_signature: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatContent {
    Text(String),
    Parts(Vec<ChatContentPart>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatContentPart {
    #[serde(rename = "type", default)]
    pub part_type: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub image_url: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ToolCallFunction,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

pub fn messages_to_input_items(messages: &[ChatMessage]) -> Vec<InputItem> {
    let mut out = Vec::new();
    for message in messages {
        let role = if message.role == "developer" {
            "system".to_string()
        } else {
            message.role.clone()
        };

        if role == "tool" {
            out.push(InputItem {
                item_type: "function_call_output".into(),
                id: None,
                call_id: message.tool_call_id.clone(),
                role: None,
                name: message.name.clone(),
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
                output: message
                    .content
                    .as_ref()
                    .map(|content| chat_content_to_content(content)),
                stdout: None,
                stderr: None,
                encrypted_content: None,
            });
            continue;
        }

        out.push(InputItem {
            item_type: "message".into(),
            id: None,
            call_id: None,
            role: Some(role),
            name: message.name.clone(),
            content: message
                .content
                .as_ref()
                .map(|content| chat_content_to_content(content)),
            reasoning_content: message.reasoning_content.clone(),
            thought_signature: message.thought_signature.clone(),
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
        });

        for tool_call in &message.tool_calls {
            out.push(InputItem {
                item_type: "function_call".into(),
                id: Some(tool_call.id.clone()),
                call_id: None,
                role: None,
                name: Some(tool_call.function.name.clone()),
                content: None,
                reasoning_content: None,
                thought_signature: message.thought_signature.clone(),
                thought: None,
                arguments: Some(Value::String(tool_call.function.arguments.clone())),
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
            });
        }
    }
    out
}

fn chat_content_to_content(content: &ChatContent) -> Content {
    match content {
        ChatContent::Text(text) => Content::Text(text.clone()),
        ChatContent::Parts(parts) => Content::Parts(
            parts
                .iter()
                .map(|part| ContentPart {
                    part_type: part.part_type.clone(),
                    text: part.text.clone(),
                    image_url: part.image_url.clone(),
                })
                .collect(),
        ),
    }
}

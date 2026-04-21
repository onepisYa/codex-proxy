use bytes::Bytes;
use futures::StreamExt;
use futures::stream::Stream;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::pin::Pin;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{debug, warn};

use crate::providers::minimax_wire::store_call_id_mapping;
use crate::schema::openai::ChatRequest;
use crate::schema::sse::{
    FailedResponseObject, FunctionCallItem, MessageItem, OutputContentPart, OutputItem,
    ReasoningItem, ResponseCompletedData, ResponseCreatedData, ResponseError, ResponseEvent,
    ResponseFailedData, ResponseObject, ResponseOutputItemAddedData, ResponseOutputItemDoneData,
    ResponseOutputTextDeltaData, Usage,
};

pub fn generate_tool_id() -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("tool_{}", timestamp)
}

pub fn stream_responses_sse(
    byte_stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
    model: &str,
    created_ts: i64,
    request: &ChatRequest,
    idle_timeout_seconds: u64,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>> {
    let model = model.to_string();
    let resp_id = format!("resp_{created_ts}");
    let req = request.clone();
    let idle_timeout_seconds = idle_timeout_seconds.max(1);

    Box::pin(async_stream::stream! {
        let mut seq_num: u64 = 0;
        let mut stream_state = StreamState {
            resp_id: resp_id.clone(),
            final_usage: None,
            blocks: HashMap::new(),
            next_output_index: 0,
            closed_items: Vec::new(),
        };

        debug!("SSE STREAM: emitting response.created");
        yield Ok(encode_event(
            &mut seq_num,
            "response.created",
            ResponseCreatedData {
                response: response_in_progress(&resp_id, &model, created_ts, &req),
            },
        ));

        let mut byte_stream = std::pin::pin!(byte_stream);
        let mut line_buf = String::new();
        let mut saw_done = false;
        let mut error_occurred = false;

        loop {
            let next_chunk = match tokio::time::timeout(
                Duration::from_secs(idle_timeout_seconds),
                byte_stream.next(),
            ).await {
                Ok(v) => v,
                Err(_) => {
                    debug!("SSE STREAM: idle timeout after {}s", idle_timeout_seconds);
                    let msg = format!("Upstream stream idle for {idle_timeout_seconds}s");
                    yield Ok(encode_event(
                        &mut seq_num,
                        "response.failed",
                        ResponseFailedData {
                            response: response_failed(&resp_id, &model, created_ts, &req, "stream_timeout", &msg),
                        },
                    ));
                    return;
                }
            };

            let Some(chunk_result) = next_chunk else {
                debug!("SSE STREAM: next_chunk is None (stream ended)");
                break;
            };

            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    debug!("SSE STREAM: stream error: {}", e);
                    let msg = format!("Upstream stream error: {e}");
                    debug!("{msg}");
                    yield Ok(encode_event(
                        &mut seq_num,
                        "response.failed",
                        ResponseFailedData {
                            response: response_failed(&resp_id, &model, created_ts, &req, "stream_error", &msg),
                        },
                    ));
                    return;
                }
            };

            let text = String::from_utf8_lossy(&chunk);
            line_buf.push_str(&text);

            if line_buf.trim_end_matches('\r') == "data: [DONE]" {
                #[allow(unused_assignments)]
                {
                    saw_done = true;
                }
                break;
            }

            while let Some(nl) = line_buf.find('\n') {
                let line = line_buf[..nl].trim_end_matches('\r').to_string();
                line_buf = line_buf[nl + 1..].to_string();

                if !line.starts_with("data: ") {
                    continue;
                }
                if line == "data: [DONE]" {
                    saw_done = true;
                    break;
                }

                let data_str = &line[6..];
                let chunk: MinimaxStreamChunk = match serde_json::from_str(data_str) {
                    Ok(d) => d,
                    Err(e) => {
                        warn!("Minimax chunk parse failed: {e}, raw: {data_str}");
                        continue;
                    }
                };
                debug!("SSE STREAM CHUNK: {} - {:?}", data_str, chunk);

                match chunk {
                    MinimaxStreamChunk::MessageStart(_) => {}
                    MinimaxStreamChunk::Ping(_) => {}
                    MinimaxStreamChunk::ContentBlockStart(block_start) => {
                        debug!("SSE STREAM: ContentBlockStart block_type={}", block_start.content_block.block_type);
                        if let Some(bytes) = stream_state.handle_content_block_start(&block_start, &mut seq_num) {
                            yield Ok(bytes);
                        }
                    }
                    MinimaxStreamChunk::ContentBlockDelta(delta) => {
                        debug!("SSE STREAM: ContentBlockDelta index={:?}", delta.index);
                        if let Some(bytes) = stream_state.handle_content_block_delta(&delta, &mut seq_num) {
                            yield Ok(bytes);
                        }
                    }
                    MinimaxStreamChunk::ContentBlockStop(block_stop) => {
                        if let Some(bytes) = stream_state.handle_content_block_stop(&block_stop, &mut seq_num) {
                            yield Ok(bytes);
                        }
                    }
                    MinimaxStreamChunk::MessageDelta(msg_delta) => {
                        stream_state.handle_message_delta(&msg_delta);
                    }
                    MinimaxStreamChunk::MessageStop(_) => {
                        saw_done = true;
                        break;
                    }
                    MinimaxStreamChunk::ErrorStream(error) => {
                        let msg = error.error.message.clone();
                        yield Ok(encode_event(
                            &mut seq_num,
                            "response.failed",
                            ResponseFailedData {
                                response: response_failed(&resp_id, &model, created_ts, &req, "upstream_error", &msg),
                            },
                        ));
                        error_occurred = true;
                        break;
                    }
                }
            }

            if saw_done || error_occurred {
                debug!("SSE STREAM: breaking loop, saw_done={}, error_occurred={}", saw_done, error_occurred);
                break;
            }
        }

        debug!("SSE STREAM: loop exited, error_occurred={}, blocks.len={}", error_occurred, stream_state.blocks.len());

        if !error_occurred {
            debug!("SSE STREAM: calling finalize_all, blocks.len={}", stream_state.blocks.len());
            for bytes in stream_state.finalize_all(&mut seq_num) {
                yield Ok(bytes);
            }

            let mut final_resp = response_in_progress(&resp_id, &model, created_ts, &req);
            final_resp.status = "completed".into();
            final_resp.completed_at = Some(now_secs());
            final_resp.usage = stream_state.final_usage.clone();
            final_resp.output = stream_state.closed_items.clone();

            debug!("SSE STREAM: emitting response.completed");
            yield Ok(encode_event(
                &mut seq_num,
                "response.completed",
                ResponseCompletedData { response: final_resp },
            ));
        }
    })
}

struct StreamState {
    resp_id: String,
    final_usage: Option<Usage>,
    blocks: HashMap<usize, BlockState>,
    next_output_index: usize,
    closed_items: Vec<OutputItem>,
}

enum BlockState {
    Message {
        output_index: usize,
        item_id: String,
        text: String,
    },
    ToolCall {
        output_index: usize,
        item_id: String,
        call_id: String,
        name: String,
        args: String,
    },
    Reasoning {
        output_index: usize,
        item_id: String,
        content_text: String,
    },
}

impl StreamState {
    fn handle_content_block_start(
        &mut self,
        block_start: &ContentBlockStartChunk,
        seq_num: &mut u64,
    ) -> Option<Bytes> {
        let index = block_start.index.unwrap_or(0) as usize;
        let block_type = &block_start.content_block.block_type;

        match block_type.as_str() {
            "text" => {
                let output_index = self.next_output_index;
                self.next_output_index += 1;
                let item_id = format!("msg_{}_{}", now_ms(), output_index);

                self.blocks.insert(
                    index,
                    BlockState::Message {
                        output_index,
                        item_id: item_id.clone(),
                        text: String::new(),
                    },
                );

                let item = OutputItem::Message(MessageItem {
                    id: item_id.clone(),
                    role: "assistant",
                    status: "in_progress".into(),
                    content: vec![OutputContentPart::OutputText {
                        text: String::new(),
                    }],
                });

                Some(make_output_item_added(
                    seq_num,
                    &self.resp_id,
                    output_index,
                    item,
                ))
            }
            "tool_use" => {
                let output_index = self.next_output_index;
                self.next_output_index += 1;
                let item_id = format!("fc_{}_{}", now_ms(), output_index);
                // Use MiniMax's original tool_use.id as call_id for our FunctionCallItem
                let call_id = block_start.content_block.id.clone()
                    .unwrap_or_else(|| format!("call_{}", now_ms()));

                tracing::debug!("MiniMax tool_use id from SSE: {:?}, using call_id: {:?}", block_start.content_block.id, call_id);

                // Store the mapping: MiniMax's call_id -> our tool_use.id (by position)
                // This allows us to look up our original ID when processing function_call_output
                store_call_id_mapping(&call_id);

                let name = block_start.content_block.name.clone().unwrap_or_default();

                self.blocks.insert(
                    index,
                    BlockState::ToolCall {
                        output_index,
                        item_id: item_id.clone(),
                        call_id: call_id.clone(),
                        name,
                        args: String::new(),
                    },
                );

                let item = OutputItem::FunctionCall(FunctionCallItem {
                    id: item_id.clone(),
                    status: "in_progress".into(),
                    name: String::new(),
                    arguments: String::new(),
                    call_id: self
                        .blocks
                        .get(&index)
                        .and_then(|b| {
                            if let BlockState::ToolCall { call_id, .. } = b {
                                Some(call_id.clone())
                            } else {
                                None
                            }
                        })
                        .unwrap_or_default(),
                    thought_signature: None,
                });

                Some(make_output_item_added(
                    seq_num,
                    &self.resp_id,
                    output_index,
                    item,
                ))
            }
            "thinking" => {
                let output_index = self.next_output_index;
                self.next_output_index += 1;
                let item_id = format!("reason_{}_{}", now_ms(), output_index);

                self.blocks.insert(
                    index,
                    BlockState::Reasoning {
                        output_index,
                        item_id: item_id.clone(),
                        content_text: String::new(),
                    },
                );

                let item = OutputItem::Reasoning(ReasoningItem {
                    id: item_id.clone(),
                    status: "in_progress".into(),
                    summary: Vec::new(),
                    content: Vec::new(),
                });

                Some(make_output_item_added(
                    seq_num,
                    &self.resp_id,
                    output_index,
                    item,
                ))
            }
            _ => {
                warn!("Unknown block type: {}", block_type);
                None
            }
        }
    }

    fn handle_content_block_delta(
        &mut self,
        chunk: &ContentBlockDeltaChunk,
        seq_num: &mut u64,
    ) -> Option<Bytes> {
        let index = chunk.index.unwrap_or(0) as usize;
        let d = &chunk.delta;

        if let Some(state) = self.blocks.get_mut(&index) {
            match state {
                BlockState::Message {
                    output_index,
                    item_id,
                    text,
                } => {
                    if let Some(text_delta) = &d.text_delta {
                        text.push_str(text_delta);
                        return Some(make_text_delta(
                            seq_num,
                            &self.resp_id,
                            item_id,
                            *output_index as i64,
                            text_delta,
                        ));
                    }
                }
                BlockState::ToolCall {
                    output_index: _,
                    item_id: _,
                    call_id: _,
                    name,
                    args,
                } => {
                    if let Some(input_json_delta) = &d.input_json_delta {
                        args.push_str(input_json_delta);
                    }
                    if let Some(content_block_start) = &d.content_block_start {
                        if let Some(name_str) = &content_block_start.name {
                            *name = name_str.clone();
                        }
                    }
                }
                BlockState::Reasoning {
                    output_index,
                    item_id,
                    content_text,
                } => {
                    if let Some(thinking_delta) = &d.thinking_delta {
                        content_text.push_str(thinking_delta);
                        return Some(make_reasoning_delta(
                            seq_num,
                            &self.resp_id,
                            item_id,
                            *output_index as i64,
                            thinking_delta,
                        ));
                    }
                    if let Some(sig_delta) = &d.signature_delta {
                        // Signature delta for reasoning - accumulate but don't emit separate event
                        content_text.push_str(sig_delta);
                    }
                }
            }
        }
        None
    }

    fn handle_content_block_stop(
        &mut self,
        block_stop: &ContentBlockStopChunk,
        seq_num: &mut u64,
    ) -> Option<Bytes> {
        let index = block_stop.index.unwrap_or(0) as usize;

        if let Some((output_index, item)) = self.blocks.remove(&index).map(|state| {
            let out_idx = match &state {
                BlockState::Message { output_index, .. } => *output_index,
                BlockState::ToolCall { output_index, .. } => *output_index,
                BlockState::Reasoning { output_index, .. } => *output_index,
            };
            let item = self.finalize_block(state);
            (out_idx, item)
        }) {
            self.closed_items.push(item.clone());
            Some(make_output_item_done(
                seq_num,
                &self.resp_id,
                output_index,
                item,
            ))
        } else {
            None
        }
    }

    fn handle_message_delta(&mut self, msg_delta: &MessageDeltaChunk) {
        if let Some(usage) = &msg_delta.usage {
            let it = usage.input_tokens.unwrap_or(0);
            let ot = usage.output_tokens.unwrap_or(0);
            self.final_usage = Some(Usage {
                input_tokens: it,
                output_tokens: ot,
                total_tokens: usage.total_tokens.unwrap_or(it + ot),
                input_tokens_details: usage.input_tokens_details.as_ref().map(|d| {
                    crate::schema::sse::InputTokensDetails {
                        cached_tokens: d.cached_tokens.unwrap_or(0),
                    }
                }),
                output_tokens_details: usage.output_tokens_details.as_ref().map(|d| {
                    crate::schema::sse::OutputTokensDetails {
                        reasoning_tokens: d.reasoning_tokens.unwrap_or(0),
                    }
                }),
            });
        }
    }

    fn finalize_all(&mut self, seq_num: &mut u64) -> Vec<Bytes> {
        let remaining: Vec<(usize, BlockState)> = self.blocks.drain().collect();
        remaining
            .into_iter()
            .map(|(_idx, state)| {
                let output_index = match &state {
                    BlockState::Message { output_index, .. } => *output_index,
                    BlockState::ToolCall { output_index, .. } => *output_index,
                    BlockState::Reasoning { output_index, .. } => *output_index,
                };
                let item = self.finalize_block(state);
                self.closed_items.push(item.clone());
                make_output_item_done(seq_num, &self.resp_id, output_index, item)
            })
            .collect()
    }

    fn finalize_block(&self, state: BlockState) -> OutputItem {
        match state {
            BlockState::Message { item_id, text, .. } => OutputItem::Message(MessageItem {
                id: item_id,
                role: "assistant",
                status: "completed".into(),
                content: vec![OutputContentPart::OutputText { text }],
            }),
            BlockState::ToolCall {
                item_id,
                call_id,
                name,
                args,
                ..
            } => {
                let final_call_id = if call_id.is_empty() {
                    format!("call_{}", now_ms())
                } else {
                    call_id
                };
                OutputItem::FunctionCall(FunctionCallItem {
                    id: item_id,
                    status: "completed".into(),
                    name,
                    arguments: args,
                    call_id: final_call_id,
                    thought_signature: None,
                })
            }
            BlockState::Reasoning {
                item_id,
                content_text,
                ..
            } => OutputItem::Reasoning(ReasoningItem {
                id: item_id,
                status: "completed".into(),
                summary: Vec::new(),
                content: vec![crate::schema::sse::ReasoningContentPart::ReasoningText {
                    text: content_text,
                }],
            }),
        }
    }
}

fn make_output_item_added(
    seq_num: &mut u64,
    resp_id: &str,
    output_index: usize,
    item: OutputItem,
) -> Bytes {
    encode_event(
        seq_num,
        "response.output_item.added",
        ResponseOutputItemAddedData {
            response_id: resp_id.to_string(),
            output_index,
            item,
        },
    )
}

fn make_output_item_done(
    seq_num: &mut u64,
    resp_id: &str,
    output_index: usize,
    item: OutputItem,
) -> Bytes {
    encode_event(
        seq_num,
        "response.output_item.done",
        ResponseOutputItemDoneData {
            response_id: resp_id.to_string(),
            output_index,
            item,
        },
    )
}

fn make_text_delta(
    seq_num: &mut u64,
    resp_id: &str,
    item_id: &str,
    output_index: i64,
    delta: &str,
) -> Bytes {
    encode_event(
        seq_num,
        "response.output_text.delta",
        ResponseOutputTextDeltaData {
            response_id: resp_id.to_string(),
            item_id: item_id.to_string(),
            output_index,
            content_index: 0,
            delta: delta.to_string(),
        },
    )
}

fn make_reasoning_delta(
    seq_num: &mut u64,
    resp_id: &str,
    item_id: &str,
    output_index: i64,
    delta: &str,
) -> Bytes {
    encode_event(
        seq_num,
        "response.output_text.delta",
        ResponseOutputTextDeltaData {
            response_id: resp_id.to_string(),
            item_id: item_id.to_string(),
            output_index,
            content_index: 0,
            delta: delta.to_string(),
        },
    )
}

// === Minimax Stream Chunk types ===

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type")]
pub enum MinimaxStreamChunk {
    #[serde(rename = "message_start")]
    MessageStart(MessageStartChunk),
    #[serde(rename = "ping")]
    Ping(PingChunk),
    #[serde(rename = "content_block_start")]
    ContentBlockStart(ContentBlockStartChunk),
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta(ContentBlockDeltaChunk),
    #[serde(rename = "content_block_stop")]
    ContentBlockStop(ContentBlockStopChunk),
    #[serde(rename = "message_delta")]
    MessageDelta(MessageDeltaChunk),
    #[serde(rename = "message_stop")]
    MessageStop(MessageStopChunk),
    #[serde(rename = "error")]
    ErrorStream(ErrorStreamChunk),
}

#[derive(Clone, Debug, Deserialize)]
pub struct MessageStartChunk {
    #[serde(default)]
    pub message: serde_json::Value,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PingChunk;

#[derive(Clone, Debug, Deserialize)]
pub struct ContentBlockStartChunk {
    #[serde(default)]
    pub index: Option<u64>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub content_block: ContentBlockDef,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct ContentBlockDef {
    #[serde(rename = "type")]
    pub block_type: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub id: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ContentBlockDeltaChunk {
    #[serde(default)]
    pub index: Option<u64>,
    #[serde(default)]
    pub delta: ContentBlockDeltaFields,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct ContentBlockDeltaFields {
    #[serde(default, rename = "text")]
    pub text_delta: Option<String>,
    #[serde(default, rename = "input_json_delta", alias = "input_json", alias = "partial_json")]
    pub input_json_delta: Option<String>,
    #[serde(default)]
    pub thinking_delta: Option<String>,
    #[serde(default)]
    pub signature_delta: Option<String>,
    #[serde(default)]
    pub content_block_start: Option<ContentBlockDeltaName>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ContentBlockDeltaName {
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ContentBlockStopChunk {
    #[serde(default)]
    pub index: Option<u64>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct MessageDeltaChunk {
    #[serde(default)]
    pub usage: Option<MinimaxUsage>,
    #[serde(default)]
    pub delta: Option<MessageDeltaDetail>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct MessageDeltaDetail {
    #[serde(default)]
    pub stop_reason: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct MinimaxUsage {
    #[serde(default)]
    pub input_tokens: Option<u64>,
    #[serde(default)]
    pub output_tokens: Option<u64>,
    #[serde(default)]
    pub total_tokens: Option<u64>,
    #[serde(default)]
    pub input_tokens_details: Option<InputTokensDetailsChunk>,
    #[serde(default)]
    pub output_tokens_details: Option<OutputTokensDetailsChunk>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct InputTokensDetailsChunk {
    #[serde(default)]
    pub cached_tokens: Option<u64>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct OutputTokensDetailsChunk {
    #[serde(default)]
    pub reasoning_tokens: Option<u64>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct MessageStopChunk;

#[derive(Clone, Debug, Deserialize)]
pub struct ErrorStreamChunk {
    pub error: ErrorDetail,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ErrorDetail {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
    pub code: Option<String>,
}

// === Helper functions ===

fn response_in_progress(
    resp_id: &str,
    model: &str,
    created_ts: i64,
    req: &ChatRequest,
) -> ResponseObject {
    ResponseObject {
        id: resp_id.to_string(),
        object: "response",
        created_at: created_ts,
        completed_at: None,
        model: model.to_string(),
        status: "in_progress".into(),
        temperature: req.temperature.unwrap_or(1.0),
        top_p: req.top_p.unwrap_or(1.0),
        tool_choice: req.tool_choice.clone().unwrap_or_else(|| "auto".into()),
        tools: req.tools.clone(),
        parallel_tool_calls: true,
        store: req.store,
        metadata: req.metadata.clone(),
        output: Vec::new(),
        usage: None,
    }
}

fn response_failed(
    resp_id: &str,
    model: &str,
    created_ts: i64,
    req: &ChatRequest,
    code: &str,
    message: &str,
) -> FailedResponseObject {
    FailedResponseObject {
        id: resp_id.to_string(),
        object: "response",
        created_at: created_ts,
        status: "failed",
        model: model.to_string(),
        error: ResponseError {
            code: code.to_string(),
            message: message.to_string(),
        },
        metadata: req.metadata.clone(),
    }
}

fn encode_event<T: Serialize>(seq_num: &mut u64, evt_type: &'static str, data: T) -> Bytes {
    *seq_num += 1;
    let evt = ResponseEvent {
        id: format!("evt_{}_{}", now_ms(), seq_num),
        object: "response.event",
        event_type: evt_type,
        created_at: now_secs(),
        sequence_number: *seq_num,
        data,
    };
    let payload = format!(
        "event: {}\ndata: {}\n\n",
        evt_type,
        serde_json::to_string(&evt).unwrap_or_default()
    );
    Bytes::from(payload)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// === Tests ===

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    fn make_test_request() -> ChatRequest {
        ChatRequest {
            model: "minimax-model".into(),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_choice: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: true,
            store: false,
            metadata: std::collections::BTreeMap::new(),
            previous_response_id: None,
            include: Vec::new(),
        }
    }

    fn make_chunk(event_type: &str, extra: serde_json::Value) -> Bytes {
        let mut obj = serde_json::json!({
            "type": event_type,
        });
        if let serde_json::Value::Object(ref mut m) = obj {
            if let serde_json::Value::Object(ref extra_obj) = extra {
                for (k, v) in extra_obj {
                    m.insert(k.clone(), v.clone());
                }
            }
        }
        // Build full SSE format with both event: and data: lines
        let json_data = serde_json::to_string(&obj).unwrap();
        let payload = format!("event: {}\ndata: {}\n\n", event_type, json_data);
        Bytes::from(payload)
    }

    fn parse_event_types_static(data: &str) -> Vec<String> {
        let mut types: Vec<String> = Vec::new();
        for line in data.lines() {
            if line.starts_with("event: ") {
                types.push(line["event: ".len()..].trim().to_string());
            }
        }
        types
    }

    fn collect_event_types(events: Vec<Result<Bytes, std::io::Error>>) -> Vec<String> {
        let mut all_types: Vec<String> = Vec::new();
        for event in events {
            if let Ok(bytes) = event {
                let text = String::from_utf8_lossy(&bytes);
                let types = parse_event_types_static(&text);
                all_types.extend(types);
            }
        }
        all_types
    }

    #[tokio::test]
    async fn stream_text_only_delta_flow() {
        let chunks = vec![
            Ok(make_chunk("message_start", serde_json::json!({}))),
            Ok(make_chunk(
                "content_block_start",
                serde_json::json!({
                    "index": 0,
                    "content_block": {"type": "text"}
                }),
            )),
            Ok(make_chunk(
                "content_block_delta",
                serde_json::json!({
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Hello"}
                }),
            )),
            Ok(make_chunk(
                "content_block_delta",
                serde_json::json!({
                    "index": 0,
                    "delta": {"type": "text_delta", "text": " world"}
                }),
            )),
            Ok(make_chunk(
                "content_block_stop",
                serde_json::json!({
                    "index": 0
                }),
            )),
            Ok(make_chunk(
                "message_delta",
                serde_json::json!({
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5
                    }
                }),
            )),
            Ok(make_chunk("message_stop", serde_json::json!({}))),
        ];

        let stream = futures::stream::iter(chunks);
        let created_ts = now_secs();
        let model = "test-model";
        let request = make_test_request();

        let result_stream = stream_responses_sse(stream, model, created_ts, &request, 60);
        let events: Vec<_> = result_stream.collect().await;

        eprintln!("TOTAL_EVENTS: {}", events.len());
        for (i, event) in events.iter().enumerate() {
            match event {
                Ok(bytes) => {
                    let s = String::from_utf8_lossy(bytes);
                    eprintln!("EVENT_{}: {}", i, &s[..s.len().min(300)]);
                }
                Err(e) => eprintln!("EVENT_{}_ERR: {:?}", i, e),
            }
        }

        let event_types = collect_event_types(events);
        eprintln!("TOTAL_TYPES: {}", event_types.len());
        for (i, t) in event_types.iter().enumerate() {
            eprintln!("TYPE_{}: {}", i, t);
        }

        assert!(event_types.iter().any(|t| t == "response.created"));
        assert!(
            event_types
                .iter()
                .any(|t| t == "response.output_item.added")
        );
        assert!(
            event_types
                .iter()
                .any(|t| t == "response.output_text.delta")
        );
        assert!(event_types.iter().any(|t| t == "response.output_item.done"));
        assert!(event_types.iter().any(|t| t == "response.completed"));
    }

    #[tokio::test]
    async fn stream_tool_use_args_accumulate() {
        let chunks = vec![
            Ok(make_chunk("message_start", serde_json::json!({}))),
            Ok(make_chunk(
                "content_block_start",
                serde_json::json!({
                    "index": 0,
                    "content_block": {"type": "tool_use", "name": "get_weather"}
                }),
            )),
            Ok(make_chunk(
                "content_block_delta",
                serde_json::json!({
                    "index": 0,
                    "delta": {"type": "input_json_delta", "input_json": "{\"city\":"}
                }),
            )),
            Ok(make_chunk(
                "content_block_delta",
                serde_json::json!({
                    "index": 0,
                    "delta": {"type": "input_json_delta", "input_json": "\"Tokyo\"}"}
                }),
            )),
            Ok(make_chunk(
                "content_block_stop",
                serde_json::json!({
                    "index": 0
                }),
            )),
            Ok(make_chunk(
                "message_delta",
                serde_json::json!({
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 20
                    }
                }),
            )),
            Ok(make_chunk("message_stop", serde_json::json!({}))),
        ];

        let stream = futures::stream::iter(chunks);
        let created_ts = now_secs();
        let model = "test-model";
        let request = make_test_request();

        let result_stream = stream_responses_sse(stream, model, created_ts, &request, 60);
        let events: Vec<_> = result_stream.collect().await;

        let event_types = collect_event_types(events);

        assert!(
            event_types
                .iter()
                .any(|t| t == "response.output_item.added")
        );
        assert!(event_types.iter().any(|t| t == "response.output_item.done"));
        assert!(event_types.iter().any(|t| t == "response.completed"));
    }

    #[tokio::test]
    async fn stream_thinking_delta_maps_to_reasoning() {
        let chunks = vec![
            Ok(make_chunk("message_start", serde_json::json!({}))),
            Ok(make_chunk(
                "content_block_start",
                serde_json::json!({
                    "index": 0,
                    "content_block": {"type": "thinking"}
                }),
            )),
            Ok(make_chunk(
                "content_block_delta",
                serde_json::json!({
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "Let me think about this..."}
                }),
            )),
            Ok(make_chunk(
                "content_block_stop",
                serde_json::json!({
                    "index": 0
                }),
            )),
            Ok(make_chunk(
                "message_delta",
                serde_json::json!({
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 15
                    }
                }),
            )),
            Ok(make_chunk("message_stop", serde_json::json!({}))),
        ];

        let stream = futures::stream::iter(chunks);
        let created_ts = now_secs();
        let model = "test-model";
        let request = make_test_request();

        let result_stream = stream_responses_sse(stream, model, created_ts, &request, 60);
        let events: Vec<_> = result_stream.collect().await;

        let event_types = collect_event_types(events);

        assert!(
            event_types
                .iter()
                .any(|t| t == "response.output_item.added")
        );
        assert!(event_types.iter().any(|t| t == "response.output_item.done"));
        assert!(event_types.iter().any(|t| t == "response.completed"));
    }

    #[tokio::test]
    async fn stream_idle_timeout_emits_failed() {
        let stream = futures::stream::pending::<Result<Bytes, reqwest::Error>>();
        let created_ts = now_secs();
        let model = "test-model";
        let request = make_test_request();

        let result_stream = stream_responses_sse(stream, model, created_ts, &request, 1);
        let events: Vec<_> = result_stream.collect().await;

        let event_types = collect_event_types(events);

        assert!(event_types.iter().any(|t| t == "response.created"));
        assert!(event_types.iter().any(|t| t == "response.failed"));
    }
}

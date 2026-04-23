use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use std::collections::HashMap;

use crate::config::with_config;
use crate::error::{ProviderError, ProxyError};
use crate::providers::base::{Provider, ProviderExecutionContext};
use crate::providers::minimax_session::TranslationSession;
use crate::providers::minimax_stream::stream_responses_sse;
use crate::providers::minimax_wire::{
    AnthropicResponse, AnthropicResponseBlock, TranslateCtx,
    check_base_resp, translate_to_anthropic_request, translate_to_responses_response,
};
use crate::schema::openai::{ChatRequest, ResponsesRequest};
use crate::schema::sse::{MessageItem, OutputContentPart, OutputItem, ResponseObject, Usage};
use fp_agent::normalize::normalize_responses_request;
use reqwest::header::AUTHORIZATION;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, error, info_span, Instrument};

pub struct ZAIProvider {
    client: reqwest::Client,
}

impl Default for ZAIProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl ZAIProvider {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

impl Clone for ZAIProvider {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
        }
    }
}

impl Provider for ZAIProvider {
    fn handle_request(
        &self,
        raw_request: ResponsesRequest,
        normalized_request: ChatRequest,
        _headers: HeaderMap,
        context: ProviderExecutionContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Response<Body>, ProxyError>> + Send + '_>,
    > {
        Box::pin(self.execute_request(raw_request, normalized_request, context))
    }

    fn handle_compact(
        &self,
        data: crate::schema::openai::CompactRequest,
        _headers: HeaderMap,
        context: ProviderExecutionContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Response<Body>, ProxyError>> + Send + '_>,
    > {
        Box::pin(self.execute_compact(data, context))
    }

    fn clone_box(&self) -> Box<dyn Provider + Send + Sync> {
        Box::new(self.clone())
    }
}

impl ZAIProvider {
    async fn execute_request(
        &self,
        raw: ResponsesRequest,
        chat: ChatRequest,
        context: ProviderExecutionContext,
    ) -> Result<Response<Body>, ProxyError> {
        let span = info_span!(
            "zai_request",
            model = %context.upstream_model(),
            stream = %chat.stream,
            account_id = %context.account.id
        );

        async move {
            let auth_value = resolve_zai_auth(&context)?;

            let base_url = resolve_endpoint_url(&context)?;
            let url = build_messages_url(&base_url);

            let is_stream = chat.stream;

            let default_max_tokens = resolve_default_max_tokens(&context)?;

            let ctx = TranslateCtx {
                reasoning: context
                    .reasoning()
                    .cloned()
                    .map(|r| crate::config::ReasoningConfig {
                        effort_levels: r
                            .preset
                            .as_ref()
                            .map(|p| {
                                let mut m = HashMap::new();
                                m.insert(
                                    p.clone(),
                                    crate::config::EffortLevel {
                                        budget: r.budget,
                                        level: r.level.clone(),
                                    },
                                );
                                m
                            })
                            .unwrap_or_default(),
                        default_effort: r.preset.clone(),
                    }),
                default_max_tokens,
                stream: is_stream,
            };

            let session = TranslationSession::new();

            let anthropic_req = translate_to_anthropic_request(&raw, &chat, &ctx);

            let mut headers = HeaderMap::new();
            headers.insert(AUTHORIZATION, auth_value);
            headers.insert(
                "Content-Type",
                "application/json"
                    .parse()
                    .map_err(|_| ProxyError::Internal("Invalid content type".into()))?,
            );

            debug!("Z.AI request to {}", url);
            debug!("Z.AI request body: {:?}", anthropic_req);

            let resp = self
                .client
                .post(&url)
                .headers(headers)
                .json(&anthropic_req)
                .send()
                .await
                .map_err(|e| {
                    error!("Z.AI HTTP request failed: {}", e);
                    ProxyError::Http(e)
                })?;

            let status = resp.status();
            debug!("Z.AI response status: {}", status);

            if is_stream {
                self.handle_stream_response(resp, &chat, &context, session)
                    .await
            } else {
                self.handle_sync_response(resp, context.upstream_model())
                    .await
            }
        }
        .instrument(span)
        .await
    }

    async fn handle_sync_response(
        &self,
        resp: reqwest::Response,
        model: &str,
    ) -> Result<Response<Body>, ProxyError> {
        let status = resp.status();

        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            error!("Z.AI sync error: {} - {}", status, body);
            if status == reqwest::StatusCode::UNAUTHORIZED
                || status == reqwest::StatusCode::FORBIDDEN
            {
                return Err(ProxyError::Auth(format!(
                    "Z.AI request unauthorized ({}). Body: {}",
                    status, body
                )));
            }
            return Err(ProxyError::Provider(ProviderError::new(
                Some(status),
                format!("Z.AI error: {} - {}", status, body),
            )));
        }

        let bytes = resp.bytes().await.map_err(|e| ProxyError::Http(e))?;

        let anthropic_resp: AnthropicResponse = serde_json::from_slice(&bytes).map_err(|e| {
            ProxyError::Internal(format!("Failed to parse Z.AI response: {}", e))
        })?;

        check_base_resp(&anthropic_resp, model)?;

        let responses_resp = translate_to_responses_response(anthropic_resp);

        let body = serde_json::to_string(&responses_resp)
            .map_err(|e| ProxyError::Internal(format!("Failed to serialize response: {}", e)))?;

        Ok(Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "application/json")
            .body(Body::from(body))
            .map_err(|e| ProxyError::Internal(e.to_string()))?)
    }

    async fn handle_stream_response(
        &self,
        resp: reqwest::Response,
        chat: &ChatRequest,
        context: &ProviderExecutionContext,
        session: TranslationSession,
    ) -> Result<Response<Body>, ProxyError> {
        let status = resp.status();

        if !status.is_success() {
            let body_bytes = resp.bytes().await.unwrap_or_default();
            let body_str = String::from_utf8_lossy(&body_bytes);
            error!("Z.AI stream error: {} - {}", status, body_str);
            if status == reqwest::StatusCode::UNAUTHORIZED
                || status == reqwest::StatusCode::FORBIDDEN
            {
                return Err(ProxyError::Auth(format!(
                    "Z.AI stream request unauthorized ({}). Body: {}",
                    status, body_str
                )));
            }
            return Err(ProxyError::Provider(ProviderError::new(
                Some(status),
                format!("Z.AI error: {} - {}", status, body_str),
            )));
        }

        let idle_timeout = with_config(&context.config, |cfg| cfg.timeouts.read_seconds);

        let model = context.upstream_model().to_string();
        let created_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        debug!(
            "SSE STREAM: building byte stream from response, idle_timeout={}",
            idle_timeout
        );
        let byte_stream = resp.bytes_stream();
        let stream =
            stream_responses_sse(byte_stream, &model, created_ts, chat, idle_timeout, session);
        debug!("SSE STREAM: stream created, converting to body");

        Ok(Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "text/event-stream")
            .header("Cache-Control", "no-cache")
            .header("Connection", "keep-alive")
            .body(Body::from_stream(stream))
            .map_err(|e| ProxyError::Internal(e.to_string()))?)
    }

    async fn execute_compact(
        &self,
        data: crate::schema::openai::CompactRequest,
        context: ProviderExecutionContext,
    ) -> Result<Response<Body>, ProxyError> {
        let model = context.upstream_model().to_string();
        let synth_raw = ResponsesRequest {
            model: model.clone(),
            input: Some(data.input.clone()),
            instructions: Some(data.instructions.clone()),
            messages: None,
            previous_response_id: None,
            store: None,
            metadata: None,
            tools: None,
            tool_choice: None,
            stream: Some(false),
            temperature: Some(0.1),
            top_p: None,
            max_tokens: Some(resolve_default_max_tokens(&context)?),
            max_output_tokens: None,
            include: None,
        };
        let synth_chat = normalize_responses_request(&synth_raw);
        let _session = TranslationSession::new();
        let anth_req = translate_to_anthropic_request(
            &synth_raw,
            &synth_chat,
            &TranslateCtx {
                stream: false,
                default_max_tokens: resolve_default_max_tokens(&context)?,
                reasoning: None,
            },
        );

        let auth = resolve_zai_auth(&context)?;
        let url = build_messages_url(&resolve_endpoint_url(&context)?);

        let mut req_headers = HeaderMap::new();
        req_headers.insert(AUTHORIZATION, auth);
        req_headers.insert(
            "Content-Type",
            "application/json"
                .parse()
                .map_err(|_| ProxyError::Internal("Invalid content type".into()))?,
        );

        debug!("Z.AI compact request to {}", url);

        let resp = self
            .client
            .post(&url)
            .headers(req_headers)
            .json(&anth_req)
            .send()
            .await
            .map_err(|e| ProxyError::Http(e))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            error!("Z.AI compact error: {} - {}", status, body);
            return Err(ProxyError::Provider(ProviderError::new(
                Some(status),
                format!("Z.AI compact error: {} - {}", status, body),
            )));
        }

        let bytes = resp.bytes().await.map_err(|e| ProxyError::Http(e))?;
        let anth: AnthropicResponse = serde_json::from_slice(&bytes).map_err(|e| {
            ProxyError::Internal(format!("Failed to parse Z.AI compact response: {}", e))
        })?;
        check_base_resp(&anth, &model)?;

        let summary = anth
            .content
            .iter()
            .filter_map(|b| match b {
                AnthropicResponseBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        let response_obj = ResponseObject {
            id: format!("cmp_{}", anth.id),
            object: "response",
            created_at: timestamp,
            completed_at: Some(timestamp),
            model: anth.model,
            status: "completed".to_string(),
            temperature: 0.0,
            top_p: 0.0,
            tool_choice: String::new(),
            tools: Vec::new(),
            parallel_tool_calls: false,
            store: false,
            metadata: Default::default(),
            output: vec![OutputItem::Message(MessageItem {
                id: format!("msg_{}", timestamp),
                role: "assistant",
                status: "completed".to_string(),
                content: vec![OutputContentPart::OutputText { text: summary }],
            })],
            usage: anth.usage.map(|u| Usage {
                input_tokens: u.input_tokens,
                output_tokens: u.output_tokens,
                total_tokens: u.input_tokens + u.output_tokens,
                input_tokens_details: None,
                output_tokens_details: None,
            }),
        };

        Ok(axum::Json(response_obj).into_response())
    }
}

fn resolve_zai_auth(context: &ProviderExecutionContext) -> Result<HeaderValue, ProxyError> {
    match &context.account.auth {
        crate::account_pool::AccountAuth::ApiKey { api_key } => {
            Ok(HeaderValue::from_str(&format!("Bearer {}", api_key))
                .map_err(|_| ProxyError::Auth("Invalid API key".into()))?)
        }
        crate::account_pool::AccountAuth::GeminiOAuth { .. } => Err(ProxyError::Auth(
            "Z.AI provider does not support OAuth authentication".into(),
        )),
    }
}

fn resolve_endpoint_url(context: &ProviderExecutionContext) -> Result<String, ProxyError> {
    with_config(&context.config, |cfg| {
        cfg.endpoint_url(context.provider(), context.endpoint_name())
    })
    .map_err(ProxyError::Config)
}

fn resolve_default_max_tokens(context: &ProviderExecutionContext) -> Result<u64, ProxyError> {
    with_config(&context.config, |cfg| {
        cfg.zai_provider_config(context.provider())
            .map(|c| c.default_max_tokens)
    })
    .map_err(ProxyError::Config)
}

fn build_messages_url(base: &str) -> String {
    let base = base.trim_end_matches('/');
    format!("{}/anthropic/v1/messages", base)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_messages_url() {
        assert_eq!(
            build_messages_url("https://open.bigmodel.cn"),
            "https://open.bigmodel.cn/anthropic/v1/messages"
        );
        assert_eq!(
            build_messages_url("https://open.bigmodel.cn/"),
            "https://open.bigmodel.cn/anthropic/v1/messages"
        );
    }
}

use axum::body::Body;
use axum::http::HeaderMap;
use axum::response::Response;
use serde::Serialize;
use serde_json::Value;

use crate::account_pool::AccountAuth;
use crate::config::{EffectiveReasoningConfig, with_config};
use crate::error::ProxyError;
use crate::providers::base::{Provider, ProviderExecutionContext};
use crate::schema::openai::{
    ChatRequest, CompactRequest, Instructions, ResponsesInput, ResponsesRequest, Tool,
    input_items_to_text, messages_to_input_items,
};

#[derive(Clone, Debug, Serialize)]
pub(crate) struct OpenAiReasoning {
    pub(crate) effort: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct OpenAiResponsesPayload {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    input: Option<ResponsesInput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<Instructions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<std::collections::BTreeMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    include: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<OpenAiReasoning>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct OpenAiCompactPayload {
    pub(crate) model: String,
    pub(crate) input: ResponsesInput,
    pub(crate) instructions: Instructions,
    pub(crate) store: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_output_tokens: Option<u64>,
    pub(crate) stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning: Option<OpenAiReasoning>,
}

pub struct OpenAiProvider {
    client: reqwest::Client,
}

impl Clone for OpenAiProvider {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
        }
    }
}

impl Default for OpenAiProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenAiProvider {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    fn resolve_endpoint_url(
        &self,
        context: &ProviderExecutionContext,
    ) -> Result<String, ProxyError> {
        with_config(&context.config, |cfg| {
            cfg.endpoint_url(context.provider(), context.endpoint_name())
        })
        .map_err(ProxyError::Config)
    }

    fn resolve_compact_endpoint_url(
        &self,
        context: &ProviderExecutionContext,
    ) -> Result<String, ProxyError> {
        let url = self.resolve_endpoint_url(context)?;
        let trimmed = url.trim_end_matches('/');
        if trimmed.ends_with("/compact") {
            return Ok(trimmed.to_string());
        }
        if trimmed.ends_with("/responses") || trimmed.ends_with("/v1/responses") {
            return Ok(format!("{trimmed}/compact"));
        }
        Ok(url)
    }

    fn resolve_models_url(&self, context: &ProviderExecutionContext) -> Result<String, ProxyError> {
        with_config(&context.config, |cfg| {
            cfg.provider_models_url(context.provider())
        })
        .ok_or_else(|| {
            ProxyError::Provider(format!(
                "Provider '{}' does not have a models_url configured",
                context.provider()
            ))
        })
    }

    async fn get_json_from(
        &self,
        endpoint_url: &str,
        mut headers: HeaderMap,
        context: &ProviderExecutionContext,
    ) -> Result<reqwest::Response, ProxyError> {
        let api_key = match &context.account.auth {
            AccountAuth::ApiKey { api_key } => api_key,
            _ => {
                return Err(ProxyError::Auth(
                    "OpenAI provider requires account auth.type=api_key".into(),
                ));
            }
        };

        headers.remove(axum::http::header::HOST);
        headers.remove(axum::http::header::CONTENT_LENGTH);
        headers.insert(
            axum::http::header::AUTHORIZATION,
            format!("Bearer {api_key}").parse().map_err(|e| {
                ProxyError::Internal(format!("Invalid OpenAI authorization header: {e}"))
            })?,
        );

        self.client
            .get(endpoint_url)
            .headers(headers)
            .timeout(std::time::Duration::from_secs(with_config(
                &context.config,
                |cfg| cfg.timeouts.read_seconds,
            )))
            .send()
            .await
            .map_err(ProxyError::Http)
    }

    async fn send_json_to<T: Serialize + ?Sized>(
        &self,
        endpoint_url: &str,
        payload: &T,
        mut headers: HeaderMap,
        context: &ProviderExecutionContext,
    ) -> Result<reqwest::Response, ProxyError> {
        let api_key = match &context.account.auth {
            AccountAuth::ApiKey { api_key } => api_key,
            _ => {
                return Err(ProxyError::Auth(
                    "OpenAI provider requires account auth.type=api_key".into(),
                ));
            }
        };

        headers.remove(axum::http::header::HOST);
        headers.remove(axum::http::header::CONTENT_LENGTH);
        headers.insert(
            axum::http::header::AUTHORIZATION,
            format!("Bearer {api_key}").parse().map_err(|e| {
                ProxyError::Internal(format!("Invalid OpenAI authorization header: {e}"))
            })?,
        );

        self.client
            .post(endpoint_url)
            .headers(headers)
            .json(payload)
            .timeout(std::time::Duration::from_secs(with_config(
                &context.config,
                |cfg| cfg.timeouts.read_seconds,
            )))
            .send()
            .await
            .map_err(ProxyError::Http)
    }

    async fn send_json<T: Serialize + ?Sized>(
        &self,
        payload: &T,
        headers: HeaderMap,
        context: &ProviderExecutionContext,
    ) -> Result<reqwest::Response, ProxyError> {
        let endpoint_url = self.resolve_endpoint_url(context)?;
        self.send_json_to(&endpoint_url, payload, headers, context)
            .await
    }

    pub(crate) async fn forward_json_to<T: Serialize + ?Sized>(
        &self,
        endpoint_url: &str,
        payload: &T,
        headers: HeaderMap,
        context: &ProviderExecutionContext,
    ) -> Result<Response<Body>, ProxyError> {
        let response = self
            .send_json_to(endpoint_url, payload, headers, context)
            .await?;

        let status = response.status();
        let response_headers = response.headers().clone();
        let bytes = response.bytes().await?;

        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err(ProxyError::Auth(format!(
                "OpenAI request unauthorized ({}): {}",
                status,
                String::from_utf8_lossy(&bytes)
            )));
        }
        if !status.is_success() {
            return Err(ProxyError::Provider(format!(
                "OpenAI request failed ({}): {}",
                status,
                String::from_utf8_lossy(&bytes)
            )));
        }

        let mut builder = Response::builder().status(status);
        for (name, value) in &response_headers {
            if name.as_str().eq_ignore_ascii_case("content-length") {
                continue;
            }
            builder = builder.header(name, value);
        }
        builder
            .body(Body::from(bytes))
            .map_err(|e| ProxyError::Internal(format!("Failed to build OpenAI response: {e}")))
    }

    pub(crate) async fn forward_json<T: Serialize + ?Sized>(
        &self,
        payload: &T,
        headers: HeaderMap,
        context: &ProviderExecutionContext,
    ) -> Result<Response<Body>, ProxyError> {
        let response = self.send_json(payload, headers, context).await?;

        let status = response.status();
        let response_headers = response.headers().clone();
        let bytes = response.bytes().await?;

        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err(ProxyError::Auth(format!(
                "OpenAI request unauthorized ({}): {}",
                status,
                String::from_utf8_lossy(&bytes)
            )));
        }
        if !status.is_success() {
            return Err(ProxyError::Provider(format!(
                "OpenAI request failed ({}): {}",
                status,
                String::from_utf8_lossy(&bytes)
            )));
        }

        let mut builder = Response::builder().status(status);
        for (name, value) in &response_headers {
            if name.as_str().eq_ignore_ascii_case("content-length") {
                continue;
            }
            builder = builder.header(name, value);
        }
        builder
            .body(Body::from(bytes))
            .map_err(|e| ProxyError::Internal(format!("Failed to build OpenAI response: {e}")))
    }
}

impl Provider for OpenAiProvider {
    fn handle_request(
        &self,
        raw_request: ResponsesRequest,
        _normalized_request: ChatRequest,
        headers: HeaderMap,
        context: ProviderExecutionContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Response<Body>, ProxyError>> + Send + '_>,
    > {
        let mut payload = build_openai_payload(&raw_request, &context, None);
        clamp_payload_max_tokens(&mut payload, &context);
        Box::pin(async move { self.forward_json(&payload, headers, &context).await })
    }

    fn handle_compact(
        &self,
        data: CompactRequest,
        headers: HeaderMap,
        context: ProviderExecutionContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Response<Body>, ProxyError>> + Send + '_>,
    > {
        let mut payload = OpenAiCompactPayload {
            model: context.upstream_model().to_string(),
            input: data.input,
            instructions: data.instructions,
            store: false,
            temperature: Some(with_config(&context.config, |cfg| cfg.compaction.temperature)),
            max_tokens: Some(4096),
            max_output_tokens: None,
            stream: false,
            reasoning: context
                .reasoning()
                .map(|reasoning| OpenAiReasoning { effort: reasoning_effort(reasoning) }),
        };
        clamp_compact_payload_max_tokens(&mut payload, &context);
        Box::pin(async move {
            let endpoint_url = self.resolve_compact_endpoint_url(&context)?;
            self.forward_json_to(&endpoint_url, &payload, headers, &context)
                .await
        })
    }

    fn list_models(
        &self,
        context: ProviderExecutionContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Vec<String>, ProxyError>> + Send + '_>,
    > {
        Box::pin(async move {
            let models_url = self.resolve_models_url(&context)?;
            let resp = self
                .get_json_from(&models_url, HeaderMap::new(), &context)
                .await?;

            let status = resp.status();
            let body = resp.text().await.map_err(ProxyError::Http)?;
            if !status.is_success() {
                return Err(ProxyError::Provider(format!(
                    "Upstream models endpoint returned status {} body={}",
                    status, body
                )));
            }

            let parsed: Value =
                serde_json::from_str(&body).map_err(|e| ProxyError::Provider(e.to_string()))?;
            let data = parsed
                .get("data")
                .and_then(|v| v.as_array())
                .ok_or_else(|| ProxyError::Provider("Upstream /models missing data[]".into()))?;

            let mut out = Vec::new();
            for item in data {
                if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                    out.push(id.to_string());
                }
            }
            Ok(out)
        })
    }

    fn clone_box(&self) -> Box<dyn Provider + Send + Sync> {
        Box::new(OpenAiProvider {
            client: self.client.clone(),
        })
    }
}

pub(crate) fn build_openai_payload(
    request: &ResponsesRequest,
    context: &ProviderExecutionContext,
    default_max_output_tokens: Option<u64>,
) -> OpenAiResponsesPayload {
    let input = resolve_input_for_payload(request);
    let mut payload = OpenAiResponsesPayload {
        model: context.upstream_model().to_string(),
        input,
        instructions: request.instructions.clone(),
        previous_response_id: request.previous_response_id.clone(),
        store: request.store,
        metadata: request.metadata.clone(),
        tools: request.tools.clone(),
        tool_choice: request.tool_choice.clone(),
        temperature: request.temperature,
        top_p: request.top_p,
        max_tokens: request.max_tokens,
        max_output_tokens: request
            .max_output_tokens
            .or(request.max_tokens)
            .or(default_max_output_tokens),
        stream: request.stream,
        include: request.include.clone(),
        reasoning: context
            .reasoning()
            .map(|reasoning| OpenAiReasoning { effort: reasoning_effort(reasoning) }),
    };
    if payload.max_output_tokens.is_none() {
        payload.max_output_tokens = payload.max_tokens;
    }
    payload
}

pub(crate) fn resolve_input_for_payload(request: &ResponsesRequest) -> Option<ResponsesInput> {
    if let Some(input) = &request.input {
        return Some(input.clone());
    }
    request
        .messages
        .as_ref()
        .map(|messages| ResponsesInput::Items(messages_to_input_items(messages)))
}

pub(crate) fn coerce_items_input_to_text(payload: &mut OpenAiResponsesPayload) {
    let Some(input) = payload.input.take() else { return };
    payload.input = Some(match input {
        ResponsesInput::Items(items) => ResponsesInput::Text(input_items_to_text(&items)),
        other => other,
    });
}

pub(crate) fn reasoning_effort(reasoning: &EffectiveReasoningConfig) -> &'static str {
    if let Some(preset) = reasoning.preset.as_deref() {
        return match preset {
            "none" => "none",
            "minimal" => "minimal",
            "low" => "low",
            "medium" => "medium",
            "high" => "high",
            "xhigh" => "xhigh",
            _ => budget_to_effort(reasoning.budget),
        };
    }
    budget_to_effort(reasoning.budget)
}

fn budget_to_effort(budget: u64) -> &'static str {
    match budget {
        0 => "none",
        1..=2048 => "minimal",
        2049..=4096 => "low",
        4097..=16384 => "medium",
        16385..=32768 => "high",
        _ => "xhigh",
    }
}

pub(crate) fn clamp_payload_max_tokens(
    payload: &mut OpenAiResponsesPayload,
    context: &ProviderExecutionContext,
) {
    let cap = with_config(&context.config, |cfg| {
        cfg.openai_provider_config(context.provider())
            .ok()
            .and_then(|cfg| cfg.max_tokens_cap)
    });
    let Some(cap) = cap else { return };
    payload.max_tokens = payload.max_tokens.map(|value| value.min(cap));
    payload.max_output_tokens = payload.max_output_tokens.map(|value| value.min(cap));
}

pub(crate) fn clamp_compact_payload_max_tokens(
    payload: &mut OpenAiCompactPayload,
    context: &ProviderExecutionContext,
) {
    let cap = with_config(&context.config, |cfg| {
        cfg.openai_provider_config(context.provider())
            .ok()
            .and_then(|cfg| cfg.max_tokens_cap)
    });
    let Some(cap) = cap else { return };
    payload.max_tokens = payload.max_tokens.map(|value| value.min(cap));
    payload.max_output_tokens = payload.max_output_tokens.map(|value| value.min(cap));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::account_pool::{Account, AccountAuth, ResolvedRoute};
    use crate::config::{
        AccessControlConfig, CompactionConfig, ModelsConfig, ProviderConfig, ProvidersConfig,
        ReasoningConfig, RoutingConfig, RoutingHealthConfig, ServerConfig, TimeoutsConfig,
    };
    use parking_lot::RwLock;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::Arc;

    fn context(
        reasoning: Option<EffectiveReasoningConfig>,
        endpoint: Option<&str>,
    ) -> ProviderExecutionContext {
        let providers: ProvidersConfig = HashMap::from([(
            "openai".into(),
            ProviderConfig::OpenAi {
                api_url: "https://example.com/v1/responses".into(),
                models_url: None,
                endpoints: HashMap::new(),
                models: Vec::new(),
                max_tokens_cap: None,
            },
        )]);
        let cfg = crate::config::Config {
            config_path: PathBuf::from("/tmp/test-config.json"),
            server: ServerConfig {
                host: "127.0.0.1".into(),
                port: 8765,
                log_level: "INFO".into(),
            },
            providers,
            models: ModelsConfig { served: Vec::new() },
            model_discovery: crate::config::ModelDiscoveryConfig::default(),
            model_metadata: crate::config::ProviderModelMetadataConfig::new(),
            models_endpoint: crate::config::ModelsEndpointConfig::default(),
            session: crate::config::SessionConfig::default(),
            auto_compaction: crate::config::AutoCompactionConfig::default(),
            routing: RoutingConfig {
                model_routes: HashMap::new(),
            },
            health: RoutingHealthConfig::default(),
            accounts: Vec::new(),
            access: AccessControlConfig::default(),
            reasoning: ReasoningConfig::default(),
            timeouts: TimeoutsConfig {
                connect_seconds: 10,
                read_seconds: 30,
            },
            compaction: CompactionConfig {
                temperature: 0.1,
                preferred_targets: Vec::new(),
            },
        };
        let config = Arc::new(RwLock::new(cfg));
        let gemini_auth = Arc::new(crate::auth::GeminiAuthManager::new(config.clone()));

        ProviderExecutionContext {
            route: ResolvedRoute {
                requested_model: "claude-sonnet-4-6".into(),
                logical_model: "claude-sonnet-4-6".into(),
                upstream_model: "gpt-5".into(),
                endpoint: endpoint.map(str::to_string),
                provider: "openai".into(),
                account_index: 0,
                account_id: "openai-a".into(),
                cache_hit: false,
                cache_key: 0,
                preferred_target_index: 0,
                reasoning,
            },
            account: Account {
                id: "openai-a".into(),
                provider: "openai".into(),
                auth: AccountAuth::ApiKey {
                    api_key: "test-key".into(),
                },
                enabled: true,
                weight: 1,
                models: None,
            },
            config,
            gemini_auth,
        }
    }

    #[test]
    fn build_payload_applies_model_and_reasoning() {
        let req = ResponsesRequest {
            model: "placeholder".into(),
            input: Some(ResponsesInput::Text("hi".into())),
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
        };
        let payload = build_openai_payload(
            &req,
            &context(
                Some(EffectiveReasoningConfig {
                    budget: 16384,
                    level: "MEDIUM".into(),
                    preset: Some("medium".into()),
                }),
                None,
            ),
            None,
        );
        assert_eq!(payload.model, "gpt-5");
        assert_eq!(payload.reasoning.as_ref().unwrap().effort, "medium");
    }

    #[test]
    fn maps_inline_budget_to_effort() {
        assert_eq!(
            reasoning_effort(&EffectiveReasoningConfig {
                budget: 50000,
                level: "HIGH".into(),
                preset: None,
            }),
            "xhigh"
        );
    }

    #[test]
    fn maps_named_reasoning_presets_to_openai_effort() {
        assert_eq!(
            reasoning_effort(&EffectiveReasoningConfig {
                budget: 0,
                level: "LOW".into(),
                preset: Some("none".into()),
            }),
            "none"
        );
        assert_eq!(
            reasoning_effort(&EffectiveReasoningConfig {
                budget: 2048,
                level: "LOW".into(),
                preset: Some("minimal".into()),
            }),
            "minimal"
        );
    }

    #[test]
    fn clamps_max_tokens_when_cap_configured() {
        let req = ResponsesRequest {
            model: "placeholder".into(),
            input: Some(ResponsesInput::Text("hi".into())),
            messages: None,
            instructions: None,
            previous_response_id: None,
            store: None,
            metadata: None,
            tools: None,
            tool_choice: None,
            temperature: None,
            top_p: None,
            max_tokens: Some(65536),
            max_output_tokens: Some(65536),
            stream: None,
            include: None,
        };

        let providers: ProvidersConfig = HashMap::from([(
            "openrouter".into(),
            ProviderConfig::OpenAi {
                api_url: "https://openrouter.example/v1/responses".into(),
                models_url: None,
                endpoints: HashMap::new(),
                models: vec!["z-ai/glm-5-turbo".into()],
                max_tokens_cap: Some(10_000),
            },
        )]);

        let cfg = crate::config::Config {
            config_path: PathBuf::from("/tmp/test-config.json"),
            server: ServerConfig {
                host: "127.0.0.1".into(),
                port: 8765,
                log_level: "INFO".into(),
            },
            providers,
            models: ModelsConfig { served: Vec::new() },
            model_discovery: crate::config::ModelDiscoveryConfig::default(),
            model_metadata: crate::config::ProviderModelMetadataConfig::new(),
            models_endpoint: crate::config::ModelsEndpointConfig::default(),
            session: crate::config::SessionConfig::default(),
            auto_compaction: crate::config::AutoCompactionConfig::default(),
            routing: RoutingConfig {
                model_routes: HashMap::new(),
            },
            health: RoutingHealthConfig::default(),
            accounts: Vec::new(),
            access: AccessControlConfig::default(),
            reasoning: ReasoningConfig::default(),
            timeouts: TimeoutsConfig {
                connect_seconds: 10,
                read_seconds: 30,
            },
            compaction: CompactionConfig {
                temperature: 0.1,
                preferred_targets: Vec::new(),
            },
        };

        let config = Arc::new(RwLock::new(cfg));
        let gemini_auth = Arc::new(crate::auth::GeminiAuthManager::new(config.clone()));
        let context = ProviderExecutionContext {
            route: ResolvedRoute {
                requested_model: "glm-5-turbo".into(),
                logical_model: "glm-5-turbo".into(),
                upstream_model: "z-ai/glm-5-turbo".into(),
                endpoint: None,
                provider: "openrouter".into(),
                account_index: 0,
                account_id: "openrouter-main".into(),
                cache_hit: false,
                cache_key: 0,
                preferred_target_index: 0,
                reasoning: None,
            },
            account: Account {
                id: "openrouter-main".into(),
                provider: "openrouter".into(),
                auth: AccountAuth::ApiKey {
                    api_key: "sk-test".into(),
                },
                enabled: true,
                weight: 1,
                models: None,
            },
            config,
            gemini_auth,
        };

        let mut payload = build_openai_payload(&req, &context, None);
        clamp_payload_max_tokens(&mut payload, &context);
        assert_eq!(payload.max_tokens, Some(10_000));
        assert_eq!(payload.max_output_tokens, Some(10_000));
    }

    #[test]
    fn infers_max_output_tokens_from_max_tokens() {
        let req = ResponsesRequest {
            model: "gpt-4.1".into(),
            input: Some(ResponsesInput::Text("hi".into())),
            messages: None,
            instructions: None,
            previous_response_id: None,
            store: None,
            metadata: None,
            tools: None,
            tool_choice: None,
            temperature: None,
            top_p: None,
            max_tokens: Some(7),
            max_output_tokens: None,
            stream: None,
            include: None,
        };
        let payload = build_openai_payload(&req, &context(None, None), None);
        assert_eq!(payload.max_output_tokens, Some(7));
    }
}

use axum::body::Body;
use axum::http::HeaderMap;
use axum::response::Response;

use crate::error::ProxyError;
use crate::providers::base::{Provider, ProviderExecutionContext};
use crate::schema::openai::{ChatRequest, CompactRequest, ResponsesRequest};

use super::openai::{
    OpenAiCompactPayload, OpenAiProvider, build_openai_payload, clamp_compact_payload_max_tokens,
    clamp_payload_max_tokens, reasoning_effort, OpenAiReasoning,
};

pub struct OpenRouterProvider {
    inner: OpenAiProvider,
}

impl Default for OpenRouterProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenRouterProvider {
    pub fn new() -> Self {
        Self {
            inner: OpenAiProvider::new(),
        }
    }
}

impl Provider for OpenRouterProvider {
    fn handle_request(
        &self,
        raw_request: ResponsesRequest,
        _normalized_request: ChatRequest,
        headers: HeaderMap,
        context: ProviderExecutionContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Response<Body>, ProxyError>> + Send + '_>,
    > {
        let inner = self.inner.clone();
        let default_max_output_tokens = resolve_openrouter_default_max_output_tokens(&context);
        let mut payload =
            build_openai_payload(&raw_request, &context, Some(default_max_output_tokens));
        clamp_payload_max_tokens(&mut payload, &context);

        Box::pin(async move { inner.forward_json(&payload, headers, &context).await })
    }

    fn handle_compact(
        &self,
        data: CompactRequest,
        headers: HeaderMap,
        context: ProviderExecutionContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Response<Body>, ProxyError>> + Send + '_>,
    > {
        let inner = self.inner.clone();
        let default_max_output_tokens = resolve_openrouter_default_max_output_tokens(&context);
        let mut payload = OpenAiCompactPayload {
            model: context.upstream_model().to_string(),
            input: data.input,
            instructions: data.instructions,
            store: false,
            temperature: None,
            max_tokens: Some(4096),
            max_output_tokens: Some(default_max_output_tokens),
            stream: false,
            reasoning: context.reasoning().map(|reasoning| OpenAiReasoning {
                effort: reasoning_effort(reasoning),
            }),
        };
        clamp_compact_payload_max_tokens(&mut payload, &context);
        Box::pin(async move { inner.forward_json(&payload, headers, &context).await })
    }

    fn list_models(
        &self,
        context: ProviderExecutionContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Vec<String>, ProxyError>> + Send + '_>,
    > {
        self.inner.list_models(context)
    }

    fn clone_box(&self) -> Box<dyn Provider + Send + Sync> {
        Box::new(OpenRouterProvider::new())
    }
}

fn resolve_openrouter_default_max_output_tokens(context: &ProviderExecutionContext) -> u64 {
    crate::config::with_config(&context.config, |cfg| {
        cfg.model_metadata(context.provider(), context.upstream_model())
            .and_then(|metadata| metadata.max_output_tokens)
            .map(u64::from)
            .unwrap_or(4096)
    })
}

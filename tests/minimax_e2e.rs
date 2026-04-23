//! E2E smoke tests for MiniMax provider.
//!
//! Run with: MINIMAX_API_KEY=xxx cargo test --test minimax_e2e -- --ignored
//!
//! These tests require a real MiniMax API key and make actual network calls.

use std::collections::HashMap;
use std::env;
use std::sync::Arc;

use axum::http::HeaderMap;
use parking_lot::RwLock;

use codex_proxy::account_pool::{Account, AccountAuth, ResolvedRoute};
use codex_proxy::auth::GeminiAuthManager;
use codex_proxy::config::{
    AccessControlConfig, AccountConfig, CompactionConfig, Config, ConfigHandle,
    ModelDiscoveryConfig, ModelsConfig, ProviderConfig, ReasoningConfig, RoutingConfig,
    RoutingHealthConfig, ServerConfig, SessionConfig, TimeoutsConfig,
};
use codex_proxy::providers::base::{Provider, ProviderExecutionContext};
use codex_proxy::providers::minimax::MinimaxProvider;
use codex_proxy::schema::openai::{ChatRequest, ResponsesInput, ResponsesRequest, Tool};
use fp_agent::normalize::normalize_responses_request;
use fp_agent::schema::openai::Instructions;

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

fn minimax_api_key() -> String {
    env::var("MINIMAX_API_KEY").expect("MINIMAX_API_KEY env var required for E2E tests")
}

fn test_config() -> ConfigHandle {
    let api_key = minimax_api_key();
    let mut providers = HashMap::new();
    providers.insert(
        "minimax".into(),
        ProviderConfig::Minimax {
            api_url: "https://api.minimax.chat".into(),
            endpoints: HashMap::new(),
            models: vec!["MiniMax-M2.7-highspeed".into()],
            default_max_tokens: Some(4096),
        },
    );

    Arc::new(RwLock::new(Config {
        config_path: Default::default(),
        server: ServerConfig {
            host: "0.0.0.0".into(),
            port: 8080,
            log_level: "error".into(),
        },
        providers,
        models: ModelsConfig { served: vec![] },
        model_discovery: ModelDiscoveryConfig::default(),
        model_metadata: HashMap::new(),
        models_endpoint: Default::default(),
        session: SessionConfig::default(),
        auto_compaction: Default::default(),
        routing: RoutingConfig {
            model_routes: HashMap::new(),
        },
        health: RoutingHealthConfig::default(),
        accounts: vec![AccountConfig {
            id: "test-minimax".into(),
            provider: "minimax".into(),
            enabled: true,
            weight: 1,
            models: None,
            auth: AccountAuth::ApiKey { api_key },
        }],
        access: AccessControlConfig::default(),
        reasoning: ReasoningConfig::default(),
        timeouts: TimeoutsConfig {
            connect_seconds: 30,
            read_seconds: 120,
        },
        compaction: CompactionConfig {
            temperature: 0.7,
            preferred_targets: Vec::new(),
        },
    }))
}

fn test_route() -> ResolvedRoute {
    ResolvedRoute {
        requested_model: "MiniMax-M2.7-highspeed".into(),
        logical_model: "MiniMax-M2.7-highspeed".into(),
        upstream_model: "MiniMax-M2.7-highspeed".into(),
        endpoint: None,
        provider: "minimax".into(),
        account_index: 0,
        account_id: "test-minimax".into(),
        cache_hit: false,
        cache_key: 0,
        preferred_target_index: 0,
        reasoning: None,
    }
}

fn test_account() -> Account {
    Account {
        id: "test-minimax".into(),
        provider: "minimax".into(),
        auth: AccountAuth::ApiKey {
            api_key: minimax_api_key(),
        },
        enabled: true,
        weight: 1,
        models: None,
    }
}

fn mock_context() -> ProviderExecutionContext {
    let config = test_config();
    ProviderExecutionContext {
        route: test_route(),
        account: test_account(),
        config: config.clone(),
        gemini_auth: Arc::new(GeminiAuthManager::new(config)),
    }
}

fn make_sync_request(prompt: &str) -> (ResponsesRequest, ChatRequest) {
    let raw = ResponsesRequest {
        model: "MiniMax-M2.7-highspeed".into(),
        input: Some(ResponsesInput::Text(prompt.into())),
        messages: None,
        instructions: None,
        previous_response_id: None,
        store: None,
        metadata: None,
        tools: None,
        tool_choice: None,
        temperature: None,
        top_p: None,
        max_tokens: Some(256),
        max_output_tokens: None,
        stream: Some(false),
        include: None,
    };
    let chat = normalize_responses_request(&raw);
    (raw, chat)
}

fn make_stream_request(prompt: &str) -> (ResponsesRequest, ChatRequest) {
    let raw = ResponsesRequest {
        model: "MiniMax-M2.7-highspeed".into(),
        input: Some(ResponsesInput::Text(prompt.into())),
        messages: None,
        instructions: None,
        previous_response_id: None,
        store: None,
        metadata: None,
        tools: None,
        tool_choice: None,
        temperature: None,
        top_p: None,
        max_tokens: Some(256),
        max_output_tokens: None,
        stream: Some(true),
        include: None,
    };
    let chat = normalize_responses_request(&raw);
    (raw, chat)
}

fn make_tool_request(prompt: &str) -> (ResponsesRequest, ChatRequest) {
    let tools = vec![Tool {
        tool_type: "function".into(),
        function: Some(codex_proxy::schema::openai::FunctionDef {
            name: "get_weather".into(),
            description: Some("Get current weather for a location".into()),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            })),
        }),
        name: None,
        description: None,
        parameters: None,
        strict: None,
        web_search: None,
        tools: None,
    }];

    let raw = ResponsesRequest {
        model: "MiniMax-M2.7-highspeed".into(),
        input: Some(ResponsesInput::Text(prompt.into())),
        messages: None,
        instructions: Some(Instructions::Text("You have access to tools.".into())),
        previous_response_id: None,
        store: None,
        metadata: None,
        tools: Some(tools),
        tool_choice: Some("auto".into()),
        temperature: None,
        top_p: None,
        max_tokens: Some(512),
        max_output_tokens: None,
        stream: Some(true),
        include: None,
    };
    let chat = normalize_responses_request(&raw);
    (raw, chat)
}

fn make_thinking_request(prompt: &str, reasoning: bool) -> (ResponsesRequest, ChatRequest) {
    let raw = ResponsesRequest {
        model: "MiniMax-M2.7-highspeed".into(),
        input: Some(ResponsesInput::Text(prompt.into())),
        messages: None,
        instructions: if reasoning {
            Some(Instructions::Text("Think step by step.".into()))
        } else {
            None
        },
        previous_response_id: None,
        store: None,
        metadata: None,
        tools: None,
        tool_choice: None,
        temperature: None,
        top_p: None,
        max_tokens: Some(512),
        max_output_tokens: None,
        stream: Some(true),
        include: None,
    };
    let chat = normalize_responses_request(&raw);
    (raw, chat)
}

// ---------------------------------------------------------------------------
// E2E tests (ignored by default)
// ---------------------------------------------------------------------------

/// E2E test: sync short answer from MiniMax.
#[tokio::test]
#[ignore]
async fn e2e_sync_short_answer() {
    let provider = MinimaxProvider::new();
    let context = mock_context();
    let (raw, chat) = make_sync_request("What is 2+2? Answer in one sentence.");

    let headers = HeaderMap::new();
    let resp = provider
        .handle_request(raw, chat, headers, context)
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200, "expected 200 OK");
    let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .expect("should read body");
    let body_str = String::from_utf8_lossy(&body);
    assert!(
        body_str.contains("output") || body_str.contains("text"),
        "response should contain output field: {}",
        body_str
    );
}

/// E2E test: streaming short answer from MiniMax.
#[tokio::test]
#[ignore]
async fn e2e_stream_short_answer() {
    let provider = MinimaxProvider::new();
    let context = mock_context();
    let (raw, chat) = make_stream_request("What is the capital of France?");

    let headers = HeaderMap::new();
    let resp = provider
        .handle_request(raw, chat, headers, context)
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200, "expected 200 OK");
    assert_eq!(
        resp.headers()
            .get("content-type")
            .map(|v| v.to_str().unwrap_or("")),
        Some("text/event-stream"),
        "should be SSE"
    );
    let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .expect("should read body");
    let body_str = String::from_utf8_lossy(&body);
    assert!(
        body_str.contains("data:"),
        "SSE should contain data events: {}",
        body_str
    );
}

/// E2E test: streaming with tool call.
#[tokio::test]
#[ignore]
async fn e2e_stream_tool_call() {
    let provider = MinimaxProvider::new();
    let context = mock_context();
    let (raw, chat) = make_tool_request("What's the weather in Tokyo?");

    let headers = HeaderMap::new();
    let resp = provider
        .handle_request(raw, chat, headers, context)
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200, "expected 200 OK");
    let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .expect("should read body");
    let body_str = String::from_utf8_lossy(&body);
    assert!(
        body_str.contains("tool_call") || body_str.contains("get_weather"),
        "response should contain tool_call: {}",
        body_str
    );
}

/// E2E test: multi-turn conversation with thinking signature.
#[tokio::test]
#[ignore]
async fn e2e_multi_turn_thinking_signature() {
    let provider = MinimaxProvider::new();
    let context = mock_context();

    let (raw1, chat1) = make_thinking_request("Explain why the sky is blue in 3 sentences.", true);
    let headers = HeaderMap::new();
    let resp1 = provider
        .handle_request(raw1, chat1, headers, context.clone())
        .await
        .expect("first request should succeed");

    assert_eq!(resp1.status(), 200, "first turn should succeed");

    let (raw2, chat2) =
        make_thinking_request("Now explain why clouds are white in 3 sentences.", false);
    let headers2 = HeaderMap::new();
    let resp2 = provider
        .handle_request(raw2, chat2, headers2, context)
        .await
        .expect("second request should succeed");

    assert_eq!(resp2.status(), 200, "second turn should succeed");

    let body1 = axum::body::to_bytes(resp1.into_body(), 1024 * 1024)
        .await
        .expect("should read first body");
    let body2 = axum::body::to_bytes(resp2.into_body(), 1024 * 1024)
        .await
        .expect("should read second body");

    let body1_str = String::from_utf8_lossy(&body1);
    let body2_str = String::from_utf8_lossy(&body2);

    assert!(
        body1_str.contains("data:") || body1_str.contains("output"),
        "first response should be valid: {}",
        body1_str
    );
    assert!(
        body2_str.contains("data:") || body2_str.contains("output"),
        "second response should be valid: {}",
        body2_str
    );
}

/// E2E test: namespace tool deferred loading (Codex MCP tools).
#[tokio::test]
#[ignore]
async fn e2e_namespace_tool_deferred_loading() {
    let provider = MinimaxProvider::new();
    let context = mock_context();
    let tools = vec![Tool {
        tool_type: "namespace".to_string(),
        function: None,
        name: Some("mcp__codex_apps__calendar".to_string()),
        description: Some("Calendar MCP".to_string()),
        parameters: None,
        strict: None,
        web_search: None,
        tools: Some(vec![Tool {
            tool_type: "function".to_string(),
            function: Some(codex_proxy::schema::openai::FunctionDef {
                name: "create_event".to_string(),
                description: Some("Create event".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"}
                    }
                })),
            }),
            name: None,
            description: None,
            parameters: None,
            strict: None,
            web_search: None,
            tools: None,
        }]),
    }];

    let raw = ResponsesRequest {
        model: "MiniMax-M2.7-highspeed".into(),
        input: Some(ResponsesInput::Text(
            "Create a calendar event titled 'Team Meeting'".into(),
        )),
        messages: None,
        instructions: Some(Instructions::Text("You have access to tools.".into())),
        previous_response_id: None,
        store: None,
        metadata: None,
        tools: Some(tools),
        tool_choice: Some("auto".into()),
        temperature: None,
        top_p: None,
        max_tokens: Some(256),
        max_output_tokens: None,
        stream: Some(false),
        include: None,
    };
    let chat = normalize_responses_request(&raw);
    let headers = HeaderMap::new();
    let resp = provider
        .handle_request(raw, chat, headers, context)
        .await
        .expect("request should succeed");
    assert_eq!(resp.status(), 200, "namespace tools should not 400");
}

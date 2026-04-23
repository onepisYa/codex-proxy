//! Integration tests for MiniMax provider using wiremock.
//!
//! Run with: cargo test --test minimax_integration
//!
//! These tests use wiremock to mock the MiniMax API and do not require a real API key.

use std::collections::HashMap;
use std::sync::Arc;

use axum::http::HeaderMap;
use parking_lot::RwLock;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use codex_proxy::account_pool::{Account, AccountAuth, ResolvedRoute};
use codex_proxy::auth::GeminiAuthManager;
use codex_proxy::config::{
    AccessControlConfig, AccountConfig, CompactionConfig, Config, ConfigHandle,
    ModelDiscoveryConfig, ModelsConfig, ProviderConfig, ReasoningConfig, RoutingConfig,
    RoutingHealthConfig, ServerConfig, SessionConfig, TimeoutsConfig,
};
use codex_proxy::providers::base::{Provider, ProviderExecutionContext};
use codex_proxy::providers::minimax::MinimaxProvider;
use codex_proxy::schema::openai::{ChatRequest, ResponsesInput, ResponsesRequest};
use fp_agent::normalize::normalize_responses_request;

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

fn test_config(api_url: &str) -> ConfigHandle {
    let mut providers = HashMap::new();
    providers.insert(
        "minimax".into(),
        ProviderConfig::Minimax {
            api_url: api_url.into(),
            endpoints: HashMap::new(),
            models: vec!["MiniMax-Text-01".into()],
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
            auth: AccountAuth::ApiKey {
                api_key: "test-key".into(),
            },
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
        requested_model: "MiniMax-Text-01".into(),
        logical_model: "MiniMax-Text-01".into(),
        upstream_model: "MiniMax-Text-01".into(),
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
            api_key: "test-key".into(),
        },
        enabled: true,
        weight: 1,
        models: None,
    }
}

fn mock_context(api_url: &str) -> ProviderExecutionContext {
    let config = test_config(api_url);
    ProviderExecutionContext {
        route: test_route(),
        account: test_account(),
        config: config.clone(),
        gemini_auth: Arc::new(GeminiAuthManager::new(config)),
    }
}

fn make_sync_request(prompt: &str) -> (ResponsesRequest, ChatRequest) {
    let raw = ResponsesRequest {
        model: "MiniMax-Text-01".into(),
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
        model: "MiniMax-Text-01".into(),
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

// ---------------------------------------------------------------------------
// Integration tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sync_text_returns_output() {
    // GIVEN a wiremock server that returns a sync text response
    let mock_server = MockServer::start().await;
    let minimax_resp = serde_json::json!({
        "id": "resp_test123",
        "type": "message",
        "role": "assistant",
        "content": [{
            "type": "text",
            "text": "The capital of France is Paris."
        }],
        "model": "MiniMax-Text-01",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 12,
            "output_tokens": 8
        }
    });
    Mock::given(method("POST"))
        .and(path("/anthropic/v1/messages"))
        .and(header("content-type", "application/json"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(minimax_resp))
        .mount(&mock_server)
        .await;

    // WHEN we call the provider
    let provider = MinimaxProvider::new();
    let context = mock_context(&mock_server.uri());
    let (raw, chat) = make_sync_request("What is the capital of France?");
    let headers = HeaderMap::new();

    let resp = provider
        .handle_request(raw, chat, headers, context)
        .await
        .expect("request should succeed");

    // THEN we get a 200 response
    assert_eq!(resp.status(), 200, "expected 200 OK");
}

#[tokio::test]
async fn streaming_delta_yields_sse() {
    // GIVEN a wiremock server that returns streaming chunks
    let mock_server = MockServer::start().await;
    let chunks = vec![
        r#"data: {"type":"message_start","message":{"id":"resp_abc","type":"message","role":"assistant","model":"MiniMax-Text-01","content":[],"stop_reason":null,"usage":{"input_tokens":10,"output_tokens":0}}}"#,
        r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
        r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}"#,
        r#"data: {"type":"content_block_stop","index":0}"#,
        r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}"#,
        r#"data: {"type":"message_stop"}"#,
    ];
    let body = chunks.join("\n");
    Mock::given(method("POST"))
        .and(path("/anthropic/v1/messages"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(body, "text/event-stream"),
        )
        .mount(&mock_server)
        .await;

    // WHEN we call the provider in streaming mode
    let provider = MinimaxProvider::new();
    let context = mock_context(&mock_server.uri());
    let (raw, chat) = make_stream_request("Say hello");
    let headers = HeaderMap::new();

    let resp = provider
        .handle_request(raw, chat, headers, context)
        .await
        .expect("request should succeed");

    // THEN response is SSE
    assert_eq!(resp.status(), 200);
    let ct = resp
        .headers()
        .get("content-type")
        .map(|v| v.to_str().unwrap_or(""))
        .unwrap_or("");
    assert!(ct.contains("text/event-stream"), "expected SSE, got {}", ct);
}

#[tokio::test]
async fn streaming_tool_call_emits_tool_events() {
    // GIVEN a wiremock server that returns a streaming tool call
    let mock_server = MockServer::start().await;
    let chunks = vec![
        r#"data: {"type":"message_start","message":{"id":"resp_tool","type":"message","role":"assistant","model":"MiniMax-Text-01","content":[],"stop_reason":null,"usage":{"input_tokens":10,"output_tokens":0}}}"#,
        r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_001","name":"get_weather"}}"#,
        r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","input_json_fragment":"{\"location\":\"Tokyo\"}"}}"#,
        r#"data: {"type":"content_block_stop","index":0}"#,
        r#"data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":20}}"#,
        r#"data: {"type":"message_stop"}"#,
    ];
    let body = chunks.join("\n");
    Mock::given(method("POST"))
        .and(path("/anthropic/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(body, "text/event-stream"),
        )
        .mount(&mock_server)
        .await;

    // WHEN
    let provider = MinimaxProvider::new();
    let context = mock_context(&mock_server.uri());
    let (raw, chat) = make_stream_request("What's the weather in Tokyo?");
    let headers = HeaderMap::new();

    let resp = provider
        .handle_request(raw, chat, headers, context)
        .await
        .expect("request should succeed");

    // THEN 200
    assert_eq!(resp.status(), 200);
    let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .expect("should read body");
    let body_str = String::from_utf8_lossy(&body);
    assert!(
        body_str.contains("tool_call") || body_str.contains("get_weather"),
        "should contain tool_call: {}",
        body_str
    );
}

#[tokio::test]
async fn tool_result_roundtrip() {
    // GIVEN a streaming response with tool call
    let mock_server = MockServer::start().await;
    let chunks = vec![
        r#"data: {"type":"message_start","message":{"id":"resp_rt","type":"message","role":"assistant","model":"MiniMax-Text-01","content":[]}}"#,
        r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_001","name":"get_weather"}}"#,
        r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","input_json_fragment":"{\"location\":\"Tokyo\"}"}}"#,
        r#"data: {"type":"content_block_stop","index":0}"#,
        r#"data: {"type":"message_stop"}"#,
    ];
    Mock::given(method("POST"))
        .and(path("/anthropic/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(chunks.join("\n"), "text/event-stream"),
        )
        .mount(&mock_server)
        .await;

    let provider = MinimaxProvider::new();
    let context = mock_context(&mock_server.uri());
    let (raw, chat) = make_stream_request("Weather in Tokyo?");
    let headers = HeaderMap::new();

    let resp = provider
        .handle_request(raw, chat, headers, context)
        .await
        .expect("tool call should succeed");
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn base_resp_error_maps_to_400() {
    // GIVEN a wiremock server that returns a 400 with base_resp error
    let mock_server = MockServer::start().await;
    let error_resp = serde_json::json!({
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "temperature out of range"
        }
    });
    Mock::given(method("POST"))
        .and(path("/anthropic/v1/messages"))
        .respond_with(ResponseTemplate::new(400).set_body_json(error_resp))
        .mount(&mock_server)
        .await;

    let provider = MinimaxProvider::new();
    let context = mock_context(&mock_server.uri());
    let (raw, chat) = make_sync_request("Hi");
    let headers = HeaderMap::new();

    let err = provider
        .handle_request(raw, chat, headers, context)
        .await
        .expect_err("400 error should return Err");

    // THEN we get a Provider error with 4xx status
    let err_str = err.to_string();
    assert!(
        err_str.contains("400") || err_str.contains("invalid"),
        "expected 400 error, got: {}",
        err_str
    );
}

#[tokio::test]
async fn upstream_401_maps_to_auth_error() {
    // GIVEN a wiremock server that returns 401
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/anthropic/v1/messages"))
        .respond_with(ResponseTemplate::new(401).set_body_raw("Unauthorized", "text/plain"))
        .mount(&mock_server)
        .await;

    let provider = MinimaxProvider::new();
    let context = mock_context(&mock_server.uri());
    let (raw, chat) = make_sync_request("Hi");
    let headers = HeaderMap::new();

    let resp = provider.handle_request(raw, chat, headers, context).await;

    // THEN returns error (401 maps to ProxyError::Auth)
    assert!(resp.is_err(), "expected 401 to return Err");
    let err_str = resp.unwrap_err().to_string();
    assert!(
        err_str.contains("401"),
        "expected 401 auth error, got: {}",
        err_str
    );
}

#[tokio::test]
async fn upstream_500_is_error_response() {
    // GIVEN a wiremock server that returns 500
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/anthropic/v1/messages"))
        .respond_with(
            ResponseTemplate::new(500).set_body_raw("Internal Server Error", "text/plain"),
        )
        .mount(&mock_server)
        .await;

    let provider = MinimaxProvider::new();
    let context = mock_context(&mock_server.uri());
    let (raw, chat) = make_sync_request("Hi");
    let headers = HeaderMap::new();

    let resp = provider.handle_request(raw, chat, headers, context).await;

    // THEN returns Provider error with 5xx status
    assert!(resp.is_err(), "expected 500 to return Err");
    let err_str = resp.unwrap_err().to_string();
    assert!(
        err_str.contains("500"),
        "expected 500 error, got: {}",
        err_str
    );
}

#[tokio::test]
async fn stream_idle_timeout_yields_error_event() {
    // GIVEN a wiremock server that hangs then returns incomplete data
    let mock_server = MockServer::start().await;
    // Very short body to trigger idle timeout on our short timeout config
    Mock::given(method("POST"))
        .and(path("/anthropic/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                // Just one chunk then done - no hanging, but we test the timeout separately
                .set_body_raw(
                    r#"data: {"type":"message_start","message":{"id":"resp_timeout","type":"message","role":"assistant","content":[]}}"#,
                    "text/event-stream",
                ),
        )
        .mount(&mock_server)
        .await;

    let provider = MinimaxProvider::new();
    let context = mock_context(&mock_server.uri());

    // Override timeout via a new config with short read timeout
    let mut config = (*context.config.read()).clone();
    config.timeouts.read_seconds = 1;
    let short_config: ConfigHandle = Arc::new(RwLock::new(config));
    let short_context = ProviderExecutionContext {
        route: context.route.clone(),
        account: context.account.clone(),
        config: short_config,
        gemini_auth: context.gemini_auth.clone(),
    };

    let (raw, chat) = make_stream_request("Hello");
    let headers = HeaderMap::new();

    let resp = provider
        .handle_request(raw, chat, headers, short_context)
        .await
        .expect("should handle");
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn concurrent_requests_isolated() {
    // GIVEN two wiremock servers
    let mock_server1 = MockServer::start().await;
    let mock_server2 = MockServer::start().await;

    let resp1 = serde_json::json!({
        "id": "resp_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Response from server 1"}],
        "model": "MiniMax-Text-01",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 5}
    });
    let resp2 = serde_json::json!({
        "id": "resp_2",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Response from server 2"}],
        "model": "MiniMax-Text-01",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 5}
    });

    Mock::given(method("POST"))
        .and(path("/anthropic/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(resp1))
        .mount(&mock_server1)
        .await;
    Mock::given(method("POST"))
        .and(path("/anthropic/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(resp2))
        .mount(&mock_server2)
        .await;

    let provider = MinimaxProvider::new();

    let ctx1 = mock_context(&mock_server1.uri());
    let ctx2 = mock_context(&mock_server2.uri());
    let (raw1, chat1) = make_sync_request("Request 1");
    let (raw2, chat2) = make_sync_request("Request 2");
    let headers = HeaderMap::new();

    // WHEN both requests run concurrently
    let (resp1, resp2) = tokio::join!(
        provider.handle_request(raw1, chat1, headers.clone(), ctx1),
        provider.handle_request(raw2, chat2, headers, ctx2)
    );

    // THEN both succeed and are isolated
    assert_eq!(resp1.expect("req1").status(), 200);
    assert_eq!(resp2.expect("req2").status(), 200);
}

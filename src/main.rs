use codex_proxy::config::{Config, with_config};
use codex_proxy::server::build_router;
use codex_proxy::state::AppState;
use parking_lot::RwLock;
use std::sync::Arc;
use tracing::info;
use tracing_subscriber::fmt::format::FmtSpan;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("CODEX_PROXY_LOG_LEVEL")
                .unwrap_or_else(|_| "codex_proxy=debug".parse().unwrap()),
        )
        .with_span_events(FmtSpan::NONE)
        .with_target(true)
        .init();

    info!("Starting codex-proxy...");

    let config = Config::new();
    let config_handle = Arc::new(RwLock::new(config));
    let addr = with_config(&config_handle, |cfg| {
        format!("{}:{}", cfg.server.host, cfg.server.port)
    });
    let (ui_host, ui_port) = with_config(&config_handle, |cfg| {
        let host = if cfg.server.host == "0.0.0.0" {
            "127.0.0.1".to_string()
        } else {
            cfg.server.host.clone()
        };
        (host, cfg.server.port)
    });
    let state = AppState::new(config_handle.clone());
    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("Failed to bind");
    info!("Server bound to {addr}");
    info!("UI: http://{}:{}/ui", ui_host, ui_port);
    info!("Admin API: http://{}:{}/api/config", ui_host, ui_port);

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("Server error");
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install CTRL+C handler");
    info!("Shutting down...");
}

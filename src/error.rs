use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

#[derive(Debug, thiserror::Error)]
pub enum ProxyError {
    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("Provider error: {0}")]
    Provider(String),

    #[error("Authentication error: {0}")]
    Auth(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("{0}")]
    Http(#[from] reqwest::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Invalid port: {0}")]
    InvalidPort(String),

    #[error("Invalid URL: {0}")]
    InvalidUrl(String),

    #[error("Invalid model prefix: {0}")]
    InvalidPrefix(String),

    #[error("Invalid value: {0}")]
    InvalidValue(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let (status, code, message) = (self.status_code(), self.error_code(), self.to_string());

        #[derive(Serialize)]
        struct ErrorEnvelope {
            error: ErrorBody,
        }

        #[derive(Serialize)]
        struct ErrorBody {
            code: &'static str,
            message: String,
        }

        (
            status,
            axum::Json(ErrorEnvelope {
                error: ErrorBody { code, message },
            }),
        )
            .into_response()
    }
}

impl ProxyError {
    pub fn status_code(&self) -> StatusCode {
        match self {
            ProxyError::Validation(_) => StatusCode::BAD_REQUEST,
            ProxyError::Provider(_) => StatusCode::BAD_GATEWAY,
            ProxyError::Auth(_) => StatusCode::UNAUTHORIZED,
            ProxyError::Config(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ProxyError::Http(e) => e
                .status()
                .and_then(|s| StatusCode::from_u16(s.as_u16()).ok())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            ProxyError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub fn error_code(&self) -> &'static str {
        match self {
            ProxyError::Validation(_) => "bad_request",
            ProxyError::Provider(_) => "provider_error",
            ProxyError::Auth(_) => "auth_error",
            ProxyError::Config(_) => "config_error",
            ProxyError::Http(_) => "http_error",
            ProxyError::Internal(_) => "internal_server_error",
        }
    }
}

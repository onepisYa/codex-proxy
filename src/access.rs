use axum::http::HeaderMap;
use rand::RngCore;
use sha2::{Digest, Sha256};

use crate::config::{AccessKeyConfig, ConfigHandle, with_config};
use crate::error::ProxyError;

#[derive(Debug, Clone)]
pub struct AuthenticatedKey {
    pub id: String,
    pub is_admin: bool,
}

pub fn authenticate_request(
    config: &ConfigHandle,
    headers: &HeaderMap,
) -> Result<Option<AuthenticatedKey>, ProxyError> {
    let (require_key, keys, bootstrap_admin_key) = with_config(config, |cfg| {
        (
            cfg.access.require_key,
            cfg.access.keys.clone(),
            cfg.access.bootstrap_admin_key.clone(),
        )
    });
    let has_bootstrap_key = bootstrap_admin_key
        .as_ref()
        .is_some_and(|v| !v.trim().is_empty())
        || std::env::var("CODEX_PROXY_BOOTSTRAP_ADMIN_KEY")
            .ok()
            .is_some_and(|v| !v.trim().is_empty());
    if !require_key && keys.is_empty() && !has_bootstrap_key {
        return Ok(None);
    }

    let Some(presented) = extract_presented_key(headers) else {
        if require_key {
            return Err(ProxyError::Auth(
                "Missing access key. Provide x-codex-proxy-key, x-api-key, or Authorization: Bearer <key>."
                    .into(),
            ));
        }
        return Ok(None);
    };

    if let Some(bootstrap) = bootstrap_admin_key
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty())
    {
        if presented == bootstrap {
            return Ok(Some(AuthenticatedKey {
                id: "bootstrap".into(),
                is_admin: true,
            }));
        }
    } else if let Ok(bootstrap) = std::env::var("CODEX_PROXY_BOOTSTRAP_ADMIN_KEY")
        && !bootstrap.is_empty()
        && presented == bootstrap
    {
        return Ok(Some(AuthenticatedKey {
            id: "bootstrap".into(),
            is_admin: true,
        }));
    }

    let digest = sha256_hex(&presented);
    let matched = keys
        .into_iter()
        .find(|key| key.enabled && key.key_sha256 == digest);
    match matched {
        Some(AccessKeyConfig { id, is_admin, .. }) => Ok(Some(AuthenticatedKey { id, is_admin })),
        None if require_key => Err(ProxyError::Auth("Invalid access key".into())),
        None => Ok(None),
    }
}

pub fn require_admin(key: Option<&AuthenticatedKey>) -> Result<(), ProxyError> {
    match key {
        Some(k) if k.is_admin => Ok(()),
        _ => Err(ProxyError::Auth("Admin access key required".into())),
    }
}

pub fn extract_presented_key(headers: &HeaderMap) -> Option<String> {
    if let Some(value) = headers
        .get("x-codex-proxy-key")
        .and_then(|v| v.to_str().ok())
    {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    if let Some(value) = headers.get("x-api-key").and_then(|v| v.to_str().ok()) {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    if let Some(value) = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
    {
        let value = value.trim();
        if let Some(rest) = value.strip_prefix("Bearer ") {
            let trimmed = rest.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

pub fn sha256_hex(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    let digest = hasher.finalize();
    to_hex(&digest)
}

pub fn generate_access_key() -> String {
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    format!("cpk_{}", to_hex(&bytes))
}

fn to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(hex_char(b >> 4));
        out.push(hex_char(b & 0x0f));
    }
    out
}

fn hex_char(n: u8) -> char {
    match n {
        0..=9 => (b'0' + n) as char,
        10..=15 => (b'a' + (n - 10)) as char,
        _ => '?',
    }
}

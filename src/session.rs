use axum::http::{HeaderMap, HeaderName, HeaderValue};
use parking_lot::RwLock;
use rand::RngCore;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

#[derive(Clone, Debug)]
pub struct SessionKey {
    pub value: String,
}

impl SessionKey {
    pub fn generate() -> Self {
        let mut bytes = [0u8; 16];
        rand::thread_rng().fill_bytes(&mut bytes);
        Self {
            value: format!("cps_{}", to_hex(&bytes)),
        }
    }

    pub fn cache_key_override(&self) -> u64 {
        let mut hasher = Sha256::new();
        hasher.update(self.value.as_bytes());
        let digest = hasher.finalize();
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&digest[..8]);
        u64::from_be_bytes(bytes)
    }
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

#[derive(Default)]
pub struct SessionStore {
    by_response_id: RwLock<HashMap<String, (SessionKey, SystemTime)>>,
    ttl: RwLock<Duration>,
}

impl SessionStore {
    pub fn new(ttl: Duration) -> Self {
        Self {
            by_response_id: RwLock::new(HashMap::new()),
            ttl: RwLock::new(ttl),
        }
    }

    pub fn set_ttl(&self, ttl: Duration) {
        *self.ttl.write() = ttl;
    }

    pub fn get_by_previous_response_id(&self, id: &str) -> Option<SessionKey> {
        let ttl = *self.ttl.read();
        let now = SystemTime::now();
        let mut guard = self.by_response_id.write();
        guard.retain(|_, (_, created_at)| {
            now.duration_since(*created_at)
                .map(|age| age <= ttl)
                .unwrap_or(false)
        });
        guard.get(id).map(|(key, _)| key.clone())
    }

    pub fn record_response_id(&self, response_id: String, session_key: SessionKey) {
        self.by_response_id
            .write()
            .insert(response_id, (session_key, SystemTime::now()));
    }
}

pub fn extract_session_key_from_headers(
    headers: &HeaderMap,
    header_name: &str,
) -> Option<SessionKey> {
    let name = HeaderName::from_bytes(header_name.as_bytes()).ok()?;
    let raw = headers.get(name)?.to_str().ok()?.trim();
    (!raw.is_empty()).then_some(SessionKey {
        value: raw.to_string(),
    })
}

pub fn extract_session_key_from_metadata(
    metadata: Option<&Value>,
    metadata_key: &str,
) -> Option<SessionKey> {
    let obj = metadata?.as_object()?;
    let raw = obj.get(metadata_key)?.as_str()?.trim();
    (!raw.is_empty()).then_some(SessionKey {
        value: raw.to_string(),
    })
}

pub fn attach_session_header(headers: &mut HeaderMap, header_name: &str, session: &SessionKey) {
    if let Ok(name) = HeaderName::from_bytes(header_name.as_bytes())
        && let Ok(value) = HeaderValue::from_str(&session.value)
    {
        headers.insert(name, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_stable_cache_key_override() {
        let key = SessionKey {
            value: "cps_test".into(),
        };
        assert_eq!(key.cache_key_override(), key.cache_key_override());
    }

    #[test]
    fn extracts_session_key_from_headers() {
        let mut headers = HeaderMap::new();
        headers.insert("x-codex-proxy-session", "abc".parse().unwrap());
        let key = extract_session_key_from_headers(&headers, "x-codex-proxy-session").unwrap();
        assert_eq!(key.value, "abc");
    }

    #[test]
    fn extracts_session_key_from_metadata() {
        let metadata = serde_json::json!({"codex_proxy_session":"xyz"});
        let key =
            extract_session_key_from_metadata(Some(&metadata), "codex_proxy_session").unwrap();
        assert_eq!(key.value, "xyz");
    }
}

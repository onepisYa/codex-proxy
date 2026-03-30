use axum::http::StatusCode;
use parking_lot::RwLock;
use serde::Serialize;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Default)]
pub struct UsageStore {
    by_key: RwLock<HashMap<String, KeyUsage>>,
    by_account: RwLock<HashMap<String, AccountUsage>>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct KeyUsage {
    pub key_id: String,
    pub requests: u64,
    pub errors: u64,
    pub last_seen_at: Option<u64>,
    pub by_provider: HashMap<String, u64>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct AccountUsage {
    pub account_id: String,
    pub provider: String,
    pub requests: u64,
    pub errors: u64,
    pub cache_hits: u64,
    pub last_seen_at: Option<u64>,
    pub by_model: HashMap<String, u64>,
}

impl UsageStore {
    pub fn record_request_result(
        &self,
        key_id: Option<&str>,
        provider: &str,
        account_id: &str,
        model: &str,
        status: StatusCode,
        cache_hit: bool,
    ) {
        let now = now_secs();
        if let Some(key_id) = key_id {
            let mut guard = self.by_key.write();
            let entry = guard.entry(key_id.to_string()).or_insert_with(|| KeyUsage {
                key_id: key_id.to_string(),
                ..KeyUsage::default()
            });
            entry.requests += 1;
            if !status.is_success() {
                entry.errors += 1;
            }
            entry.last_seen_at = Some(now);
            *entry.by_provider.entry(provider.to_string()).or_insert(0) += 1;
        }

        let mut guard = self.by_account.write();
        let k = format!("{provider}:{account_id}");
        let entry = guard.entry(k).or_insert_with(|| AccountUsage {
            account_id: account_id.to_string(),
            provider: provider.to_string(),
            ..AccountUsage::default()
        });
        entry.requests += 1;
        if !status.is_success() {
            entry.errors += 1;
        }
        if cache_hit {
            entry.cache_hits += 1;
        }
        entry.last_seen_at = Some(now);
        *entry.by_model.entry(model.to_string()).or_insert(0) += 1;
    }

    pub fn snapshot_keys(&self) -> Vec<KeyUsage> {
        let guard = self.by_key.read();
        let mut out: Vec<KeyUsage> = guard.values().cloned().collect();
        out.sort_by(|a, b| a.key_id.cmp(&b.key_id));
        out
    }

    pub fn snapshot_accounts(&self) -> Vec<AccountUsage> {
        let guard = self.by_account.read();
        let mut out: Vec<AccountUsage> = guard.values().cloned().collect();
        out.sort_by(|a, b| {
            (a.provider.clone(), a.account_id.clone())
                .cmp(&(b.provider.clone(), b.account_id.clone()))
        });
        out
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

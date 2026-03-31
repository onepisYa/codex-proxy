use axum::http::StatusCode;
use parking_lot::RwLock;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Default)]
pub struct UsageStore {
    by_key: RwLock<HashMap<String, KeyUsage>>,
    by_account: RwLock<HashMap<String, AccountUsage>>,
    by_minute: RwLock<BTreeMap<u64, MinuteUsage>>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct KeyUsage {
    pub key_id: String,
    pub requests: u64,
    pub errors: u64,
    pub request_bytes: u64,
    pub response_bytes: u64,
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
    pub request_bytes: u64,
    pub response_bytes: u64,
    pub last_seen_at: Option<u64>,
    pub by_model: HashMap<String, u64>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct MinuteUsage {
    pub ts: u64,
    pub requests: u64,
    pub errors: u64,
    pub request_bytes: u64,
    pub response_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct UsageHandle {
    minute_ts: u64,
    key_id: Option<String>,
    provider: String,
    account_id: String,
}

const BASE_BUCKET_SECONDS: u64 = 60;
const RETENTION_SECONDS: u64 = 14 * 24 * 60 * 60;

impl UsageStore {
    fn record_request_start_at(
        &self,
        now: u64,
        minute_ts: u64,
        key_id: Option<&str>,
        provider: &str,
        account_id: &str,
        model: &str,
        status: StatusCode,
        cache_hit: bool,
        request_bytes: u64,
    ) {
        let is_error = !status.is_success();

        self.prune_old(now);

        if let Some(key_id) = key_id {
            let mut guard = self.by_key.write();
            let entry = guard.entry(key_id.to_string()).or_insert_with(|| KeyUsage {
                key_id: key_id.to_string(),
                ..KeyUsage::default()
            });
            entry.requests += 1;
            if is_error {
                entry.errors += 1;
            }
            entry.request_bytes += request_bytes;
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
        if is_error {
            entry.errors += 1;
        }
        if cache_hit {
            entry.cache_hits += 1;
        }
        entry.request_bytes += request_bytes;
        entry.last_seen_at = Some(now);
        *entry.by_model.entry(model.to_string()).or_insert(0) += 1;

        let mut guard = self.by_minute.write();
        let entry = guard.entry(minute_ts).or_insert_with(|| MinuteUsage {
            ts: minute_ts,
            ..MinuteUsage::default()
        });
        entry.requests += 1;
        if is_error {
            entry.errors += 1;
        }
        entry.request_bytes += request_bytes;
    }

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
        let minute_ts = now - (now % BASE_BUCKET_SECONDS);
        self.record_request_start_at(
            now, minute_ts, key_id, provider, account_id, model, status, cache_hit, 0,
        );
    }

    pub fn record_request_start_handle(
        &self,
        key_id: Option<&str>,
        provider: &str,
        account_id: &str,
        model: &str,
        status: StatusCode,
        cache_hit: bool,
        request_bytes: u64,
    ) -> UsageHandle {
        let now = now_secs();
        let minute_ts = now - (now % BASE_BUCKET_SECONDS);
        self.record_request_start_at(
            now,
            minute_ts,
            key_id,
            provider,
            account_id,
            model,
            status,
            cache_hit,
            request_bytes,
        );
        UsageHandle {
            minute_ts,
            key_id: key_id.map(|v| v.to_string()),
            provider: provider.to_string(),
            account_id: account_id.to_string(),
        }
    }

    pub fn record_response_bytes(&self, handle: &UsageHandle, response_bytes: u64) {
        if response_bytes == 0 {
            return;
        }
        let now = now_secs();
        self.prune_old(now);

        if let Some(key_id) = &handle.key_id {
            let mut guard = self.by_key.write();
            if let Some(entry) = guard.get_mut(key_id) {
                entry.response_bytes += response_bytes;
                entry.last_seen_at = Some(now);
            }
        }

        let mut guard = self.by_account.write();
        let k = format!("{}:{}", handle.provider, handle.account_id);
        if let Some(entry) = guard.get_mut(&k) {
            entry.response_bytes += response_bytes;
            entry.last_seen_at = Some(now);
        }

        let mut guard = self.by_minute.write();
        let entry = guard
            .entry(handle.minute_ts)
            .or_insert_with(|| MinuteUsage {
                ts: handle.minute_ts,
                ..MinuteUsage::default()
            });
        entry.response_bytes += response_bytes;
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

    pub fn snapshot_series(&self, bucket_seconds: u64, window_seconds: u64) -> Vec<MinuteUsage> {
        let now = now_secs();
        let bucket_seconds = bucket_seconds.max(BASE_BUCKET_SECONDS);
        let window_seconds = window_seconds.max(bucket_seconds).min(RETENTION_SECONDS);
        let end = now - (now % bucket_seconds) + bucket_seconds;
        let start = end.saturating_sub(window_seconds);

        let mut buckets: BTreeMap<u64, MinuteUsage> = BTreeMap::new();
        let mut ts = start - (start % bucket_seconds);
        while ts < end {
            buckets.insert(
                ts,
                MinuteUsage {
                    ts,
                    ..MinuteUsage::default()
                },
            );
            ts += bucket_seconds;
        }

        let guard = self.by_minute.read();
        for (minute_ts, item) in guard.range(start..end) {
            let ts = minute_ts - (minute_ts % bucket_seconds);
            let entry = buckets.entry(ts).or_insert_with(|| MinuteUsage {
                ts,
                ..MinuteUsage::default()
            });
            entry.requests += item.requests;
            entry.errors += item.errors;
            entry.request_bytes += item.request_bytes;
            entry.response_bytes += item.response_bytes;
        }

        buckets.into_values().collect()
    }

    fn prune_old(&self, now: u64) {
        let cutoff = now.saturating_sub(RETENTION_SECONDS);
        let mut guard = self.by_minute.write();
        while let Some((&ts, _)) = guard.first_key_value() {
            if ts >= cutoff {
                break;
            }
            guard.pop_first();
        }
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

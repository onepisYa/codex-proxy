use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crate::account_pool::{AccountPool, RoutingState};
use crate::auth::GeminiAuthManager;
use crate::config::ConfigHandle;
use crate::model_catalog::ModelCatalog;
use crate::providers::ProviderRegistry;
use crate::session::SessionStore;
use crate::usage::UsageStore;

#[derive(Clone)]
pub struct AppState {
    inner: Arc<AppStateInner>,
}

struct AppStateInner {
    pub config: ConfigHandle,
    pub accounts: AccountPool,
    pub routing: RoutingState,
    pub usage: UsageStore,
    pub providers: ProviderRegistry,
    pub gemini_auth: Arc<GeminiAuthManager>,
    pub recovery_probes_started: AtomicBool,
    pub model_discovery_started: AtomicBool,
    pub model_catalog: ModelCatalog,
    pub sessions: SessionStore,
}

impl AppState {
    pub fn new(config: ConfigHandle) -> Self {
        let ttl_seconds =
            crate::config::with_config(&config, |cfg| cfg.session.response_id_ttl_seconds);
        let sessions = SessionStore::new(std::time::Duration::from_secs(ttl_seconds));
        Self {
            inner: Arc::new(AppStateInner {
                config: config.clone(),
                accounts: AccountPool::new(),
                routing: RoutingState::new(),
                usage: UsageStore::default(),
                providers: ProviderRegistry::new(),
                gemini_auth: Arc::new(GeminiAuthManager::new(config.clone())),
                recovery_probes_started: AtomicBool::new(false),
                model_discovery_started: AtomicBool::new(false),
                model_catalog: ModelCatalog::new(),
                sessions,
            }),
        }
    }

    pub fn config(&self) -> &ConfigHandle {
        &self.inner.config
    }

    pub fn accounts(&self) -> &AccountPool {
        &self.inner.accounts
    }

    pub fn routing(&self) -> &RoutingState {
        &self.inner.routing
    }

    pub fn usage(&self) -> &UsageStore {
        &self.inner.usage
    }

    pub fn providers(&self) -> &ProviderRegistry {
        &self.inner.providers
    }

    pub fn gemini_auth(&self) -> Arc<GeminiAuthManager> {
        self.inner.gemini_auth.clone()
    }

    pub fn recovery_started_flag(&self) -> &AtomicBool {
        &self.inner.recovery_probes_started
    }

    pub fn model_discovery_started_flag(&self) -> &AtomicBool {
        &self.inner.model_discovery_started
    }

    pub fn model_catalog(&self) -> &ModelCatalog {
        &self.inner.model_catalog
    }

    pub fn sessions(&self) -> &SessionStore {
        &self.inner.sessions
    }
}

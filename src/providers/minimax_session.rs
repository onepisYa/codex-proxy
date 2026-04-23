use std::collections::HashMap;
use tracing::debug;

/// Per-request session for translating between Codex/OpenAI and Anthropic/MiniMax.
/// Replaces thread_local storage to avoid race conditions in concurrent requests.
///
/// Stores thinking signatures for multi-turn roundtrip.
#[derive(Debug, Default)]
pub struct TranslationSession {
    /// Stores thinking signatures by item_id (for multi-turn roundtrip)
    thinking_signatures: HashMap<String, String>,
}

impl TranslationSession {
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a thinking signature for an item_id
    pub fn store_thinking_signature(&mut self, item_id: &str, signature: &str) {
        if !signature.is_empty() {
            debug!(
                "TranslationSession: stored thinking_signature for item_id={}",
                item_id
            );
            self.thinking_signatures
                .insert(item_id.to_string(), signature.to_string());
        }
    }

    /// Look up a thinking signature by item_id
    pub fn lookup_thinking_signature(&self, item_id: &str) -> Option<String> {
        self.thinking_signatures.get(item_id).cloned()
    }
}

use std::collections::HashMap;
use tracing::debug;

/// Per-request session for translating between Codex/OpenAI and Anthropic/MiniMax.
/// Replaces thread_local storage to avoid race conditions in concurrent requests.
///
/// Stores:
/// - Codex call_id → our tool_use.id mapping (from function_call translation)
/// - MiniMax call_id → our tool_use.id mapping (from SSE processing)
/// - Our generated tool_use.id sequence (for SSE correlation)
/// - Thinking signatures (for multi-turn roundtrip)
#[derive(Debug, Default)]
pub struct TranslationSession {
    /// Maps Codex call_id (from function_call.call_id) to our tool_use.id
    codex_call_id_map: HashMap<String, String>,
    /// Maps MiniMax call_id (from SSE) to our original tool_use.id
    minimax_call_id_map: HashMap<String, String>,
    /// Our tool_use.id sequence (ordered as we sent to MiniMax)
    our_tool_use_ids: Vec<String>,
    /// Counter for tracking which tool_use we're processing during SSE
    tool_use_counter: usize,
    /// Stores thinking signatures by item_id (for multi-turn roundtrip)
    thinking_signatures: HashMap<String, String>,
}

impl TranslationSession {
    pub fn new() -> Self {
        Self::default()
    }

    /// Store mapping from Codex call_id to our tool_use.id
    pub fn store_codex_call_id(&mut self, codex_id: &str, our_id: &str) {
        debug!(
            "TranslationSession: stored codex_call_id mapping: codex_id={} -> our_id={}",
            codex_id, our_id
        );
        self.codex_call_id_map
            .insert(codex_id.to_string(), our_id.to_string());
    }

    /// Store our tool_use.id sequence before sending to MiniMax
    pub fn set_our_tool_use_ids(&mut self, ids: Vec<String>) {
        debug!("TranslationSession: stored {} tool_use.ids", ids.len());
        self.our_tool_use_ids = ids;
    }

    /// Called during SSE processing when we receive tool_use from MiniMax.
    /// Maps MiniMax's call_id -> our tool_use.id (by position).
    pub fn store_minimax_call_id(&mut self, minimax_call_id: &str) {
        let our_id = self
            .our_tool_use_ids
            .get(self.tool_use_counter)
            .cloned()
            .unwrap_or_else(|| format!("unknown_tool_{}", self.tool_use_counter));

        debug!(
            "TranslationSession: stored minimax_call_id mapping: minimax_call_id={} -> our_id={}",
            minimax_call_id, our_id
        );
        self.minimax_call_id_map
            .insert(minimax_call_id.to_string(), our_id.clone());

        // Increment counter for next tool_use
        self.tool_use_counter += 1;
    }

    /// Look up our original tool_use.id given a call_id.
    /// First tries Codex call_id mapping, falls back to MiniMax call_id mapping.
    pub fn lookup_our_tool_use_id(&self, call_id: &str) -> Option<String> {
        self.codex_call_id_map
            .get(call_id)
            .cloned()
            .or_else(|| self.minimax_call_id_map.get(call_id).cloned())
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

//! Discovered context windows per provider/model, populated at runtime by
//! `list_models()`. Not persisted. Used to override the fallback for providers
//! that accept arbitrary models (e.g. llama-cpp router mode, ollama).

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

use crate::provider::ProviderKind;

static CONTEXT_WINDOWS: OnceLock<RwLock<HashMap<(ProviderKind, String), u32>>> = OnceLock::new();

fn map() -> &'static RwLock<HashMap<(ProviderKind, String), u32>> {
    CONTEXT_WINDOWS.get_or_init(|| RwLock::new(HashMap::new()))
}

pub fn set_context_window(provider: ProviderKind, model_id: String, window: u32) {
    map().write().unwrap().insert((provider, model_id), window);
}

pub fn get_context_window(provider: ProviderKind, model_id: &str) -> Option<u32> {
    map().read().unwrap().get(&(provider, model_id.to_string())).copied()
}

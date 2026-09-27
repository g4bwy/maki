use std::sync::Arc;

use maki_storage::id::SessionRef;
use smol::lock::Mutex;

use crate::providers::openai::routing::RoutingState;
use crate::providers::openai::websocket::ResponsesSession;

/// Provider state owned by one conversation. It is handed to every request a
/// session makes, including the ones a subagent or a compaction makes on its
/// behalf, so a provider can keep whatever affinity it needs across turns.
///
/// The identity is separate from the session file: a child gets a fresh
/// `thread_id` so a subagent is its own upstream thread, while the cache key
/// stays the parent's because the shared prompt prefix is what gets cached.
#[derive(Clone)]
pub struct ProviderSession {
    inner: Arc<SessionState>,
}

struct SessionState {
    session_ref: SessionRef,
    thread_id: SessionRef,
    cache_key: String,
    routing: RoutingState,
    responses: Mutex<ResponsesSession>,
}

impl ProviderSession {
    pub fn new(session_ref: SessionRef) -> Self {
        Self::with_identity(
            session_ref.clone(),
            session_ref.clone(),
            session_ref.as_str().into(),
        )
    }

    fn with_identity(session_ref: SessionRef, thread_id: SessionRef, cache_key: String) -> Self {
        Self {
            inner: Arc::new(SessionState {
                session_ref,
                thread_id,
                cache_key,
                routing: RoutingState::default(),
                responses: Mutex::new(ResponsesSession::default()),
            }),
        }
    }

    /// A subagent's session: same conversation and cache prefix, own thread.
    pub fn child(&self) -> Self {
        Self::with_identity(
            self.inner.session_ref.clone(),
            SessionRef::generate(),
            self.cache_key().into(),
        )
    }

    pub fn session_ref(&self) -> &SessionRef {
        &self.inner.session_ref
    }

    pub fn thread_id(&self) -> &str {
        self.inner.thread_id.as_str()
    }

    pub fn cache_key(&self) -> &str {
        &self.inner.cache_key
    }

    /// One user turn, which may be several requests: retries, a subagent, a
    /// summary. Whatever the upstream handed the turn first is what the rest of
    /// it has to keep sending, so this is where that state is dropped.
    pub async fn begin_turn(&self) {
        // Held for the lock alone: a turn already on the socket must finish
        // writing its continuation state before a new one may clear it.
        let _responses = self.inner.responses.lock().await;
        self.inner.routing.clear();
    }

    pub(crate) fn routing(&self) -> &RoutingState {
        &self.inner.routing
    }

    pub(crate) fn responses(&self) -> &Mutex<ResponsesSession> {
        &self.inner.responses
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_child_keeps_the_cache_prefix_and_starts_a_new_thread() {
        let parent = ProviderSession::new(SessionRef::generate());
        let child = parent.child();
        assert_eq!(child.cache_key(), parent.cache_key());
        assert_eq!(child.session_ref(), parent.session_ref());
        assert_ne!(child.thread_id(), parent.thread_id());
        assert_ne!(child.thread_id(), child.session_ref().as_str());

        let sibling = child.child();
        assert_eq!(sibling.cache_key(), parent.cache_key());
        assert_ne!(sibling.thread_id(), child.thread_id());
    }
}

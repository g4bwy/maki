use std::sync::Arc;

use arc_swap::ArcSwap;
use futures_lite::{
    StreamExt,
    io::{AsyncBufReadExt, BufReader},
};
use isahc::Request;
use isahc::config::Configurable;
use serde::Deserialize;
use tracing::{debug, info, warn};

use super::{ResolvedAuth, user_agent};
use crate::providers::Timeouts;

/// Model loading progress state, shared via ArcSwap between SSE task and UI.
#[derive(Debug, Clone, Default)]
pub struct ModelLoadingState {
    pub model: String,
    pub stage: String,
    pub stages: Vec<String>,
    pub progress: f32,
    pub status: LoadingStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoadingStatus {
    #[default]
    Idle,
    Loading,
    Ready,
    Failed,
}

#[derive(Deserialize)]
struct SseEnvelope {
    #[serde(default)]
    model: String,
    #[serde(rename = "event")]
    event_type: String,
    #[serde(default)]
    data: Option<serde_json::Value>,
}

pub fn sse_url(base_url: Option<&str>, compat_base_url: &str) -> Option<String> {
    let base = base_url.unwrap_or(compat_base_url);
    let root = base.strip_suffix("/v1").unwrap_or(base);
    Some(format!("{root}/models/sse"))
}

async fn check_health(base_url: &str, auth: &ResolvedAuth) -> Option<LoadingStatus> {
    let health_url = format!("{base_url}/health");
    let client = isahc::HttpClient::builder()
        .connect_timeout(Timeouts::default().connect)
        .build()
        .ok()?;

    let mut request = Request::builder()
        .method("GET")
        .uri(&health_url)
        .header("user-agent", user_agent());

    for (key, value) in &auth.headers {
        request = request.header(key, value);
    }

    let request = request.body(()).ok()?;

    match client.send_async(request).await {
        Ok(response) => {
            if response.status().is_success() {
                info!(url = %health_url, "SSE: /health OK, model ready");
                Some(LoadingStatus::Ready)
            } else {
                info!(url = %health_url, status = %response.status(), "SSE: /health not ready");
                Some(LoadingStatus::Loading)
            }
        }
        Err(e) => {
            warn!(url = %health_url, error = %e, "SSE: /health check failed");
            None
        }
    }
}

const SSE_RECONNECT_INTERVAL: std::time::Duration = std::time::Duration::from_secs(2);

pub async fn sse_loop(sse_url: &str, auth: &ResolvedAuth, state: Arc<ArcSwap<ModelLoadingState>>) {
    let health_url = sse_url.strip_suffix("/models/sse").unwrap_or(sse_url);

    loop {
        info!(url = sse_url, "SSE: connecting to llama.cpp");

        if let Some(initial) = check_health(health_url, auth).await {
            state.rcu(|old| {
                let mut s = (**old).clone();
                s.status = initial;
                info!(?initial, "SSE: initial state from /health");
                s
            });
        }

        let client = isahc::HttpClient::builder()
            .connect_timeout(Timeouts::default().connect)
            .low_speed_timeout(1, std::time::Duration::from_secs(60))
            .build()
            .unwrap_or_else(|_| {
                panic!("failed to build HTTP client for SSE");
            });

        let request = auth
            .configure_request(
                Request::builder()
                    .method("GET")
                    .uri(sse_url)
                    .header("accept", "text/event-stream")
                    .header("user-agent", user_agent()),
            )
            .body(())
            .unwrap_or_else(|e| panic!("failed to build SSE request: {e}"));

        let response = match client.send_async(request).await {
            Ok(r) => {
                info!(url = sse_url, status = %r.status(), "SSE: connected");
                r
            }
            Err(e) => {
                info!(url = sse_url, error = %e, "SSE: connection failed, reconnecting");
                smol::Timer::after(SSE_RECONNECT_INTERVAL).await;
                continue;
            }
        };

        if !response.status().is_success() {
            info!(url = sse_url, status = %response.status(), "SSE: non-success response, reconnecting");
            smol::Timer::after(SSE_RECONNECT_INTERVAL).await;
            continue;
        }

        let reader = BufReader::new(response.into_body());
        let mut lines = reader.lines();

        let mut event_type = String::new();
        let mut data_lines = Vec::<String>::new();

        loop {
            let Some(Ok(line)) = lines.next().await else {
                info!(url = sse_url, "SSE: stream ended or error, reconnecting");
                break;
            };

            let trimmed = line.trim();
            if trimmed.is_empty() {
                if !data_lines.is_empty() {
                    let payload = data_lines.join("\n");
                    let mut envelope: SseEnvelope = match serde_json::from_str(&payload) {
                        Ok(e) => e,
                        Err(e) => {
                            debug!(payload = %payload, error = %e, "SSE: failed to parse event");
                            continue;
                        }
                    };
                    if !event_type.is_empty() {
                        envelope.event_type = event_type.clone();
                    }
                    info!(event = %envelope.event_type, model = %envelope.model, "SSE: received event");
                    process_event(&envelope, &state);
                }
                event_type.clear();
                data_lines.clear();
                continue;
            }

            if let Some(value) = trimmed.strip_prefix("data: ") {
                data_lines.push(value.to_string());
            } else if let Some(value) = trimmed.strip_prefix("data:") {
                data_lines.push(value.to_string());
            } else if let Some(value) = trimmed.strip_prefix("event: ") {
                event_type = value.to_string();
            } else if let Some(value) = trimmed.strip_prefix("event:") {
                event_type = value.to_string();
            } else {
                debug!(line = %trimmed, "SSE: unknown line format");
            }
        }

        smol::Timer::after(SSE_RECONNECT_INTERVAL).await;
    }
}

fn process_event(envelope: &SseEnvelope, state: &Arc<ArcSwap<ModelLoadingState>>) {
    let data = match &envelope.data {
        Some(d) => d,
        None => return,
    };

    match envelope.event_type.as_str() {
        "model_status" => {
            if let Some(status) = data.get("status").and_then(|v| v.as_str())
                && status == "loading"
            {
                state.rcu(|old| {
                    let mut s = (**old).clone();
                    s.model = envelope.model.clone();
                    s.status = LoadingStatus::Loading;
                    s.progress = 0.0;
                    s.stage = String::new();
                    s.stages = Vec::new();
                    info!(model = %s.model, "SSE: state -> Loading");
                    s
                });
            }
        }
        "status_change" => {
            let status = match data.get("status").and_then(|v| v.as_str()) {
                Some(s) => s,
                None => return,
            };

            match status {
                "loading" => {
                    if let Some(progress) = data.get("progress") {
                        let value = progress
                            .get("value")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0);
                        let stage = progress
                            .get("current")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let stages: Vec<String> = progress
                            .get("stages")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str())
                                    .map(String::from)
                                    .collect()
                            })
                            .unwrap_or_default();

                        state.rcu(|old| {
                            let mut s = (**old).clone();
                            s.status = LoadingStatus::Loading;
                            s.progress = value as f32;
                            s.stage = stage.clone();
                            s.stages = stages.clone();
                            info!(progress = value, stage = %stage, "SSE: progress update");
                            s
                        });
                    }
                }
                "loaded" => {
                    state.rcu(|old| {
                        let mut s = (**old).clone();
                        s.status = LoadingStatus::Ready;
                        s.progress = 1.0;
                        info!("SSE: state -> Ready");
                        s
                    });
                }
                "unloaded" => {
                    state.rcu(|old| {
                        let mut s = (**old).clone();
                        s.status = LoadingStatus::Failed;
                        info!("SSE: state -> Failed");
                        s
                    });
                }
                _ => {}
            }
        }
        _ => {}
    }
}

#[cfg(test)]
fn reset_state(state: &Arc<ArcSwap<ModelLoadingState>>) {
    state.rcu(|old| {
        let mut s = (**old).clone();
        s.status = LoadingStatus::Idle;
        s
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    #[test_case(Some("http://localhost:8080/v1"), "http://localhost:8080/v1", "http://localhost:8080/models/sse" ; "with_v1_suffix")]
    #[test_case(Some("http://localhost:8080"), "http://localhost:8080/v1", "http://localhost:8080/models/sse" ; "without_v1_suffix")]
    #[test_case(None, "http://localhost:8080/v1", "http://localhost:8080/models/sse" ; "fallback_to_compat")]
    fn sse_url_builds_correctly(base_url: Option<&str>, compat: &str, expected: &str) {
        assert_eq!(sse_url(base_url, compat).as_deref(), Some(expected));
    }

    #[test]
    fn model_loading_state_defaults_to_idle() {
        let state = ModelLoadingState::default();
        assert_eq!(state.status, LoadingStatus::Idle);
        assert_eq!(state.progress, 0.0);
        assert!(state.model.is_empty());
    }

    #[test]
    fn model_loading_state_clone_works() {
        let state = ModelLoadingState {
            model: "test".into(),
            stage: "text_model".into(),
            stages: vec!["text_model".into()],
            progress: 0.5,
            status: LoadingStatus::Loading,
        };
        let cloned = state.clone();
        assert_eq!(cloned.model, "test");
        assert_eq!(cloned.status, LoadingStatus::Loading);
    }

    mod process_event {
        use super::super::*;

        fn make_envelope(event: &str, data: serde_json::Value) -> SseEnvelope {
            SseEnvelope {
                model: "test-model".into(),
                event_type: event.into(),
                data: Some(data),
            }
        }

        #[test]
        fn model_status_loading_sets_loading() {
            let state = Arc::new(ArcSwap::<ModelLoadingState>::default());
            let envelope = make_envelope("model_status", serde_json::json!({"status": "loading"}));
            process_event(&envelope, &state);
            let s = state.load();
            assert_eq!(s.status, LoadingStatus::Loading);
            assert_eq!(s.model, "test-model");
            assert_eq!(s.progress, 0.0);
        }

        #[test]
        fn status_change_loading_updates_progress() {
            let state = Arc::new(ArcSwap::<ModelLoadingState>::default());
            let envelope = make_envelope(
                "status_change",
                serde_json::json!({
                    "status": "loading",
                    "progress": {
                        "stages": ["text_model", "spec_model"],
                        "current": "text_model",
                        "value": 0.45
                    }
                }),
            );
            process_event(&envelope, &state);
            let s = state.load();
            assert_eq!(s.status, LoadingStatus::Loading);
            assert_eq!(s.progress, 0.45);
            assert_eq!(s.stage, "text_model");
            assert_eq!(s.stages.len(), 2);
        }

        #[test]
        fn status_change_loaded_sets_ready() {
            let state = Arc::new(ArcSwap::<ModelLoadingState>::default());
            let envelope = make_envelope(
                "status_change",
                serde_json::json!({"status": "loaded", "info": {}}),
            );
            process_event(&envelope, &state);
            let s = state.load();
            assert_eq!(s.status, LoadingStatus::Ready);
            assert_eq!(s.progress, 1.0);
        }

        #[test]
        fn status_change_unloaded_sets_failed() {
            let state = Arc::new(ArcSwap::<ModelLoadingState>::default());
            let envelope = make_envelope(
                "status_change",
                serde_json::json!({"status": "unloaded", "exit_code": 1}),
            );
            process_event(&envelope, &state);
            let s = state.load();
            assert_eq!(s.status, LoadingStatus::Failed);
        }

        #[test]
        fn unknown_event_ignored() {
            let state = Arc::new(ArcSwap::<ModelLoadingState>::default());
            let envelope = make_envelope("unknown_event", serde_json::json!({"foo": "bar"}));
            process_event(&envelope, &state);
            let s = state.load();
            assert_eq!(s.status, LoadingStatus::Idle);
        }

        #[test]
        fn missing_data_ignored() {
            let state = Arc::new(ArcSwap::<ModelLoadingState>::default());
            let envelope = SseEnvelope {
                model: "test".into(),
                event_type: "model_status".into(),
                data: None,
            };
            process_event(&envelope, &state);
            let s = state.load();
            assert_eq!(s.status, LoadingStatus::Idle);
        }

        #[test]
        fn malformed_json_does_not_panic() {
            let line = r#"{"model": "test", "event": "model_status"}"#;
            let envelope: Result<SseEnvelope, _> = serde_json::from_str(line);
            assert!(envelope.is_ok());
        }

        #[test]
        fn status_change_loading_missing_progress_ignored() {
            let state = Arc::new(ArcSwap::<ModelLoadingState>::default());
            let envelope = make_envelope("status_change", serde_json::json!({"status": "loading"}));
            process_event(&envelope, &state);
            let s = state.load();
            assert_eq!(s.status, LoadingStatus::Idle);
        }
    }

    mod reset_state {
        use super::super::*;

        #[test]
        fn resets_to_idle_preserving_model() {
            let state = Arc::new(ArcSwap::from_pointee(ModelLoadingState {
                model: "test".into(),
                stage: "text_model".into(),
                stages: vec![],
                progress: 0.5,
                status: LoadingStatus::Loading,
            }));
            reset_state(&state);
            let s = state.load();
            assert_eq!(s.status, LoadingStatus::Idle);
            assert_eq!(s.model, "test");
        }
    }
}

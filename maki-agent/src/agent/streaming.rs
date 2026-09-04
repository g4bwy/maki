use std::time::{Duration, Instant};

use maki_providers::provider::Provider;
use maki_providers::retry::{MAX_TIMEOUT_RETRIES, RetryState};
use maki_providers::{
    ContentBlock, Message, Model, ProviderEvent, RequestOptions, StreamResponse,
    estimate_prompt_tokens,
};
use maki_storage::id::SessionRef;
use serde_json::Value;
use tracing::warn;

use crate::cancel::CancelToken;
use crate::{AgentError, AgentEvent, EventSender};

const FUNCTIONS_PREFIX: &str = "functions.";

/// GPT models sometimes emit `functions.<name>`, a Codex training habit.
/// Stripped here at the provider boundary so no raw name enters the agent;
/// the batch plugin mirrors the rule in Lua.
pub(crate) fn canonical_tool_name(name: &str) -> &str {
    name.strip_prefix(FUNCTIONS_PREFIX).unwrap_or(name)
}

fn canonicalize_tool_names(message: &mut Message) {
    for block in &mut message.content {
        if let ContentBlock::ToolUse { name, .. } = block {
            *name = canonical_tool_name(name).to_owned();
        }
    }
}

async fn forward_provider_events(
    prx: flume::Receiver<ProviderEvent>,
    event_tx: &EventSender,
) -> String {
    let mut streamed = String::new();
    while let Ok(pe) = prx.recv_async().await {
        let ae = match pe {
            ProviderEvent::TextDelta { text } => {
                streamed.push_str(&text);
                AgentEvent::TextDelta { text }
            }
            ProviderEvent::ThinkingDelta { text } => AgentEvent::ThinkingDelta { text },
            ProviderEvent::ToolUseStart { id, name } => AgentEvent::ToolPending {
                id,
                name: canonical_tool_name(&name).to_owned(),
            },
            ProviderEvent::PromptProgress {
                processed,
                total,
                cache,
            } => AgentEvent::PromptProgress {
                processed,
                total,
                cache,
            },
        };
        if event_tx.send(ae).is_err() {
            break;
        }
    }
    streamed
}

/// Cancelling mid-stream carries the text the user still sees on screen,
/// so the caller can keep it in history. A cancel during the retry backoff
/// carries nothing: the `Retry` event already made the view drop the failed
/// attempt's text (`stream_reset`), and history must agree with the view.
#[derive(Debug)]
pub(crate) enum StreamError {
    Cancelled { streamed: String },
    Other(AgentError),
}

impl From<AgentError> for StreamError {
    fn from(e: AgentError) -> Self {
        Self::Other(e)
    }
}

impl From<StreamError> for AgentError {
    fn from(e: StreamError) -> Self {
        match e {
            StreamError::Cancelled { .. } => Self::Cancelled,
            StreamError::Other(e) => e,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn stream_with_retry(
    provider: &dyn Provider,
    model: &Model,
    messages: &[Message],
    system: &str,
    tools: &Value,
    event_tx: &EventSender,
    cancel: &CancelToken,
    opts: RequestOptions,
    session_id: Option<&SessionRef>,
    context_floor: u32,
) -> Result<StreamResponse, StreamError> {
    let opts = opts.clamped(model);
    let messages = maki_providers::adapt_images_for_model(model, messages);
    let messages = &*messages;
    // Strict servers 400 on `prompt + max_tokens > context window`, and on the
    // first request of a resumed session nothing was measured yet: take the
    // better of the last measured size and what is about to be sent.
    let mut measured_prompt = context_floor.max(estimate_prompt_tokens(messages, system, tools));
    let mut request_model = model.with_output_budget(measured_prompt);
    let mut overflow_recovery = true;
    let mut retry = RetryState::new();
    loop {
        let started = Instant::now();
        let (ptx, prx) = flume::unbounded();
        let forwarder = smol::spawn({
            let event_tx = event_tx.clone();
            async move { forward_provider_events(prx, &event_tx).await }
        });
        let result = futures_lite::future::race(
            provider.stream_message(
                &request_model,
                messages,
                system,
                tools,
                &ptx,
                opts,
                session_id,
            ),
            async {
                cancel.cancelled().await;
                Err(AgentError::Cancelled)
            },
        )
        .await;
        drop(ptx);
        let streamed = forwarder.await;
        match result {
            Ok(mut r) => {
                canonicalize_tool_names(&mut r.message);
                emit_api_request(&request_model, &r, opts, started.elapsed());
                return Ok(r);
            }
            Err(AgentError::Cancelled) => return Err(StreamError::Cancelled { streamed }),
            Err(e) if e.is_retryable() => {
                emit_api_error(&request_model, &e, retry.attempts() + 1, started.elapsed());
                if e.should_rotate_key()
                    && let Ok(true) = provider.rotate_key().await
                {
                    warn!("rotated API key after error: {e}");
                }
                let (attempt, delay) = retry.next_delay();
                if matches!(e, AgentError::Timeout { .. }) && attempt > MAX_TIMEOUT_RETRIES {
                    return Err(e.into());
                }
                let delay_ms = delay.as_millis() as u64;
                warn!(attempt, delay_ms, error = %e, "retryable, will retry");
                event_tx.send(AgentEvent::Retry {
                    attempt,
                    message: e.retry_message(),
                    delay_ms,
                })?;
                futures_lite::future::race(
                    async {
                        smol::Timer::after(delay).await;
                    },
                    cancel.cancelled(),
                )
                .await;
                if cancel.is_cancelled() {
                    return Err(StreamError::Cancelled {
                        streamed: String::new(),
                    });
                }
            }
            Err(e) => {
                // Some strict servers report the exact prompt length in the
                // rejection; re-clamp against it once instead of failing the
                // run because the local gauge ran optimistic.
                if overflow_recovery
                    && e.is_context_overflow()
                    && let Some(n) = e.reported_prompt_tokens()
                    && n > measured_prompt
                {
                    warn!(
                        reported = n,
                        gauge = measured_prompt,
                        "server measured a longer prompt, re-clamping output budget"
                    );
                    measured_prompt = n;
                    request_model = model.with_output_budget(n);
                    overflow_recovery = false;
                    continue;
                }
                emit_api_error(&request_model, &e, retry.attempts() + 1, started.elapsed());
                return Err(e.into());
            }
        }
    }
}

fn emit_api_request(model: &Model, r: &StreamResponse, opts: RequestOptions, took: Duration) {
    if !maki_otel::enabled() {
        return;
    }
    let usage = &r.usage;
    maki_otel::emit::api_request(&maki_otel::emit::ApiRequest {
        model: &model.id,
        provider: &model.provider,
        input_tokens: u64::from(usage.input),
        output_tokens: u64::from(usage.output),
        cache_read_tokens: u64::from(usage.cache_read),
        cache_creation_tokens: u64::from(usage.cache_creation),
        cost_usd: model.billed_cost(usage, opts.fast).unwrap_or(0.0),
        duration: took,
        stop_reason: r.stop_reason.map(<&'static str>::from),
    });
}

fn emit_api_error(model: &Model, error: &AgentError, attempt: u32, took: Duration) {
    if !maki_otel::enabled() {
        return;
    }
    maki_otel::emit::api_error(&maki_otel::emit::ApiError {
        model: &model.id,
        provider: &model.provider,
        error: &error_description(error),
        status_code: match error {
            AgentError::Api { status, .. } => Some(*status),
            _ => None,
        },
        attempt,
        duration: took,
    });
}

/// A provider's error body is often echoed request content (quoted message
/// text, masked keys, whatever a gateway returns), so only the status is
/// reported. Every other variant is generated locally.
fn error_description(error: &AgentError) -> String {
    match error {
        AgentError::Api { status, .. } => format!("API error ({status})"),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use maki_providers::Role;
    use serde_json::json;

    use super::*;

    const SECRET_BODY: &str = "messages.0.content: \"my private prompt\", key sk-abc";

    #[test]
    fn tool_use_names_canonicalized() {
        let mut message = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text { text: "hi".into() },
                ContentBlock::tool_use("t1", "functions.bash", json!({})),
                ContentBlock::tool_use("t2", "read", json!({})),
                ContentBlock::tool_use("t3", "my_functions.x", json!({})),
            ],
            ..Default::default()
        };
        canonicalize_tool_names(&mut message);
        let names: Vec<&str> = message.tool_uses().map(|(_, name, _)| name).collect();
        assert_eq!(names, ["bash", "read", "my_functions.x"]);
    }

    #[test]
    fn a_reported_api_error_leaves_the_provider_body_behind() {
        let error = AgentError::Api {
            status: 400,
            message: SECRET_BODY.into(),
        };
        let reported = error_description(&error);
        assert!(!reported.contains("private"));
        assert_eq!(reported, "API error (400)");
    }

    use std::sync::Mutex;

    use maki_providers::provider::{BoxFuture, Provider};
    use maki_providers::{StopReason, TokenUsage};

    /// Verbatim shape of the captured vLLM 400.
    const VLLM_REJECTION: &str = "This model's maximum context length is 262144 tokens. However, you requested 100000 output tokens and your prompt contains at least 162145 input tokens, for a total of at least 262145 tokens. (parameter=input_tokens, value=162145)";

    struct BudgetSpy {
        budgets: Mutex<Vec<Option<u32>>>,
        responses: Mutex<Vec<Result<StreamResponse, AgentError>>>,
    }

    impl Provider for BudgetSpy {
        fn stream_message<'a>(
            &'a self,
            model: &'a Model,
            _: &'a [Message],
            _: &'a str,
            _: &'a Value,
            _: &'a flume::Sender<ProviderEvent>,
            _: RequestOptions,
            _: Option<&maki_storage::id::SessionRef>,
        ) -> BoxFuture<'a, Result<StreamResponse, AgentError>> {
            Box::pin(async {
                self.budgets.lock().unwrap().push(model.max_output_tokens);
                self.responses.lock().unwrap().remove(0)
            })
        }

        fn list_models(&self) -> BoxFuture<'_, Result<Vec<maki_providers::ModelInfo>, AgentError>> {
            Box::pin(async { unimplemented!() })
        }
    }

    fn text_response() -> StreamResponse {
        StreamResponse {
            message: Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text { text: "ok".into() }],
                ..Default::default()
            },
            usage: TokenUsage::default(),
            stop_reason: Some(StopReason::EndTurn),
        }
    }

    #[test]
    fn overflow_rejection_reclamps_and_retries_once() {
        let spy = BudgetSpy {
            budgets: Mutex::new(Vec::new()),
            responses: Mutex::new(vec![
                Err(AgentError::Api {
                    status: 400,
                    message: VLLM_REJECTION.into(),
                }),
                Ok(text_response()),
            ]),
        };
        let mut model = Model::from_spec("openai/gpt-4o").unwrap();
        model.context_window = 262_144;
        model.max_output_tokens = Some(100_000);

        let (raw_tx, _raw_rx) = flume::unbounded();
        let event_tx = EventSender::new(raw_tx, 0);
        let result = smol::block_on(stream_with_retry(
            &spy,
            &model,
            &[],
            "",
            &Value::Array(vec![]),
            &event_tx,
            &CancelToken::none(),
            RequestOptions::default(),
            None,
            0,
        ));

        assert!(result.is_ok());
        assert_eq!(
            *spy.budgets.lock().unwrap(),
            vec![Some(100_000), Some(97_378)]
        );
    }
}

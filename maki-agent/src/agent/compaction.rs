use std::env;

use maki_providers::{ContentBlock, Message, Model, RequestOptions, TokenUsage};
use tracing::info;

use super::history::History;
use super::streaming::stream_with_retry;
use crate::cancel::CancelToken;
use crate::{AgentError, AgentEvent, EventSender, TurnCompleteEvent};

pub(super) const CONTINUE_AFTER_COMPACT: &str = "Continue if you have next steps, or stop and ask for clarification if you are unsure how to proceed. If you learned important project context during this session, consider saving it to memory before it's lost.";
const IMAGE_PLACEHOLDER: &str = "[image]";

/// Safety factor applied to the token budget to account for tokenizer mismatch.
///
/// The `estimate_message_tokens` heuristic uses `chars / 4`, which is calibrated
/// for Anthropic's cl100k_base tokenizer.  Llama-family tokenizers (and any
/// non-cl100k tokenizer) can produce significantly more tokens per character
/// for code-heavy content — observed ratios reach ~1.25×.  Multiplying the
/// budget by this factor ensures the *real* token count stays under the
/// context window even when the estimator is optimistic.
const ESTIMATOR_SAFETY_FACTOR: f64 = 0.8;

/// Estimated token cost of the accumulated summary message that gets prepended
/// to each multi-pass compaction request (after the first pass).  This is a
/// conservative upper bound — real summaries are usually smaller.
const SUMMARY_OVERHEAD_ESTIMATE: u32 = 5_000;

/// Rough estimate of tokens in a `Message`. 1 token ≈ 4 bytes for English
/// text (cl100k_base). Images are estimated at 768 tokens (Anthropic's
/// standard). This is intentionally approximate — it only needs to be
/// conservative enough to avoid sending requests that exceed the context
/// window.
fn estimate_message_tokens(msg: &Message) -> u32 {
    msg.content
        .iter()
        .map(|block| match block {
            ContentBlock::Text { text } | ContentBlock::ToolResult { content: text, .. } => {
                (text.len().max(1) as u32).div_ceil(4)
            }
            ContentBlock::ToolUse { input, .. } => {
                (input.to_string().len().max(1) as u32).div_ceil(4)
            }
            ContentBlock::Thinking { thinking, .. } => (thinking.len().max(1) as u32).div_ceil(4),
            ContentBlock::RedactedThinking { data } => (data.len().max(1) as u32).div_ceil(4),
            ContentBlock::Image { .. } => 768,
        })
        .sum()
}

fn estimate_messages_tokens(messages: &[Message]) -> u32 {
    messages.iter().map(estimate_message_tokens).sum()
}

/// Split `messages` into chunks where each chunk fits within `budget` tokens.
///
/// Messages are grouped from oldest to newest. If a single message exceeds
/// the budget it gets its own chunk (it will be handled by the last-resort
/// path in the caller). The returned chunks cover every input message —
/// nothing is dropped.
fn split_into_chunks(messages: &[Message], budget: u32) -> Vec<Vec<Message>> {
    let mut chunks: Vec<Vec<Message>> = Vec::new();
    let mut current: Vec<Message> = Vec::new();
    let mut current_tokens: u32 = 0;

    for msg in messages {
        let msg_tokens = estimate_message_tokens(msg);

        // If the message alone exceeds the budget, give it its own chunk.
        if msg_tokens > budget {
            if !current.is_empty() {
                chunks.push(std::mem::take(&mut current));
            }
            chunks.push(vec![msg.clone()]);
            continue;
        }

        // If adding this message would exceed the budget, flush the current
        // chunk and start a new one.
        if current_tokens + msg_tokens > budget && !current.is_empty() {
            chunks.push(std::mem::take(&mut current));
            current_tokens = 0;
        }

        current.push(msg.clone());
        current_tokens += msg_tokens;
    }

    if !current.is_empty() {
        chunks.push(current);
    }

    chunks
}

pub(super) async fn compact_history(
    provider: &dyn maki_providers::provider::Provider,
    model: &Model,
    history: &mut History,
    event_tx: &EventSender,
    cancel: &CancelToken,
    compaction_buffer: u32,
) -> Result<TokenUsage, AgentError> {
    let compact_start = std::time::Instant::now();
    let mut compaction_history: Vec<Message> = history.as_slice().to_vec();
    strip_images(&mut compaction_history);
    strip_thinking(&mut compaction_history);

    let system_tokens =
        estimate_message_tokens(&Message::user(crate::prompt::COMPACTION_SYSTEM.to_string()));
    let user_prompt_tokens =
        estimate_message_tokens(&Message::user(crate::prompt::COMPACTION_USER.to_string()));

    // The API request consists of: system_prompt + messages + compaction_user_prompt.
    // The server tokenizes all of this as input.  We need:
    //   real_tokens(system + messages + user_prompt) + output_reserve <= context_window
    //
    // Because our estimator undercounts for non-cl100k tokenizers, we apply a
    // safety factor to the budget so the real token count stays under limit.
    //
    //   estimated_total * (1 / SAFETY) <= context_window - output_reserve
    // → estimated_total <= (context_window - output_reserve) * SAFETY
    //
    // estimated_total = system_tokens + history_tokens + user_prompt_tokens
    // → history_tokens <= (context_window - output_reserve) * SAFETY - sys - user
    let single_pass_budget = compaction_history_budget(
        model.context_window,
        compaction_buffer,
        system_tokens,
        user_prompt_tokens,
        0, // no summary overhead for single-pass
    );

    let history_tokens = estimate_messages_tokens(&compaction_history);

    if history_tokens <= single_pass_budget {
        // Fast path: history fits in one pass.
        compaction_history.push(Message::user(crate::prompt::COMPACTION_USER.to_string()));

        let empty_tools = serde_json::json!([]);
        let response = stream_with_retry(
            provider,
            model,
            &compaction_history,
            crate::prompt::COMPACTION_SYSTEM,
            &empty_tools,
            event_tx,
            cancel,
            RequestOptions::default(),
            None,
        )
        .await?;

        event_tx.send(AgentEvent::TurnComplete(Box::new(TurnCompleteEvent {
            message: response.message.clone(),
            usage: response.usage,
            model: model.id.clone(),
            context_size: Some(response.usage.output),
        })))?;

        let new_history = vec![
            Message::user("What did we do so far?".into()),
            response.message,
        ];
        history.replace(new_history);
        info!(
            model = %model.id,
            passes = 1,
            duration_ms = compact_start.elapsed().as_millis() as u64,
            "compaction completed"
        );

        return Ok(response.usage);
    }

    // Slow path: history is too large for one pass — use multi-pass compaction.
    // Each pass sends: [previous_summary] + chunk + [compaction_user_prompt].
    // The chunk budget must account for the accumulated summary overhead.
    let chunk_budget = compaction_history_budget(
        model.context_window,
        compaction_buffer,
        system_tokens,
        user_prompt_tokens,
        SUMMARY_OVERHEAD_ESTIMATE,
    );

    let chunks = split_into_chunks(&compaction_history, chunk_budget);
    let pass_count = chunks.len();
    info!(
        model = %model.id,
        passes = pass_count,
        "compaction requires multi-pass (history too large for single request)"
    );

    let mut total_usage = TokenUsage::default();
    let mut previous_summary: Option<String> = None;

    for (i, chunk) in chunks.into_iter().enumerate() {
        if cancel.is_cancelled() {
            return Err(AgentError::Cancelled);
        }

        let mut pass_messages: Vec<Message> = Vec::new();

        // Feed the accumulated summary from previous passes so context is preserved.
        if let Some(ref summary) = previous_summary {
            pass_messages.push(Message::user(format!(
                "[Earlier conversation summary, to be merged with the conversation below:]\n{summary}"
            )));
        }

        pass_messages.extend(chunk);
        pass_messages.push(Message::user(crate::prompt::COMPACTION_USER.to_string()));

        let empty_tools = serde_json::json!([]);
        let response = stream_with_retry(
            provider,
            model,
            &pass_messages,
            crate::prompt::COMPACTION_SYSTEM,
            &empty_tools,
            event_tx,
            cancel,
            RequestOptions::default(),
            None,
        )
        .await?;

        total_usage.input += response.usage.input;
        total_usage.output += response.usage.output;
        total_usage.cache_read += response.usage.cache_read;
        total_usage.cache_creation += response.usage.cache_creation;

        let summary: String = response
            .message
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        if i < pass_count - 1 {
            // Not the final pass — keep the summary for the next iteration.
            previous_summary = Some(summary);
        } else {
            // Final pass — replace history with the accumulated summary.
            event_tx.send(AgentEvent::TurnComplete(Box::new(TurnCompleteEvent {
                message: response.message.clone(),
                usage: total_usage,
                model: model.id.clone(),
                context_size: Some(total_usage.output),
            })))?;

            let new_history = vec![
                Message::user("What did we do so far?".into()),
                response.message,
            ];
            history.replace(new_history);
        }
    }

    info!(
        model = %model.id,
        passes = pass_count,
        duration_ms = compact_start.elapsed().as_millis() as u64,
        "compaction completed"
    );

    Ok(total_usage)
}

/// Compute the maximum estimated token budget for the history portion of a
/// compaction request, given the model's context window and the overhead of
/// the surrounding prompts.
///
/// `output_reserve` is the token budget reserved for the model's summary
/// response (taken from `AgentConfig.compaction_buffer`).
/// `summary_overhead` is the estimated token cost of any accumulated summary
/// message prepended to the request (0 for single-pass, >0 for multi-pass).
fn compaction_history_budget(
    context_window: u32,
    output_reserve: u32,
    system_tokens: u32,
    user_prompt_tokens: u32,
    summary_overhead: u32,
) -> u32 {
    let available = (context_window as f64 - output_reserve as f64).max(0.0);
    // Apply safety factor to the available budget, then subtract fixed overhead.
    let scaled_budget = available * ESTIMATOR_SAFETY_FACTOR;
    let overhead = (system_tokens + user_prompt_tokens + summary_overhead) as f64;
    (scaled_budget - overhead).max(0.0) as u32
}

pub async fn compact(
    provider: &dyn maki_providers::provider::Provider,
    model: &Model,
    history: &mut History,
    event_tx: &EventSender,
    compaction_buffer: u32,
) -> Result<(), AgentError> {
    let cancel = CancelToken::none();
    let usage = compact_history(
        provider,
        model,
        history,
        event_tx,
        &cancel,
        compaction_buffer,
    )
    .await?;

    event_tx.send(AgentEvent::Done {
        usage,
        num_turns: 1,
        stop_reason: None,
    })?;

    Ok(())
}

/// Check whether the accumulated token usage has crossed the safety-adjusted
/// overflow threshold.
///
/// Applies `ESTIMATOR_SAFETY_FACTOR` so we trigger compaction earlier than the
/// raw context-window limit — the `chars/4` estimator undercounts for non-
/// cl100k tokenizers, so the real token count may be ~25 % higher than our
/// estimate.  By firing at ≈80 % of the usable window we leave headroom for
/// that mismatch.
pub(super) fn is_overflow(usage: &TokenUsage, model: &Model, compaction_buffer: u32) -> bool {
    let reserved = compaction_buffer.min(model.max_output_tokens) as f64;
    let usable = (model.context_window as f64 - reserved).max(0.0) * ESTIMATOR_SAFETY_FACTOR;
    usage.context_tokens() as f64 >= usable
}

fn strip_images(messages: &mut [Message]) {
    for msg in messages {
        for block in &mut msg.content {
            if matches!(block, ContentBlock::Image { .. }) {
                *block = ContentBlock::Text {
                    text: IMAGE_PLACEHOLDER.into(),
                };
            }
        }
    }
}

fn strip_thinking(messages: &mut [Message]) {
    for msg in messages {
        msg.content.retain(|block| {
            !matches!(
                block,
                ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. }
            )
        });
    }
}

pub(super) fn auto_compact_enabled() -> bool {
    env::var("MAKI_DISABLE_AUTOCOMPACT")
        .map(|v| v != "1" && v != "true")
        .unwrap_or(true)
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use maki_providers::provider::{BoxFuture, Provider};
    use maki_providers::{
        ContentBlock, Message, Model, ProviderEvent, RequestOptions, Role, StopReason,
        StreamResponse, TokenUsage,
    };
    use serde_json::Value;
    use test_case::test_case;

    use super::*;
    use crate::AgentConfig;

    struct MockProvider {
        responses: Mutex<Vec<StreamResponse>>,
    }

    impl MockProvider {
        fn new(responses: Vec<StreamResponse>) -> Self {
            Self {
                responses: Mutex::new(responses),
            }
        }
    }

    impl Provider for MockProvider {
        fn stream_message<'a>(
            &'a self,
            _: &'a Model,
            _: &'a [Message],
            _: &'a str,
            _: &'a Value,
            _: &'a flume::Sender<ProviderEvent>,
            _: RequestOptions,
            _: Option<&str>,
        ) -> BoxFuture<'a, Result<StreamResponse, AgentError>> {
            Box::pin(async {
                let mut responses = self.responses.lock().unwrap();
                assert!(!responses.is_empty(), "MockProvider: no more responses");
                Ok(responses.remove(0))
            })
        }

        fn list_models(&self) -> BoxFuture<'_, Result<Vec<String>, AgentError>> {
            Box::pin(async { unimplemented!() })
        }
    }

    fn default_model() -> Model {
        Model::from_spec("anthropic/claude-sonnet-4-20250514").unwrap()
    }

    fn small_context_model(context_window: u32, max_output_tokens: u32) -> Model {
        let mut model = default_model();
        model.context_window = context_window;
        model.max_output_tokens = max_output_tokens;
        model
    }

    fn text_response(stop_reason: StopReason) -> StreamResponse {
        StreamResponse {
            message: Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text {
                    text: "response".into(),
                }],
                ..Default::default()
            },
            usage: TokenUsage::default(),
            stop_reason: Some(stop_reason),
        }
    }

    fn text_response_with_usage(
        stop_reason: StopReason,
        input: u32,
        output: u32,
    ) -> StreamResponse {
        StreamResponse {
            message: Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text {
                    text: "summary".into(),
                }],
                ..Default::default()
            },
            usage: TokenUsage {
                input,
                output,
                cache_read: 0,
                cache_creation: 0,
            },
            stop_reason: Some(stop_reason),
        }
    }

    #[test]
    fn compact_replaces_history_with_summary() {
        smol::block_on(async {
            let provider: std::sync::Arc<dyn Provider> =
                std::sync::Arc::new(MockProvider::new(vec![text_response(StopReason::EndTurn)]));
            let model = default_model();
            let (raw_tx, _rx) = flume::unbounded();
            let mut history = History::new(vec![
                Message::user("first".into()),
                Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::Text {
                        text: "reply".into(),
                    }],
                    ..Default::default()
                },
            ]);

            compact(
                &*provider,
                &model,
                &mut history,
                &EventSender::new(raw_tx, 0),
                AgentConfig::default().compaction_buffer,
            )
            .await
            .unwrap();

            let msgs = history.as_slice();
            assert_eq!(msgs.len(), 2);
            assert!(matches!(msgs[0].role, Role::User));
            assert!(matches!(msgs[1].role, Role::Assistant));
        });
    }

    #[test_case(143_999, 0,       0,       0,      200_000, 20_000, false ; "below_threshold")]
    #[test_case(144_000, 0,       0,       0,      200_000, 20_000, true  ; "at_threshold")]
    #[test_case(190_000, 0,       0,       0,      200_000, 10_000, true  ; "small_max_output_uses_it_as_reserve")]
    #[test_case(100,     0,       0,       0,      100,     20_000, true  ; "tiny_context_window")]
    #[test_case(5_000,   165_000, 10_000,  0,      200_000, 20_000, true  ; "cached_tokens_count_toward_overflow")]
    #[test_case(100_000, 0,       0,       80_000, 200_000, 20_000, true  ; "output_tokens_count_toward_overflow")]
    fn overflow_detection(
        input: u32,
        cache_read: u32,
        cache_creation: u32,
        output: u32,
        ctx_window: u32,
        max_out: u32,
        expected: bool,
    ) {
        let model = small_context_model(ctx_window, max_out);
        let usage = TokenUsage {
            input,
            output,
            cache_read,
            cache_creation,
        };
        assert_eq!(
            is_overflow(&usage, &model, AgentConfig::default().compaction_buffer),
            expected
        );
    }

    #[test]
    fn strip_images_replaces_with_placeholder() {
        use maki_providers::{ImageMediaType, ImageSource};
        use std::sync::Arc;
        let source = ImageSource::new(ImageMediaType::Png, Arc::from("abc"));
        let mut messages = vec![Message::user_with_images("hello".into(), vec![source])];
        strip_images(&mut messages);
        assert_eq!(messages[0].content.len(), 2);
        assert!(
            matches!(&messages[0].content[0], ContentBlock::Text { text } if text == IMAGE_PLACEHOLDER)
        );
        assert!(matches!(&messages[0].content[1], ContentBlock::Text { text } if text == "hello"));
    }

    #[test]
    fn strip_thinking_removes_thinking_blocks() {
        let mut messages = vec![Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Thinking {
                    thinking: "hmm".into(),
                    signature: Some("sig".into()),
                },
                ContentBlock::Text {
                    text: "hello".into(),
                },
                ContentBlock::RedactedThinking {
                    data: "opaque".into(),
                },
            ],
            ..Default::default()
        }];
        strip_thinking(&mut messages);
        assert_eq!(messages[0].content.len(), 1);
        assert!(matches!(&messages[0].content[0], ContentBlock::Text { text } if text == "hello"));
    }

    #[test]
    fn estimate_message_tokens_text_is_chars_div_4() {
        // 40 chars → 10 tokens
        let msg = Message::user("A".repeat(40));
        assert_eq!(estimate_message_tokens(&msg), 10);
    }

    #[test]
    fn estimate_message_tokens_empty_text_is_one_token() {
        let msg = Message::user(String::new());
        assert_eq!(estimate_message_tokens(&msg), 1);
    }

    #[test]
    fn estimate_message_tokens_image_is_768() {
        use maki_providers::{ImageMediaType, ImageSource};
        use std::sync::Arc;
        let source = ImageSource::new(ImageMediaType::Png, Arc::from("abc"));
        let msg = Message::user_with_images(String::new(), vec![source]);
        assert_eq!(estimate_message_tokens(&msg), 768);
    }

    #[test]
    fn estimate_messages_tokens_sums_all_messages() {
        // "abcd" → 1 token, "efghijkl" → 2 tokens
        let msgs = vec![
            Message::user("abcd".into()),
            Message::user("efghijkl".into()),
        ];
        assert_eq!(estimate_messages_tokens(&msgs), 3);
    }

    // -- split_into_chunks tests --

    #[test]
    fn split_into_chunks_empty_input() {
        let chunks = split_into_chunks(&[], 1000);
        assert!(chunks.is_empty());
    }

    #[test]
    fn split_into_chunks_single_chunk_when_fits() {
        let msgs = vec![Message::user("hello".into()), Message::user("world".into())];
        let chunks = split_into_chunks(&msgs, 10_000);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 2);
    }

    #[test]
    fn split_into_chunks_splits_when_too_large() {
        // Each message is 1000 chars → 250 tokens. Budget = 500 → 2 messages per chunk.
        let big = "X".repeat(1000);
        let msgs: Vec<Message> = (0..5)
            .map(|i| Message::user(format!("msg-{i}: {big}")))
            .collect();
        let chunks = split_into_chunks(&msgs, 500);
        assert!(
            chunks.len() >= 2,
            "should split into multiple chunks, got {}",
            chunks.len()
        );
        // Every message should appear in exactly one chunk.
        let total: usize = chunks.iter().map(|c| c.len()).sum();
        assert_eq!(total, msgs.len(), "all messages accounted for");
    }

    #[test]
    fn split_into_chunks_giant_message_gets_own_chunk() {
        // A single message that exceeds the budget.
        let giant = "Z".repeat(100_000); // 25_000 tokens
        let msgs = vec![
            Message::user("small".into()),
            Message::user(giant.clone()),
            Message::user("small2".into()),
        ];
        let chunks = split_into_chunks(&msgs, 100);
        // The giant message gets its own chunk.
        assert!(
            chunks
                .iter()
                .any(|c| c.len() == 1 && estimate_message_tokens(&c[0]) > 100),
            "giant message should be in its own chunk"
        );
    }

    // -- multi-pass compaction tests --

    #[test]
    fn multi_pass_compaction_accumulates_summary() {
        smol::block_on(async {
            // We need a model where the history is too large for single-pass
            // but splits into multiple chunks.
            //
            // budget = (context_window - COMPACT_OUTPUT_RESERVE) * 0.8 - sys - user - summary_overhead
            // With summary_overhead=0 for single-pass, and 5000 for multi-pass.
            //
            // For context_window=41000:
            //   single_pass_budget = (41000 - 40000) * 0.8 - sys - user
            //                       = 800 - sys - user ≈ 450
            //   chunk_budget = 450 - 5000 (clamped to 0) → too small!
            //
            // We need a larger context window. For context_window=100000:
            //   single_pass_budget = (100000 - 40000) * 0.8 - sys - user
            //                       = 48000 - sys - user ≈ 47650
            //   chunk_budget = 47650 - 5000 = 42650
            //
            // With 4 messages of 1000 chars each (250 est tokens):
            //   history_tokens = 1000, which fits in 47650 → single pass!
            //
            // We need bigger messages. With 10000 chars (2500 est tokens):
            //   history_tokens = 10000, which fits in 47650 → still single pass!
            //
            // With 20000 chars (5000 est tokens) × 4 = 20000:
            //   history_tokens = 20000, which fits in 47650 → still single pass!
            //
            // With 20000 chars × 10 = 200000:
            //   history_tokens = 200000, which exceeds 47650 → multi-pass!
            //   chunk_budget = 42650, so each chunk holds ~17 messages... too few chunks.
            //
            // Actually let's just pick a tiny context window where multi-pass triggers:
            // For context_window=50000:
            //   single_pass_budget = (50000 - 40000) * 0.8 - 167 - 184 = 8000 - 351 = 7649
            //   chunk_budget = 7649 - 5000 = 2649
            //
            // With 10000 chars (2500 est tokens) × 8 = 20000:
            //   history_tokens = 20000 > 7649 → multi-pass!
            //   chunk_budget = 2649 → each chunk holds 1 message (2500 > 2649? no, 2500 < 2649)
            //   Actually 2500 < 2649, so 1 message per chunk. 8 chunks.
            //   But 2500 + 2500 = 5000 > 2649, so only 1 per chunk.
            //   That's 8 API calls — too many for the test.
            //
            // Let's use 4 messages of 10000 chars (2500 est tokens each):
            //   history_tokens = 10000 > 7649 → multi-pass!
            //   chunk_budget = 2649 → 1 message per chunk (2500 < 2649, but 5000 > 2649)
            //   4 chunks, 4 API calls.
            let model = small_context_model(50_000, 10_000);

            let big = "X".repeat(10000);
            let messages: Vec<Message> = (0..4)
                .map(|i| Message::user(format!("msg-{i}: {big}")))
                .collect();

            let history_tokens = estimate_messages_tokens(&messages);
            let system_tokens = estimate_message_tokens(&Message::user(
                crate::prompt::COMPACTION_SYSTEM.to_string(),
            ));
            let user_prompt_tokens =
                estimate_message_tokens(&Message::user(crate::prompt::COMPACTION_USER.to_string()));
            let compact_output_reserve = AgentConfig::default().compaction_buffer;
            let single_pass_budget = compaction_history_budget(
                model.context_window,
                compact_output_reserve,
                system_tokens,
                user_prompt_tokens,
                0,
            );
            let chunk_budget = compaction_history_budget(
                model.context_window,
                compact_output_reserve,
                system_tokens,
                user_prompt_tokens,
                SUMMARY_OVERHEAD_ESTIMATE,
            );

            // Skip if the budget happens to fit everything in one chunk.
            if history_tokens <= single_pass_budget {
                return;
            }

            let chunks = split_into_chunks(&messages, chunk_budget);
            assert!(
                chunks.len() > 1,
                "should need multiple chunks, got {}",
                chunks.len()
            );

            // Provide one response per chunk.
            let responses: Vec<StreamResponse> = chunks
                .iter()
                .map(|_| text_response_with_usage(StopReason::EndTurn, 100, 50))
                .collect();

            let provider: std::sync::Arc<dyn Provider> =
                std::sync::Arc::new(MockProvider::new(responses));

            let (raw_tx, _rx) = flume::unbounded();
            let mut history = History::new(messages);

            let usage = compact_history(
                &*provider,
                &model,
                &mut history,
                &EventSender::new(raw_tx, 0),
                &CancelToken::none(),
                AgentConfig::default().compaction_buffer,
            )
            .await
            .unwrap();

            // Verify usage is accumulated across passes.
            assert_eq!(usage.input, (chunks.len() as u32) * 100);
            assert_eq!(usage.output, (chunks.len() as u32) * 50);

            // History should be replaced with 2 messages.
            let msgs = history.as_slice();
            assert_eq!(msgs.len(), 2);
        });
    }

    #[test]
    fn single_pass_when_history_fits() {
        smol::block_on(async {
            let provider: std::sync::Arc<dyn Provider> =
                std::sync::Arc::new(MockProvider::new(vec![text_response(StopReason::EndTurn)]));
            let model = default_model();
            let (raw_tx, _rx) = flume::unbounded();

            // Small history that definitely fits.
            let mut history = History::new(vec![
                Message::user("hello".into()),
                Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::Text { text: "hi".into() }],
                    ..Default::default()
                },
            ]);

            let usage = compact_history(
                &*provider,
                &model,
                &mut history,
                &EventSender::new(raw_tx, 0),
                &CancelToken::none(),
                AgentConfig::default().compaction_buffer,
            )
            .await
            .unwrap();

            // Should have used exactly one API call.
            assert_eq!(usage.input, 0); // default usage in mock
            let msgs = history.as_slice();
            assert_eq!(msgs.len(), 2);
        });
    }
}

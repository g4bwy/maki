use std::env;
use std::future::Future;
use std::io;
use std::time::Duration;

use async_tungstenite::WebSocketStream;
use async_tungstenite::smol::{ConnectStream, connect_async};
use async_tungstenite::tungstenite::http::HeaderName;
use async_tungstenite::tungstenite::{
    Error as WebSocketError, Message as Frame, client::IntoClientRequest,
};
use flume::Sender;
use futures::{SinkExt, StreamExt};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tracing::{debug, warn};

use super::responses::{self, ResponseAccumulator};
use super::routing::{RoutingState, TURN_STATE_HEADER};
use crate::providers::{ResolvedAuth, Timeouts};
use crate::{
    AgentError, ContentBlock, Message, Model, ProviderEvent, ProviderSession, Role, StreamResponse,
};

const TRANSPORT_ENV: &str = "MAKI_OPENAI_RESPONSES_TRANSPORT";
const WEBSOCKET: &str = "websocket";
const HTTP: &str = "http";
const WEBSOCKET_BETA: &str = "responses_websockets=2026-02-06";
const PREVIOUS_NOT_FOUND: &str = "previous_response_not_found";
const CONNECTION_LIMIT: &str = "websocket_connection_limit_reached";
const CLOSED_MESSAGE: &str = "connection closed before response completion";
const RECOVERY_ATTEMPTS: usize = 2;

type Socket = WebSocketStream<ConnectStream>;

/// How a plan turn reaches the upstream. The WebSocket transport is opt-in for
/// now: it is a beta endpoint, it brings a second TLS stack with it, and
/// `MAKI_OPENAI_RESPONSES_TRANSPORT=websocket` is what turns it on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Transport {
    Http,
    Websocket,
}

impl Transport {
    pub(crate) fn from_env() -> Self {
        Self::parse(env::var(TRANSPORT_ENV).ok().as_deref())
    }

    fn parse(value: Option<&str>) -> Self {
        match value {
            None | Some(HTTP) => Self::Http,
            Some(WEBSOCKET) => Self::Websocket,
            Some(other) => {
                warn!(
                    value = other,
                    expected = format!("{HTTP} or {WEBSOCKET}"),
                    "ignoring {TRANSPORT_ENV}"
                );
                Self::Http
            }
        }
    }
}

#[derive(Default)]
pub(crate) struct ResponsesSession {
    connection: Option<Socket>,
    connection_key: Option<[u8; 32]>,
    generation: u64,
    previous: Option<Continuation>,
    /// The transport failed once for this session; do not try it again until
    /// the session is rebuilt. Every request still goes out over HTTP.
    fallback: bool,
}

impl ResponsesSession {
    fn forget(&mut self) {
        self.connection = None;
        self.previous = None;
    }
}

struct Continuation {
    response_id: String,
    properties: Value,
    input: Vec<Value>,
}

fn properties(body: &Value) -> Value {
    body.as_object().map_or(Value::Null, |fields| {
        Value::Object(
            fields
                .iter()
                .filter(|(name, _)| name.as_str() != "input")
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect(),
        )
    })
}

fn request_payload(body: &Value, previous: Option<&Continuation>) -> (Value, &'static str) {
    let mut payload = body.clone();
    let reason = match previous {
        Some(previous) if previous.properties != properties(body) => "properties_changed",
        Some(previous) => {
            if let Some(input) = body["input"].as_array()
                && input.starts_with(&previous.input)
            {
                payload["input"] = json!(&input[previous.input.len()..]);
                payload["previous_response_id"] = json!(previous.response_id);
                "incremental"
            } else {
                "history_changed"
            }
        }
        None => "no_previous_response",
    };
    payload["type"] = json!("response.create");
    if let Some(object) = payload.as_object_mut() {
        object.remove("stream");
    }
    (payload, reason)
}

fn continuation(
    body: &Value,
    response: &StreamResponse,
    completed: &Value,
) -> Option<Continuation> {
    if completed["status"].as_str() != Some("completed") {
        return None;
    }
    let output = completed["output"].as_array()?;
    let content = responses::output_content(output)?;
    let raw = Message {
        role: Role::Assistant,
        content,
        ..Default::default()
    };
    let raw_input = responses::coding_plan_input(&[raw]);
    if raw_input != responses::coding_plan_input(std::slice::from_ref(&response.message)) {
        return None;
    }
    // Compare the representation the agent will retain, including its tool-name normalization.
    let mut message = response.message.clone();
    for block in &mut message.content {
        if let ContentBlock::ToolUse { name, .. } = block
            && let Some(canonical) = name.strip_prefix("functions.")
        {
            *name = canonical.to_owned();
        }
    }
    let mut input = body["input"].as_array()?.clone();
    input.extend(
        responses::coding_plan_input(&[message])
            .as_array()?
            .iter()
            .cloned(),
    );
    Some(Continuation {
        response_id: completed["id"].as_str()?.to_owned(),
        properties: properties(body),
        input,
    })
}

/// A dead or unusable socket, in the same shape a dropped HTTP connection
/// arrives in. `store: false` means the upstream kept nothing of the request,
/// so the retry layer may send the whole turn again, and the pooled socket is
/// already gone by the time this is returned: the next attempt reconnects.
fn transport_failure(error: impl ToString) -> AgentError {
    AgentError::Io(io::Error::new(
        io::ErrorKind::ConnectionAborted,
        error.to_string(),
    ))
}

async fn timeout<T>(duration: Duration, future: impl Future<Output = T>) -> Result<T, AgentError> {
    futures_lite::future::race(async { Ok(future.await) }, async {
        smol::Timer::after(duration).await;
        Err(AgentError::Timeout {
            secs: duration.as_secs(),
        })
    })
    .await
}

fn websocket_url(base: &str) -> Result<String, AgentError> {
    let mut url =
        url::Url::parse(&format!("{}/responses", base.trim_end_matches('/'))).map_err(|error| {
            AgentError::Config {
                message: error.to_string(),
            }
        })?;
    let scheme = match url.scheme() {
        "http" => "ws",
        "https" => "wss",
        _ => {
            return Err(AgentError::Config {
                message: "Responses endpoint must use HTTP or HTTPS".into(),
            });
        }
    };
    url.set_scheme(scheme).map_err(|()| AgentError::Config {
        message: "invalid WebSocket endpoint".into(),
    })?;
    Ok(url.into())
}

async fn connect(
    auth: &ResolvedAuth,
    timeouts: Timeouts,
    state: &mut ResponsesSession,
    routing: &RoutingState,
) -> Result<Socket, AgentError> {
    let url = websocket_url(auth.base_url.as_deref().unwrap_or_default())?;
    let mut request = url
        .into_client_request()
        .map_err(|error| AgentError::Config {
            message: error.to_string(),
        })?;
    for (name, value) in auth
        .headers
        .iter()
        .map(|(name, value)| (name.as_str(), value.as_str()))
        .chain([
            ("user-agent", super::super::user_agent()),
            ("openai-beta", WEBSOCKET_BETA),
        ])
    {
        request.headers_mut().insert(
            name.parse::<HeaderName>()
                .map_err(|error| AgentError::Config {
                    message: format!("invalid header name: {error}"),
                })?,
            value.parse().map_err(|error| AgentError::Config {
                message: format!("invalid header value: {error}"),
            })?,
        );
    }
    if let Some(turn_state) = routing.value() {
        request.headers_mut().insert(
            TURN_STATE_HEADER,
            turn_state.parse().map_err(|error| AgentError::Config {
                message: format!("invalid routing state: {error}"),
            })?,
        );
    }
    let result = timeout(timeouts.connect, connect_async(request)).await?;
    let (socket, response) = result.map_err(|error| match error {
        WebSocketError::Http(response) => {
            routing.retain(
                response
                    .headers()
                    .get(TURN_STATE_HEADER)
                    .and_then(|value| value.to_str().ok()),
            );
            AgentError::api(
                response.status().as_u16(),
                response.body().as_ref().map_or_else(
                    || "WebSocket upgrade rejected".into(),
                    |body| String::from_utf8_lossy(body).into_owned(),
                ),
            )
        }
        other => AgentError::Config {
            message: format!("WebSocket connection failed: {other}"),
        },
    })?;
    routing.retain(
        response
            .headers()
            .get(TURN_STATE_HEADER)
            .and_then(|value| value.to_str().ok()),
    );
    state.generation += 1;
    Ok(socket)
}

/// `None` asks the caller to send the full request through its existing HTTP path.
pub(super) async fn stream(
    session: &ProviderSession,
    model: &Model,
    body: &Value,
    event_tx: &Sender<ProviderEvent>,
    auth: &ResolvedAuth,
    timeouts: Timeouts,
    transport: Transport,
) -> Result<Option<StreamResponse>, AgentError> {
    let mut state = session.responses().lock().await;
    let routing = session.routing();
    routing.prepare(auth)?;
    let key: [u8; 32] = Sha256::digest(serde_json::to_vec(&(
        auth.base_url.as_ref(),
        &auth.headers,
    ))?)
    .into();
    if state.connection_key != Some(key) {
        *state = ResponsesSession {
            connection_key: Some(key),
            generation: state.generation,
            ..Default::default()
        };
    }
    if transport == Transport::Http {
        state.forget();
        return Ok(None);
    }
    if state.fallback {
        return Ok(None);
    }
    for recovery in 0..RECOVERY_ATTEMPTS {
        let reused = state.connection.is_some();
        // The future owns the socket until a terminal response; cancellation cannot return it to the pool.
        let mut socket = if let Some(socket) = state.connection.take() {
            socket
        } else {
            state.previous = None;
            match connect(auth, timeouts, &mut state, routing).await {
                Ok(socket) => socket,
                Err(
                    error @ AgentError::Api {
                        status: 401 | 403 | 429 | 500..=599,
                        ..
                    },
                ) => return Err(error),
                Err(error) => {
                    warn!(error = %error, "Responses WebSocket unavailable, using HTTP");
                    state.forget();
                    state.fallback = true;
                    return Ok(None);
                }
            }
        };
        let (mut payload, reason) = request_payload(body, state.previous.take().as_ref());
        if let Some(turn_state) = routing.value() {
            payload["client_metadata"][TURN_STATE_HEADER] = json!(turn_state);
        }
        let bytes = serde_json::to_string(&payload)?;
        debug!(model = %model.id, reused, reason, generation = state.generation, "sending Responses WebSocket request");
        timeout(timeouts.stream, socket.send(Frame::Text(bytes.into())))
            .await
            .map_err(transport_failure)?
            .map_err(transport_failure)?;
        let mut accumulator = ResponseAccumulator::coding_plan();
        let mut observed_content = false;
        loop {
            let frame = timeout(timeouts.stream, socket.next())
                .await
                .map_err(transport_failure)?
                .ok_or_else(|| transport_failure(CLOSED_MESSAGE))?
                .map_err(transport_failure)?;
            let data = match frame {
                Frame::Text(data) => data,
                Frame::Ping(_) | Frame::Pong(_) => {
                    timeout(timeouts.stream, socket.flush())
                        .await
                        .map_err(transport_failure)?
                        .map_err(transport_failure)?;
                    continue;
                }
                Frame::Close(_) => return Err(transport_failure(CLOSED_MESSAGE)),
                _ => return Err(transport_failure("unexpected binary Responses frame")),
            };
            let parsed: Value = serde_json::from_str(&data).map_err(transport_failure)?;
            let event = parsed["type"]
                .as_str()
                .ok_or_else(|| transport_failure("Responses event has no type"))?;
            routing.observe(event, &parsed);
            let code = parsed["error"]["code"].as_str();
            if event == "error"
                && !observed_content
                && recovery + 1 < RECOVERY_ATTEMPTS
                && matches!(code, Some(PREVIOUS_NOT_FOUND | CONNECTION_LIMIT))
            {
                if code == Some(PREVIOUS_NOT_FOUND) {
                    state.connection = Some(socket);
                }
                break;
            }
            if event == "error"
                && !observed_content
                && matches!(code, Some(PREVIOUS_NOT_FOUND | CONNECTION_LIMIT))
            {
                return Err(AgentError::api(
                    400,
                    "Responses continuation recovery exhausted",
                ));
            }
            if event == "error" && !observed_content {
                let status = parsed["status"]
                    .as_u64()
                    .and_then(|status| u16::try_from(status).ok());
                if let Some(status) = status {
                    return Err(AgentError::api(
                        status,
                        parsed["error"]["message"]
                            .as_str()
                            .unwrap_or("Responses request rejected"),
                    ));
                }
            }
            observed_content |= matches!(
                event,
                "response.output_text.delta"
                    | "response.reasoning_text.delta"
                    | "response.reasoning_summary_text.delta"
                    | "response.output_item.added"
                    | "response.output_item.done"
                    | "response.function_call_arguments.delta"
            );
            let terminal = accumulator.push(event, &parsed, event_tx).await?;
            if terminal {
                let completed = accumulator.replay_output().cloned();
                let response = accumulator.finish().map_err(transport_failure)?;
                if let Some(completed) = completed {
                    state.previous = continuation(body, &response, &completed);
                }
                state.connection = Some(socket);
                return Ok(Some(response));
            }
        }
    }
    Err(AgentError::api(
        400,
        "Responses continuation recovery exhausted",
    ))
}

#[cfg(test)]
mod tests {
    use crate::providers::openai::responses::{Plan, coding_plan_input, do_stream};
    use crate::providers::openai::routing::{ROUTING_HINT_HEADER, TURN_STATE_HEADER, routing_hint};
    use async_tungstenite::tungstenite::handshake::server::{
        Request as HandshakeRequest, Response as HandshakeResponse,
    };
    use async_tungstenite::{accept_async, accept_hdr_async};
    use futures::StreamExt;
    use futures_lite::io::{AsyncReadExt, AsyncWriteExt};
    use isahc::HttpClient;
    use maki_storage::id::SessionRef;
    use serde_json::{Value, json};
    use smol::net::{TcpListener, TcpStream};
    use test_case::test_case;

    use super::{Continuation, Transport, continuation, properties, request_payload, stream};
    use crate::providers::ResolvedAuth;
    use crate::providers::openai::responses::{ResponseAccumulator, build_body};
    use crate::{AgentError, Message, Model, ProviderSession, Timeouts};

    #[test_case(None, Transport::Http ; "unset")]
    #[test_case(Some("http"), Transport::Http ; "http")]
    #[test_case(Some("websocket"), Transport::Websocket ; "websocket")]
    #[test_case(Some("auto"), Transport::Http ; "unknown_value_stays_on_http")]
    fn transport_is_opt_in(value: Option<&str>, expected: Transport) {
        assert_eq!(Transport::parse(value), expected);
    }

    const RESPONSE_ID: &str = "resp_test";
    const ANSWER: &str = "answer";
    const TOOL_ID: &str = "call_test";
    const TOOL_NAME: &str = "read";
    const ERROR_MESSAGE: &str = "response is no longer available";

    fn request() -> Value {
        build_body(
            &Model::from_spec("openai/gpt-5.6-luna").unwrap(),
            &[Message::user("hello".into())],
            "instructions",
            &json!([]),
        )
    }

    fn completed() -> Value {
        json!({"type": "response.completed", "response": {"id": RESPONSE_ID, "status": "completed",
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": ANSWER}]}],
            "usage": {"input_tokens": 100, "output_tokens": 10, "input_tokens_details": {"cached_tokens": 80}}}})
    }

    async fn response(socket: &mut async_tungstenite::WebSocketStream<TcpStream>) {
        socket
            .send(
                json!({"type": "response.output_text.delta", "delta": ANSWER})
                    .to_string()
                    .into(),
            )
            .await
            .unwrap();
        let mut terminal = completed();
        socket.send(json!({"type":"response.output_item.done", "output_index":0, "item":terminal["response"]["output"][0]}).to_string().into()).await.unwrap();
        terminal["response"]["output"] = json!([]);
        socket.send(terminal.to_string().into()).await.unwrap();
    }

    async fn received(socket: &mut async_tungstenite::WebSocketStream<TcpStream>) -> Value {
        let message = socket.next().await.unwrap().unwrap();
        serde_json::from_str(message.to_text().unwrap()).unwrap()
    }

    async fn received_http(tcp: &mut TcpStream) -> String {
        let mut headers = Vec::new();
        let mut byte = [0];
        while !headers.ends_with(b"\r\n\r\n") {
            tcp.read_exact(&mut byte).await.unwrap();
            headers.push(byte[0]);
        }
        let headers = String::from_utf8(headers).unwrap().to_ascii_lowercase();
        let length: usize = headers
            .lines()
            .find_map(|line| line.strip_prefix("content-length: "))
            .map_or(0, |value| value.parse().unwrap());
        tcp.read_exact(&mut vec![0; length]).await.unwrap();
        headers
    }

    async fn server() -> (TcpListener, ResolvedAuth) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let auth = ResolvedAuth::for_test(
            Some(format!("http://{}", listener.local_addr().unwrap())),
            vec![],
        );
        (listener, auth)
    }

    fn session() -> ProviderSession {
        ProviderSession::new(SessionRef::generate())
    }

    async fn call(
        session: &ProviderSession,
        body: &Value,
        auth: &ResolvedAuth,
    ) -> Result<Option<crate::StreamResponse>, AgentError> {
        let model = Model::from_spec("openai/gpt-5.6-luna").unwrap();
        let (tx, _rx) = flume::unbounded();
        stream(
            session,
            &model,
            body,
            &tx,
            auth,
            Timeouts::default(),
            Transport::Websocket,
        )
        .await
    }

    fn followup(body: &mut Value) {
        body["input"].as_array_mut().unwrap().extend([
            json!({"type":"message", "role":"assistant", "content":[{"type":"output_text", "text":ANSWER}]}),
            json!({"type":"message", "role":"user", "content":[{"type":"input_text", "text":"next"}]}),
        ]);
    }

    #[test_case("instructions" ; "instructions_changed")]
    #[test_case("tools" ; "tools_changed")]
    #[test_case("reasoning" ; "reasoning_changed")]
    #[test_case("model" ; "model_changed")]
    #[test_case("service_tier" ; "service_tier_changed")]
    #[test_case("prompt_cache_key" ; "routing_changed")]
    fn request_properties_reset_continuation(field: &str) {
        let original = request();
        let previous = Continuation {
            response_id: RESPONSE_ID.into(),
            properties: properties(&original),
            input: original["input"].as_array().unwrap().clone(),
        };
        let mut changed = original;
        changed[field] = json!("changed");
        let (payload, reason) = request_payload(&changed, Some(&previous));
        assert!(payload.get("previous_response_id").is_none());
        assert_eq!(reason, "properties_changed");
        assert_eq!(payload["input"], changed["input"]);
    }

    #[test_case(true ; "compacted")]
    #[test_case(false ; "edited")]
    fn history_change_resets_continuation(compacted: bool) {
        let mut body = request();
        let previous = Continuation {
            response_id: RESPONSE_ID.into(),
            properties: properties(&body),
            input: body["input"].as_array().unwrap().clone(),
        };
        body["input"] = if compacted {
            json!([])
        } else {
            json!([{"type":"message", "content":"edited"}])
        };
        let (payload, reason) = request_payload(&body, Some(&previous));
        assert!(payload.get("previous_response_id").is_none());
        assert_eq!(reason, "history_changed");
    }

    #[test_case(false ; "new_user_turn")]
    #[test_case(true ; "same_process_prefix_change")]
    fn reuse_socket_across_turns(prefix_change: bool) {
        smol::block_on(async {
            let (listener, auth) = server().await;
            let server = smol::spawn(async move {
                let (tcp, _) = listener.accept().await.unwrap();
                let mut socket = accept_async(tcp).await.unwrap();
                let first = received(&mut socket).await;
                assert!(first.get("stream").is_none());
                assert!(first.get("previous_response_id").is_none());
                response(&mut socket).await;
                let second = received(&mut socket).await;
                if prefix_change {
                    assert!(second.get("previous_response_id").is_none());
                    assert_eq!(second["input"].as_array().unwrap().len(), 3);
                } else {
                    assert_eq!(second["previous_response_id"], RESPONSE_ID);
                    assert_eq!(second["input"].as_array().unwrap().len(), 1);
                    assert_eq!(first["instructions"], second["instructions"]);
                }
                response(&mut socket).await;
            });
            let session = session();
            let mut body = request();
            let result = call(&session, &body, &auth).await.unwrap().unwrap();
            assert_eq!(result.usage.cache_read, 80);
            session.begin_turn().await;
            followup(&mut body);
            if prefix_change {
                body["instructions"] = json!("changed instructions");
            }
            call(&session, &body, &auth).await.unwrap().unwrap();
            server.await;
            assert_eq!(session.responses().lock().await.generation, 1);
        });
    }

    #[test_case("previous_response_not_found" ; "missing_response")]
    #[test_case("websocket_connection_limit_reached" ; "connection_limit")]
    fn recover_once_with_full_history(code: &'static str) {
        smol::block_on(async {
            let (listener, auth) = server().await;
            let server = smol::spawn(async move {
                let (tcp, _) = listener.accept().await.unwrap();
                let mut socket = accept_async(tcp).await.unwrap();
                received(&mut socket).await;
                response(&mut socket).await;
                let incremental = received(&mut socket).await;
                assert_eq!(incremental["previous_response_id"], RESPONSE_ID);
                socket.send(json!({"type":"error", "status":400,"error":{"code":code,"message":ERROR_MESSAGE}}).to_string().into()).await.unwrap();
                if code == "websocket_connection_limit_reached" {
                    let (tcp, _) = listener.accept().await.unwrap();
                    socket = accept_async(tcp).await.unwrap();
                }
                let replay = received(&mut socket).await;
                assert!(replay.get("previous_response_id").is_none());
                assert_eq!(replay["input"].as_array().unwrap().len(), 3);
                response(&mut socket).await;
            });
            let session = session();
            let mut body = request();
            call(&session, &body, &auth).await.unwrap();
            followup(&mut body);
            call(&session, &body, &auth).await.unwrap();
            server.await;
        });
    }

    #[test_case(false ; "close_before_content")]
    #[test_case(true ; "close_after_content")]
    fn a_dead_socket_is_dropped_and_worth_retrying(content: bool) {
        smol::block_on(async {
            let (listener, auth) = server().await;
            let server = smol::spawn(async move {
                let (tcp, _) = listener.accept().await.unwrap();
                let mut socket = accept_async(tcp).await.unwrap();
                received(&mut socket).await;
                if content {
                    socket
                        .send(
                            json!({"type":"response.output_text.delta","delta":ANSWER})
                                .to_string()
                                .into(),
                        )
                        .await
                        .unwrap();
                }
                socket.close(None).await.unwrap();
            });
            let session = session();
            let error = call(&session, &request(), &auth).await.unwrap_err();
            assert!(matches!(error, AgentError::Io(_)));
            // The same answer a dropped HTTP connection gives: the pooled
            // socket is gone, and `store: false` means nothing was kept
            // upstream, so the turn can simply be sent again.
            assert!(error.is_retryable());
            let state = session.responses().lock().await;
            assert!(state.connection.is_none());
            assert!(state.previous.is_none());
            server.await;
        });
    }

    #[test_case("message" ; "text")]
    #[test_case("reasoning" ; "reasoning")]
    #[test_case("function_call" ; "tool_round")]
    #[test_case("unknown" ; "unknown_output")]
    fn continuation_uses_retained_assistant_representation(kind: &str) {
        smol::block_on(async {
            let (tx, _rx) = flume::unbounded();
            let mut accumulator = ResponseAccumulator::coding_plan();
            accumulator
                .push(
                    "response.output_text.delta",
                    &json!({"delta":"  answer"}),
                    &tx,
                )
                .await
                .unwrap();
            accumulator.push("response.output_item.done", &json!({"item":{"type":"function_call","call_id":TOOL_ID,"name":"functions.read","arguments":"{\"path\":\"file\"}"}}), &tx).await.unwrap();
            let mut event = completed();
            event["response"]["output"].as_array_mut().unwrap().push(json!({"type":"function_call","call_id":TOOL_ID,"name":"functions.read","arguments":"{\"path\":\"file\"}"}));
            if kind == "reasoning" {
                event["response"]["output"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!({"type":"reasoning", "encrypted_content":"opaque"}));
            }
            if kind == "unknown" {
                event["response"]["output"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!({"type":"unknown"}));
            }
            accumulator
                .push("response.completed", &event, &tx)
                .await
                .unwrap();
            let response = accumulator.finish().unwrap();
            let body = request();
            let previous = continuation(&body, &response, &event["response"]);
            if kind == "unknown" {
                assert!(previous.is_none());
                return;
            }
            let previous = previous.unwrap();
            assert_eq!(previous.input[1]["content"][0]["text"], ANSWER);
            assert_eq!(previous.input[2]["name"], TOOL_NAME);
            let mut next = body;
            next["input"] = json!(previous.input);
            next["input"]
                .as_array_mut()
                .unwrap()
                .push(json!({"type":"function_call_output","call_id":TOOL_ID,"output":"contents"}));
            let (payload, _) = request_payload(&next, Some(&previous));
            assert_eq!(payload["input"].as_array().unwrap().len(), 1);
            assert_eq!(payload["input"][0]["call_id"], TOOL_ID);
        });
    }

    #[test_case(false ; "child_isolation")]
    #[test_case(true ; "auth_revision")]
    #[allow(clippy::result_large_err)]
    fn independent_connections(auth_change: bool) {
        smol::block_on(async {
            let (listener, mut auth) = server().await;
            let server = smol::spawn(async move {
                for _ in 0..2 {
                    let (tcp, _) = listener.accept().await.unwrap();
                    let mut socket = accept_hdr_async(
                        tcp,
                        |request: &HandshakeRequest, response: HandshakeResponse| {
                            assert!(request.headers().get(TURN_STATE_HEADER).is_none());
                            Ok(response)
                        },
                    )
                    .await
                    .unwrap();
                    let request = received(&mut socket).await;
                    assert!(request.get("previous_response_id").is_none());
                    socket.send(json!({"type":"response.metadata","headers":{TURN_STATE_HEADER:TURN_STATE}}).to_string().into()).await.unwrap();
                    response(&mut socket).await;
                }
            });
            let parent = session();
            let child = parent.child();
            assert_eq!(parent.cache_key(), child.cache_key());
            assert_ne!(parent.thread_id(), child.thread_id());
            call(&parent, &request(), &auth).await.unwrap();
            assert_eq!(parent.routing().value().as_deref(), Some(TURN_STATE));
            if auth_change {
                auth.set_header("authorization", "Bearer changed".into());
            }
            call(
                if auth_change { &parent } else { &child },
                &request(),
                &auth,
            )
            .await
            .unwrap();
            server.await;
        });
    }
    #[test_case(false ; "missing_id")]
    #[test_case(true ; "incomplete")]
    fn incomplete_metadata_disables_continuation(incomplete: bool) {
        smol::block_on(async {
            let (listener, auth) = server().await;
            let server = smol::spawn(async move {
                let (tcp, _) = listener.accept().await.unwrap();
                let mut socket = accept_async(tcp).await.unwrap();
                received(&mut socket).await;
                let mut event = completed();
                if incomplete {
                    event["type"] = json!("response.incomplete");
                    event["response"]["status"] = json!("incomplete");
                } else {
                    event["response"].as_object_mut().unwrap().remove("id");
                }
                socket.send(event.to_string().into()).await.unwrap();
            });
            let session = session();
            call(&session, &request(), &auth).await.unwrap();
            assert!(session.responses().lock().await.previous.is_none());
            server.await;
        });
    }

    #[test_case(426, true ; "unsupported_upgrade")]
    #[test_case(401, false ; "auth_rejection")]
    #[test_case(429, false ; "rate_limit")]
    fn upgrade_failure_policy(status: u16, fallback: bool) {
        smol::block_on(async {
            let (listener, auth) = server().await;
            let server = smol::spawn(async move {
                let (mut tcp, _) = listener.accept().await.unwrap();
                received_http(&mut tcp).await;
                tcp.write_all(format!("HTTP/1.1 {status} Rejected\r\nContent-Length: 0\r\nConnection: close\r\n{TURN_STATE_HEADER}: {TURN_STATE}\r\n\r\n").as_bytes()).await.unwrap();
                if fallback {
                    let (mut tcp, _) = listener.accept().await.unwrap();
                    let request = received_http(&mut tcp).await;
                    assert!(request.contains(&format!("{TURN_STATE_HEADER}: {TURN_STATE}")));
                    let body = format!("data: {}\n\n", completed());
                    tcp.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
                }
            });
            let session = session();
            let result = call(&session, &request(), &auth).await;
            if fallback {
                assert!(result.unwrap().is_none());
                assert!(session.responses().lock().await.fallback);
                let (tx, _rx) = flume::unbounded();
                do_stream(
                    &HttpClient::new().unwrap(),
                    &Model::from_spec("openai/gpt-5.6-luna").unwrap(),
                    &request(),
                    &tx,
                    &auth,
                    Timeouts::default().stream,
                    Plan::Coding(Some(session.routing())),
                )
                .await
                .unwrap();
                assert_eq!(session.routing().value().as_deref(), Some(TURN_STATE));
            } else {
                assert!(
                    matches!(result, Err(AgentError::Api { status: actual, .. }) if actual == status)
                );
            }
            server.await;
        });
    }

    #[test_case("{invalid" ; "malformed_json")]
    #[test_case("{}" ; "missing_event_type")]
    fn malformed_stream_is_not_replayed(data: &'static str) {
        smol::block_on(async {
            let (listener, auth) = server().await;
            let server = smol::spawn(async move {
                let (tcp, _) = listener.accept().await.unwrap();
                let mut socket = accept_async(tcp).await.unwrap();
                received(&mut socket).await;
                socket.send(data.to_owned().into()).await.unwrap();
            });
            let error = call(&session(), &request(), &auth).await.unwrap_err();
            assert!(matches!(error, AgentError::Io(_)));
            server.await;
        });
    }

    #[test_case(())]
    fn cancelled_future_discards_connection(_: ()) {
        smol::block_on(async {
            let (listener, auth) = server().await;
            let (sent_tx, sent_rx) = flume::bounded(1);
            let server = smol::spawn(async move {
                let (tcp, _) = listener.accept().await.unwrap();
                let mut socket = accept_async(tcp).await.unwrap();
                received(&mut socket).await;
                sent_tx.send_async(()).await.unwrap();
                assert!(socket.next().await.is_none_or(|result| result.is_err()));
                let (tcp, _) = listener.accept().await.unwrap();
                let mut socket = accept_async(tcp).await.unwrap();
                let request = received(&mut socket).await;
                assert!(request.get("previous_response_id").is_none());
                response(&mut socket).await;
            });
            let session = session();
            let request_session = session.clone();
            let request_auth = auth.clone();
            let task =
                smol::spawn(async move { call(&request_session, &request(), &request_auth).await });
            sent_rx.recv_async().await.unwrap();
            task.cancel().await;
            assert!(session.responses().lock().await.connection.is_none());
            call(&session, &request(), &auth).await.unwrap();
            server.await;
        });
    }
    #[test_case(())]
    fn missing_response_recovery_is_bounded_without_status(_: ()) {
        smol::block_on(async {
            let (listener, auth) = server().await;
            let server = smol::spawn(async move {
                let (tcp, _) = listener.accept().await.unwrap();
                let mut socket = accept_async(tcp).await.unwrap();
                for _ in 0..2 {
                    received(&mut socket).await;
                    socket.send(json!({"type":"error","error":{"code":"previous_response_not_found","message":ERROR_MESSAGE}}).to_string().into()).await.unwrap();
                }
                assert!(socket.next().await.is_none_or(|result| result.is_err()));
            });
            let error = call(&session(), &request(), &auth).await.unwrap_err();
            assert!(!error.is_retryable());
            assert!(matches!(error, AgentError::Api { status: 400, .. }));
            server.await;
        });
    }
    #[test_case(())]
    fn empty_terminal_retains_parallel_output_items_in_index_order(_: ()) {
        smol::block_on(async {
            let (tx, _rx) = flume::unbounded();
            let mut accumulator = ResponseAccumulator::coding_plan();
            let first = json!({"type":"function_call", "call_id":"first", "name":TOOL_NAME, "arguments":"{}"});
            let second = json!({"type":"function_call", "call_id":"second", "name":TOOL_NAME, "arguments":"{}"});
            for (index, item) in [(1, &first), (2, &second)] {
                accumulator
                    .push(
                        "response.output_item.added",
                        &json!({"output_index":index,"item":item}),
                        &tx,
                    )
                    .await
                    .unwrap();
            }
            for (index, item) in [(2, &second), (1, &first)] {
                accumulator
                    .push(
                        "response.output_item.done",
                        &json!({"output_index":index,"item":item}),
                        &tx,
                    )
                    .await
                    .unwrap();
            }
            accumulator.push("response.output_item.done", &json!({"output_index":0,"item":{"type":"reasoning","encrypted_content":"opaque"}}), &tx).await.unwrap();
            let mut terminal = completed();
            terminal["response"]["output"] = json!([]);
            accumulator
                .push("response.completed", &terminal, &tx)
                .await
                .unwrap();
            let completed = accumulator.completed.clone().unwrap();
            assert_eq!(completed["output"][1], first);
            assert_eq!(completed["output"][2], second);
            let response = accumulator.finish().unwrap();
            let body = request();
            let previous = continuation(&body, &response, &completed).unwrap();
            let mut next = body;
            next["input"] = json!(previous.input);
            let results = json!([
                {"type":"function_call_output","call_id":"first","output":"first result"},
                {"type":"function_call_output","call_id":"second","output":"second result"}
            ]);
            next["input"]
                .as_array_mut()
                .unwrap()
                .extend(results.as_array().unwrap().iter().cloned());
            let (payload, reason) = request_payload(&next, Some(&previous));
            assert_eq!(reason, "incremental");
            assert_eq!(payload["input"], results);
        });
    }
    const TURN_STATE: &str = "retained-state";
    const LATER_STATE: &str = "ignored-state";
    const ROUTE: &str = "model=gpt-5.6-luna;tier=default";

    #[test_case("response.metadata", false ; "response_metadata")]
    #[test_case("codex.response.metadata", false ; "codex_metadata")]
    #[test_case("response.metadata", true ; "handshake_wins")]
    #[allow(clippy::result_large_err)]
    fn routing_feedback_survives_reconnect(event: &'static str, handshake: bool) {
        smol::block_on(async {
            let (listener, mut auth) = server().await;
            auth.set_header(ROUTING_HINT_HEADER, ROUTE.into());
            let server = smol::spawn(async move {
                let (tcp, _) = listener.accept().await.unwrap();
                let mut socket = accept_hdr_async(
                    tcp,
                    move |request: &HandshakeRequest, mut response: HandshakeResponse| {
                        assert_eq!(request.headers()[ROUTING_HINT_HEADER], ROUTE);
                        assert!(request.headers().get(TURN_STATE_HEADER).is_none());
                        if handshake {
                            response
                                .headers_mut()
                                .insert(TURN_STATE_HEADER, TURN_STATE.parse().unwrap());
                        }
                        Ok(response)
                    },
                )
                .await
                .unwrap();
                let first = received(&mut socket).await;
                assert_eq!(
                    first["client_metadata"][TURN_STATE_HEADER].as_str(),
                    handshake.then_some(TURN_STATE)
                );
                for value in [
                    if handshake { LATER_STATE } else { TURN_STATE },
                    LATER_STATE,
                ] {
                    socket
                        .send(
                            json!({"type":event,"headers":{TURN_STATE_HEADER:value}})
                                .to_string()
                                .into(),
                        )
                        .await
                        .unwrap();
                }
                response(&mut socket).await;
                let second = received(&mut socket).await;
                assert_eq!(second["client_metadata"][TURN_STATE_HEADER], TURN_STATE);
                assert_eq!(second["previous_response_id"], RESPONSE_ID);
                response(&mut socket).await;
                let (tcp, _) = listener.accept().await.unwrap();
                let mut socket = accept_hdr_async(
                    tcp,
                    |request: &HandshakeRequest, response: HandshakeResponse| {
                        assert_eq!(request.headers()[TURN_STATE_HEADER], TURN_STATE);
                        Ok(response)
                    },
                )
                .await
                .unwrap();
                let third = received(&mut socket).await;
                assert_eq!(third["client_metadata"][TURN_STATE_HEADER], TURN_STATE);
                assert!(third.get("previous_response_id").is_none());
                response(&mut socket).await;
                let fourth = received(&mut socket).await;
                assert!(fourth.get("client_metadata").is_none());
                response(&mut socket).await;
            });
            let session = session();
            let mut body = request();
            call(&session, &body, &auth).await.unwrap();
            assert_eq!(session.routing().value().as_deref(), Some(TURN_STATE));
            assert!(session.child().routing().value().is_none());
            followup(&mut body);
            call(&session, &body, &auth).await.unwrap();
            session.responses().lock().await.connection = None;
            call(&session, &body, &auth).await.unwrap();
            session.begin_turn().await;
            call(&session, &body, &auth).await.unwrap();
            server.await;
        });
    }

    #[test_case(false ; "model_change")]
    #[test_case(true ; "tier_change")]
    #[allow(clippy::result_large_err)]
    fn route_change_reconnects_without_clearing_turn_state(tier: bool) {
        smol::block_on(async {
            let (listener, mut auth) = server().await;
            let mut body = request();
            let original_hint = routing_hint(&body);
            auth.set_header(ROUTING_HINT_HEADER, original_hint.clone());
            let mut changed = body.clone();
            changed[if tier { "service_tier" } else { "model" }] =
                json!(if tier { "priority" } else { "gpt-other" });
            let changed_hint = routing_hint(&changed);
            let expected_hint = changed_hint.clone();
            let server = smol::spawn(async move {
                for hint in [original_hint, expected_hint] {
                    let (tcp, _) = listener.accept().await.unwrap();
                    let mut socket = accept_hdr_async(
                        tcp,
                        move |request: &HandshakeRequest, response: HandshakeResponse| {
                            assert_eq!(request.headers()[ROUTING_HINT_HEADER], hint);
                            Ok(response)
                        },
                    )
                    .await
                    .unwrap();
                    assert!(
                        received(&mut socket)
                            .await
                            .get("previous_response_id")
                            .is_none()
                    );
                    socket.send(json!({"type":"response.metadata","headers":{TURN_STATE_HEADER:TURN_STATE}}).to_string().into()).await.unwrap();
                    response(&mut socket).await;
                }
            });
            let session = session();
            call(&session, &body, &auth).await.unwrap();
            body = changed;
            auth.set_header(ROUTING_HINT_HEADER, changed_hint);
            call(&session, &body, &auth).await.unwrap();
            assert_eq!(session.responses().lock().await.generation, 2);
            assert_eq!(session.routing().value().as_deref(), Some(TURN_STATE));
            server.await;
        });
    }

    #[test_case(false ; "full_after_instruction_change")]
    #[test_case(true ; "full_after_reconnect")]
    fn encrypted_reasoning_continuation_and_full_replay(reconnect: bool) {
        smol::block_on(async {
            let (listener, auth) = server().await;
            let reasoning = json!({"type":"reasoning","id":"rs_test","summary":[],"encrypted_content":TURN_STATE});
            let expected = reasoning.clone();
            let server = smol::spawn(async move {
                let (tcp, _) = listener.accept().await.unwrap();
                let mut socket = accept_async(tcp).await.unwrap();
                received(&mut socket).await;
                let mut terminal = completed();
                terminal["response"]["output"]
                    .as_array_mut()
                    .unwrap()
                    .insert(0, expected.clone());
                socket
                    .send(
                        json!({"type":"response.output_text.delta","delta":ANSWER})
                            .to_string()
                            .into(),
                    )
                    .await
                    .unwrap();
                socket.send(terminal.to_string().into()).await.unwrap();
                let incremental = received(&mut socket).await;
                assert_eq!(incremental["previous_response_id"], RESPONSE_ID);
                assert_eq!(incremental["input"].as_array().unwrap().len(), 1);
                response(&mut socket).await;
                if reconnect {
                    let (tcp, _) = listener.accept().await.unwrap();
                    socket = accept_async(tcp).await.unwrap();
                }
                let full = received(&mut socket).await;
                assert!(full.get("previous_response_id").is_none());
                assert_eq!(full["input"][1], expected);
                assert_eq!(
                    full["input"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .filter(|item| item["type"] == "reasoning")
                        .count(),
                    1
                );
                response(&mut socket).await;
            });
            let session = session();
            let mut body = request();
            let first = call(&session, &body, &auth).await.unwrap().unwrap();
            assert!(
                matches!(&first.message.content[0], crate::ContentBlock::OpenAiReasoning { item } if item == &reasoning)
            );
            body["input"].as_array_mut().unwrap().extend(
                coding_plan_input(&[first.message])
                    .as_array()
                    .unwrap()
                    .iter()
                    .cloned(),
            );
            body["input"].as_array_mut().unwrap().extend(
                coding_plan_input(&[Message::user(ANSWER.into())])
                    .as_array()
                    .unwrap()
                    .iter()
                    .cloned(),
            );
            let second = call(&session, &body, &auth).await.unwrap().unwrap();
            body["input"].as_array_mut().unwrap().extend(
                coding_plan_input(&[second.message])
                    .as_array()
                    .unwrap()
                    .iter()
                    .cloned(),
            );
            if reconnect {
                session.responses().lock().await.connection = None;
            } else {
                body["instructions"] = json!(ANSWER);
            }
            call(&session, &body, &auth).await.unwrap();
            server.await;
        });
    }
}

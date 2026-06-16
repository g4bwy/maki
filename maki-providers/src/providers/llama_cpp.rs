use std::sync::{Arc, Mutex};
use std::time::Duration;

use flume::Sender;
use serde_json::{Value, json};

use crate::model::{Model, ModelEntry};
use crate::provider::{BoxFuture, Provider};
use crate::{AgentError, Message, ProviderEvent, RequestOptions, StreamResponse, ThinkingConfig};

use super::llama_cpp_server::{LlamaCppServer, ModelStatus};
use super::openai_compat::{OpenAiCompatConfig, OpenAiCompatProvider};
use super::{KeyPool, ResolvedAuth};

const HOST_ENV: &str = "LLAMA_CPP_HOST";
const API_KEY_ENV: &str = "LLAMA_CPP_API_KEY";
const HOST_NOT_SET: &str = "LLAMA_CPP_HOST not set";
const AUTO_LOAD_TIMEOUT_SECS: u64 = 120;

static CONFIG: OpenAiCompatConfig = OpenAiCompatConfig {
    api_key_env: "",
    base_url: "http://localhost:8080/v1",
    max_tokens_field: "max_tokens",
    include_stream_usage: true,
    provider_name: "LlamaCpp",
};

pub(crate) fn models() -> &'static [ModelEntry] {
    &[]
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ThinkingBudgets {
    pub minimal: u32,
    pub low: u32,
    pub medium: u32,
    pub high: u32,
}

impl Default for ThinkingBudgets {
    fn default() -> Self {
        Self {
            minimal: 1024,
            low: 2048,
            medium: 8192,
            high: 16384,
        }
    }
}

impl ThinkingBudgets {
    fn from_config(budgets: &maki_config::ThinkingBudgets) -> Self {
        Self {
            minimal: budgets.minimal,
            low: budgets.low,
            medium: budgets.medium,
            high: budgets.high,
        }
    }
}

pub struct LlamaCpp {
    compat: OpenAiCompatProvider,
    auth: Arc<Mutex<ResolvedAuth>>,
    key_pool: Option<KeyPool>,
    system_prefix: Option<String>,
    servers: Vec<LlamaCppServer>,
    #[allow(dead_code)]
    thinking_budgets: ThinkingBudgets,
    auto_load: bool,
}

impl LlamaCpp {
    pub fn new(timeouts: super::Timeouts) -> Result<Self, AgentError> {
        let key_pool = KeyPool::from_env(API_KEY_ENV).ok();
        Self::from_env(timeouts, key_pool, std::env::var(HOST_ENV).ok())
    }

    pub fn from_config(
        config: &maki_config::LlamaCppConfig,
        timeouts: super::Timeouts,
    ) -> Result<Self, AgentError> {
        let servers = config.resolve_servers();
        let first_server = servers.first().ok_or_else(|| AgentError::Config {
            message: "no llama-cpp servers configured".into(),
        })?;

        let base_url = format!("{}/v1", first_server.url);
        let headers = first_server
            .api_key
            .as_ref()
            .map(|key| vec![("authorization".into(), format!("Bearer {key}"))])
            .unwrap_or_default();

        let llama_servers: Vec<LlamaCppServer> = servers
            .into_iter()
            .map(|s| LlamaCppServer::new(&s.url, s.api_key))
            .collect();

        Ok(Self {
            compat: OpenAiCompatProvider::new(&CONFIG, timeouts),
            auth: Arc::new(Mutex::new(ResolvedAuth {
                base_url: Some(base_url),
                headers,
            })),
            key_pool: None,
            system_prefix: None,
            servers: llama_servers,
            thinking_budgets: ThinkingBudgets::from_config(&config.thinking_budgets),
            auto_load: config.auto_load,
        })
    }

    pub(crate) fn with_auth(auth: Arc<Mutex<ResolvedAuth>>, timeouts: super::Timeouts) -> Self {
        Self {
            compat: OpenAiCompatProvider::new(&CONFIG, timeouts),
            auth,
            key_pool: None,
            system_prefix: None,
            servers: Vec::new(),
            thinking_budgets: ThinkingBudgets::default(),
            auto_load: true,
        }
    }

    pub(crate) fn with_system_prefix(mut self, prefix: Option<String>) -> Self {
        self.system_prefix = prefix;
        self
    }

    fn from_env(
        timeouts: super::Timeouts,
        key_pool: Option<KeyPool>,
        host: Option<String>,
    ) -> Result<Self, AgentError> {
        let base_url = match host {
            Some(h) => format!("{h}/v1"),
            None => {
                return Err(AgentError::Config {
                    message: HOST_NOT_SET.into(),
                });
            }
        };
        let headers = match key_pool.as_ref().map(|p| p.current().to_string()) {
            Some(key) => vec![("authorization".into(), format!("Bearer {key}"))],
            None => Vec::new(),
        };
        Ok(Self {
            compat: OpenAiCompatProvider::new(&CONFIG, timeouts),
            auth: Arc::new(Mutex::new(ResolvedAuth {
                base_url: Some(base_url),
                headers,
            })),
            key_pool,
            system_prefix: None,
            servers: Vec::new(),
            thinking_budgets: ThinkingBudgets::default(),
            auto_load: true,
        })
    }

    #[allow(dead_code)]
    fn resolve_server_for_model(&self, model: &Model) -> Option<&LlamaCppServer> {
        self.servers
            .iter()
            .find(|s| s.models.iter().any(|m| m.id == model.id))
            .or(self.servers.first())
    }

    fn inject_thinking_config(&self, body: &mut Value, thinking: ThinkingConfig) {
        match thinking {
            ThinkingConfig::Off => {
                body["chat_template_kwargs"] = json!({ "enable_thinking": false });
            }
            ThinkingConfig::Budget(n) => {
                body["thinking_budget_tokens"] = json!(n);
            }
            ThinkingConfig::Adaptive => {}
        }
    }

    async fn ensure_model_loaded(&self, model_id: &str) -> Result<(), AgentError> {
        if !self.auto_load {
            return Ok(());
        }
        let server = match self.servers.first() {
            Some(s) => s,
            None => return Ok(()),
        };
        let status = server.model_status(model_id).await?;
        if status == ModelStatus::Unloaded {
            server.load_model(model_id).await?;
            server
                .poll_until_loaded(model_id, Duration::from_secs(AUTO_LOAD_TIMEOUT_SECS))
                .await?;
        }
        Ok(())
    }
}

impl Provider for LlamaCpp {
    fn stream_message<'a>(
        &'a self,
        model: &'a Model,
        messages: &'a [Message],
        system: &'a str,
        tools: &'a Value,
        event_tx: &'a Sender<ProviderEvent>,
        opts: RequestOptions,
        _session_id: Option<&str>,
    ) -> BoxFuture<'a, Result<StreamResponse, AgentError>> {
        Box::pin(async move {
            let auth = self.auth.lock().unwrap().clone();
            let mut buf = String::new();
            let system = super::with_prefix(&self.system_prefix, system, &mut buf);
            let mut body = self.compat.build_body(model, messages, system, tools);
            self.inject_thinking_config(&mut body, opts.thinking);

            self.ensure_model_loaded(&model.id).await?;

            self.compat
                .do_stream(model, &[], &body, event_tx, &auth)
                .await
        })
    }

    fn list_models(&self) -> BoxFuture<'_, Result<Vec<String>, AgentError>> {
        Box::pin(async move {
            let mut all_models: Vec<String> = Vec::new();

            for server in &self.servers {
                if let Ok(models) = server.fetch_models().await {
                    for m in models {
                        if !all_models.contains(&m.id) {
                            all_models.push(m.id);
                        }
                    }
                }
            }

            all_models.sort();
            Ok(all_models)
        })
    }

    fn rotate_key(&self) -> BoxFuture<'_, Result<bool, AgentError>> {
        Box::pin(async {
            Ok(self.key_pool.as_ref().is_some_and(|p| {
                p.rotate_headers(&self.auth, |key| {
                    vec![("authorization".into(), format!("Bearer {key}"))]
                })
            }))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_TIMEOUTS: super::super::Timeouts = super::super::Timeouts {
        connect: Duration::from_secs(10),
        low_speed: Duration::from_secs(30),
        stream: Duration::from_secs(300),
    };

    #[test]
    fn from_env_without_host_or_api_key_errors() {
        match LlamaCpp::from_env(TEST_TIMEOUTS, None, None) {
            Err(AgentError::Config { message }) => assert_eq!(message, HOST_NOT_SET),
            Err(other) => panic!("expected Config error, got {other:?}"),
            Ok(_) => panic!("expected error when host and api_key are None"),
        }
    }

    #[test]
    fn from_env_with_host_builds_auth() {
        let llama = LlamaCpp::from_env(TEST_TIMEOUTS, None, Some("http://x:1234".into())).unwrap();
        let auth = llama.auth.lock().unwrap();
        assert_eq!(auth.base_url.as_deref(), Some("http://x:1234/v1"));
        assert!(auth.headers.is_empty());
    }

    #[test]
    fn from_env_with_api_key_uses_host_with_auth() {
        let pool = KeyPool::from_keys(vec!["test-key".into()]);
        let llama = LlamaCpp::from_env(TEST_TIMEOUTS, Some(pool), Some("http://local:1234".into()))
            .unwrap();
        let auth = llama.auth.lock().unwrap();
        assert_eq!(auth.base_url.as_deref(), Some("http://local:1234/v1"));
        assert_eq!(auth.headers.len(), 1);
        assert_eq!(auth.headers[0].1, "Bearer test-key");
    }

    #[test]
    fn thinking_budgets_defaults() {
        let budgets = ThinkingBudgets::default();
        assert_eq!(budgets.minimal, 1024);
        assert_eq!(budgets.low, 2048);
        assert_eq!(budgets.medium, 8192);
        assert_eq!(budgets.high, 16384);
    }

    #[test]
    fn inject_thinking_config_off() {
        let mut body = json!({"model": "test"});
        let llama = LlamaCpp::from_env(TEST_TIMEOUTS, None, Some("http://x:1234".into())).unwrap();
        llama.inject_thinking_config(&mut body, ThinkingConfig::Off);
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
        assert!(body.get("thinking_budget_tokens").is_none());
    }

    #[test]
    fn inject_thinking_config_budget() {
        let mut body = json!({"model": "test"});
        let llama = LlamaCpp::from_env(TEST_TIMEOUTS, None, Some("http://x:1234".into())).unwrap();
        llama.inject_thinking_config(&mut body, ThinkingConfig::Budget(4096));
        assert_eq!(body["thinking_budget_tokens"], 4096);
        assert!(body.get("chat_template_kwargs").is_none());
    }

    #[test]
    fn inject_thinking_config_adaptive() {
        let mut body = json!({"model": "test"});
        let llama = LlamaCpp::from_env(TEST_TIMEOUTS, None, Some("http://x:1234".into())).unwrap();
        llama.inject_thinking_config(&mut body, ThinkingConfig::Adaptive);
        assert!(body.get("chat_template_kwargs").is_none());
        assert!(body.get("thinking_budget_tokens").is_none());
    }

    #[test]
    fn from_config_with_servers() {
        let config = maki_config::LlamaCppConfig {
            servers: vec![maki_config::LlamaCppServerConfig {
                url: "http://server1:8080".to_string(),
                api_key: Some("key1".to_string()),
            }],
            thinking_budgets: maki_config::ThinkingBudgets {
                minimal: 512,
                low: 1024,
                medium: 4096,
                high: 8192,
            },
            auto_load: false,
        };
        let llama = LlamaCpp::from_config(&config, TEST_TIMEOUTS).unwrap();
        assert_eq!(llama.servers.len(), 1);
        assert_eq!(llama.servers[0].url, "http://server1:8080");
        assert_eq!(llama.thinking_budgets.minimal, 512);
        assert!(!llama.auto_load);
    }

    #[test]
    fn from_config_with_multiple_servers() {
        let config = maki_config::LlamaCppConfig {
            servers: vec![
                maki_config::LlamaCppServerConfig {
                    url: "http://server1:8080".to_string(),
                    api_key: None,
                },
                maki_config::LlamaCppServerConfig {
                    url: "http://server2:9090".to_string(),
                    api_key: Some("key2".to_string()),
                },
            ],
            thinking_budgets: maki_config::ThinkingBudgets::default(),
            auto_load: true,
        };
        let llama = LlamaCpp::from_config(&config, TEST_TIMEOUTS).unwrap();
        assert_eq!(llama.servers.len(), 2);
        assert_eq!(llama.servers[0].url, "http://server1:8080");
        assert_eq!(llama.servers[1].url, "http://server2:9090");
        assert!(llama.auto_load);
    }

    #[test]
    fn from_config_strips_trailing_slash() {
        let config = maki_config::LlamaCppConfig {
            servers: vec![maki_config::LlamaCppServerConfig {
                url: "http://server1:8080/".to_string(),
                api_key: None,
            }],
            thinking_budgets: maki_config::ThinkingBudgets::default(),
            auto_load: true,
        };
        let llama = LlamaCpp::from_config(&config, TEST_TIMEOUTS).unwrap();
        assert_eq!(llama.servers[0].url, "http://server1:8080");
    }

    #[test]
    fn from_config_no_servers_falls_back_to_default() {
        // Clear env vars that would interfere with fallback chain
        unsafe {
            std::env::remove_var("LLAMA_SERVER_URL");
            std::env::remove_var("LLAMA_CPP_HOST");
        }
        let config = maki_config::LlamaCppConfig {
            servers: vec![],
            thinking_budgets: maki_config::ThinkingBudgets::default(),
            auto_load: true,
        };
        let llama = LlamaCpp::from_config(&config, TEST_TIMEOUTS).unwrap();
        assert_eq!(llama.servers[0].url, "http://127.0.0.1:8080");
    }

    #[test]
    fn resolve_server_for_model_returns_first_when_no_match() {
        let llama = LlamaCpp::from_env(TEST_TIMEOUTS, None, Some("http://x:1234".into())).unwrap();
        assert!(llama.servers.is_empty(), "no servers configured via env");
    }
}

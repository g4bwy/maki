use std::time::{Duration, Instant};

use isahc::{AsyncReadResponseExt, config::Configurable};
use serde::Deserialize;

use crate::AgentError;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerMode {
    Router,
    Single,
    Legacy,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelStatus {
    Loaded,
    Loading,
    Failed,
    Sleeping,
    Unloaded,
    Unauthorized,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ServerModel {
    pub id: String,
    pub name: String,
    pub status: ModelStatus,
    pub capabilities: Vec<ModelCapability>,
    pub context_size: u32,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelCapability {
    Text,
    Image,
}

#[derive(Debug, Clone)]
pub struct LlamaCppServer {
    pub url: String,
    pub api_key: Option<String>,
    #[allow(dead_code)]
    pub mode: Option<ServerMode>,
    #[allow(dead_code)]
    pub models: Vec<ServerModel>,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct HealthResponse {
    #[allow(dead_code)]
    status: String,
}

#[derive(Deserialize)]
struct ModelsResponse {
    #[serde(default)]
    data: Vec<ModelData>,
    #[serde(default)]
    models: Vec<SingleModelData>,
}

#[derive(Deserialize)]
struct ModelData {
    #[serde(alias = "id")]
    model: String,
    #[serde(default)]
    aliases: Vec<String>,
    #[serde(default)]
    meta: ModelMeta,
    #[serde(default)]
    status: String,
    #[serde(default)]
    architecture: Option<String>,
}

#[derive(Deserialize, Default)]
struct ModelMeta {
    #[serde(default)]
    n_ctx: Option<u32>,
}

#[derive(Deserialize)]
struct SingleModelData {
    #[serde(alias = "id")]
    model: String,
}

#[allow(dead_code)]
#[derive(Deserialize)]
pub(crate) struct PropsResponse {
    #[serde(default)]
    role: String,
    #[serde(default)]
    error: Option<PropError>,
    #[serde(default)]
    modalities: Vec<String>,
    #[serde(default)]
    is_sleeping: bool,
    #[serde(default)]
    chat_template_caps: Vec<String>,
    #[serde(default)]
    n_ctx: Option<u32>,
}

#[derive(Deserialize)]
struct PropError {
    #[serde(default)]
    code: u16,
    #[serde(default)]
    message: String,
}

impl LlamaCppServer {
    pub fn new(url: &str, api_key: Option<String>) -> Self {
        let url = url.strip_suffix('/').unwrap_or(url).to_string();
        Self {
            url,
            api_key,
            mode: None,
            models: Vec::new(),
        }
    }

    fn client(&self) -> isahc::HttpClient {
        isahc::HttpClient::builder()
            .connect_timeout(Duration::from_secs(10))
            .low_speed_timeout(1, Duration::from_secs(30))
            .build()
            .expect("failed to build HTTP client")
    }

    fn request(&self, method: &str, path: &str) -> isahc::http::request::Builder {
        let mut builder = isahc::Request::builder()
            .method(method)
            .uri(format!("{}/{}", self.url, path));
        if let Some(key) = &self.api_key {
            builder = builder.header("Authorization", format!("Bearer {key}"));
        }
        builder
    }

    #[allow(dead_code)]
    pub async fn health(&self, _timeout: Duration) -> Result<ModelStatus, AgentError> {
        let client = self.client();
        let request = self.request("GET", "health").body(())?;
        let mut response = match client.send_async(request).await {
            Ok(r) => r,
            Err(_) => return Ok(ModelStatus::Unloaded),
        };
        let status = response.status().as_u16();
        if status == 200 {
            Ok(ModelStatus::Loaded)
        } else if status == 401 {
            Ok(ModelStatus::Unauthorized)
        } else {
            let _ = response.text().await;
            Ok(ModelStatus::Unloaded)
        }
    }

    pub async fn fetch_models(&self) -> Result<Vec<ServerModel>, AgentError> {
        let client = self.client();
        let request = self.request("GET", "v1/models").body(())?;
        let mut response = client.send_async(request).await?;
        if response.status().as_u16() != 200 {
            return Ok(Vec::new());
        }
        let body: ModelsResponse = serde_json::from_str(&response.text().await?)?;

        let mut models: Vec<ServerModel> = body
            .data
            .into_iter()
            .map(|m| {
                let name = m
                    .aliases
                    .first()
                    .cloned()
                    .unwrap_or_else(|| m.model.clone());
                let status = match m.status.as_str() {
                    "loaded" => ModelStatus::Loaded,
                    "loading" => ModelStatus::Loading,
                    "failed" => ModelStatus::Failed,
                    "sleeping" => ModelStatus::Sleeping,
                    _ => ModelStatus::Unloaded,
                };
                let context_size = m.meta.n_ctx.unwrap_or(128_000);
                let capabilities = if m
                    .architecture
                    .as_ref()
                    .is_some_and(|a| a.contains("clip") || a.contains("mllama"))
                {
                    vec![ModelCapability::Text, ModelCapability::Image]
                } else {
                    vec![ModelCapability::Text]
                };
                ServerModel {
                    id: m.model,
                    name,
                    status,
                    capabilities,
                    context_size,
                }
            })
            .collect();

        if models.is_empty() {
            models = body
                .models
                .into_iter()
                .map(|m| ServerModel {
                    id: m.model.clone(),
                    name: m.model,
                    status: ModelStatus::Loaded,
                    capabilities: vec![ModelCapability::Text],
                    context_size: 128_000,
                })
                .collect();
        }

        models.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(models)
    }

    pub async fn fetch_props(&self, model_id: Option<&str>) -> Result<PropsResponse, AgentError> {
        let client = self.client();
        let path = match model_id {
            Some(id) => format!("props?model={}&autoload=false", super::urlenc(id)),
            None => "props".to_string(),
        };
        let request = self.request("GET", &path).body(())?;
        let mut response = client.send_async(request).await?;
        let status = response.status().as_u16();
        let text = response.text().await.unwrap_or_default();

        let props: PropsResponse = match serde_json::from_str(&text) {
            Ok(p) => p,
            Err(_) => {
                return Ok(PropsResponse {
                    role: String::new(),
                    error: Some(PropError {
                        code: status,
                        message: text,
                    }),
                    modalities: Vec::new(),
                    is_sleeping: false,
                    chat_template_caps: Vec::new(),
                    n_ctx: None,
                });
            }
        };
        Ok(props)
    }

    #[allow(dead_code)]
    pub async fn detect_mode(&self) -> Result<ServerMode, AgentError> {
        let props = self.fetch_props(None).await?;
        if props.role == "router" {
            return Ok(ServerMode::Router);
        }
        let models = self.fetch_models().await?;
        if models.iter().any(|m| m.context_size != 128_000) {
            return Ok(ServerMode::Legacy);
        }
        Ok(ServerMode::Single)
    }

    pub async fn load_model(&self, model_id: &str) -> Result<(), AgentError> {
        let client = self.client();
        let body = serde_json::json!({ "model": model_id });
        let json_body = serde_json::to_vec(&body)?;
        let request = self
            .request("POST", "models/load")
            .header("Content-Type", "application/json")
            .body(json_body)?;
        let mut response = client.send_async(request).await?;
        let status = response.status().as_u16();
        if (200..300).contains(&status) {
            Ok(())
        } else {
            let text = response.text().await.unwrap_or_default();
            Err(AgentError::Api {
                status,
                message: text,
            })
        }
    }

    #[allow(dead_code)]
    pub async fn unload_model(&self, model_id: &str) -> Result<(), AgentError> {
        let client = self.client();
        let body = serde_json::json!({ "model": model_id });
        let json_body = serde_json::to_vec(&body)?;
        let request = self
            .request("POST", "models/unload")
            .header("Content-Type", "application/json")
            .body(json_body)?;
        let mut response = client.send_async(request).await?;
        let status = response.status().as_u16();
        if (200..300).contains(&status) {
            Ok(())
        } else {
            let text = response.text().await.unwrap_or_default();
            Err(AgentError::Api {
                status,
                message: text,
            })
        }
    }

    pub async fn poll_until_loaded(
        &self,
        model_id: &str,
        timeout: Duration,
    ) -> Result<(), AgentError> {
        let deadline = Instant::now() + timeout;
        loop {
            let status = self.model_status(model_id).await?;
            match status {
                ModelStatus::Loaded | ModelStatus::Sleeping => return Ok(()),
                ModelStatus::Failed => {
                    return Err(AgentError::Api {
                        status: 500,
                        message: format!("model {model_id} failed to load"),
                    });
                }
                ModelStatus::Unauthorized => {
                    return Err(AgentError::Api {
                        status: 401,
                        message: format!("model {model_id} unauthorized"),
                    });
                }
                _ => {}
            }
            if Instant::now() >= deadline {
                return Err(AgentError::Timeout {
                    secs: timeout.as_secs(),
                });
            }
            smol::Timer::after(Duration::from_secs(1)).await;
        }
    }

    pub async fn model_status(&self, model_id: &str) -> Result<ModelStatus, AgentError> {
        let props = self.fetch_props(Some(model_id)).await?;
        if props.is_sleeping {
            return Ok(ModelStatus::Sleeping);
        }
        if let Some(err) = &props.error {
            return match err.code {
                401 => Ok(ModelStatus::Unauthorized),
                503 => Ok(ModelStatus::Loading),
                400 if err.message.contains("model is not loaded") => Ok(ModelStatus::Unloaded),
                _ => Ok(ModelStatus::Failed),
            };
        }
        Ok(ModelStatus::Loaded)
    }

    #[allow(dead_code)]
    pub async fn context_size(&self, model_id: &str) -> Result<u32, AgentError> {
        let props = self.fetch_props(Some(model_id)).await?;
        if let Some(n_ctx) = props.n_ctx {
            return Ok(n_ctx);
        }
        let models = self.fetch_models().await?;
        if let Some(model) = models.iter().find(|m| m.id == model_id) {
            return Ok(model.context_size);
        }
        Ok(128_000)
    }

    #[allow(dead_code)]
    pub async fn capabilities(&self, model_id: &str) -> Result<Vec<ModelCapability>, AgentError> {
        let models = self.fetch_models().await?;
        if let Some(model) = models.iter().find(|m| m.id == model_id) {
            return Ok(model.capabilities.clone());
        }
        let props = self.fetch_props(Some(model_id)).await?;
        let has_image = props.modalities.iter().any(|m| m == "image");
        if has_image {
            Ok(vec![ModelCapability::Text, ModelCapability::Image])
        } else {
            Ok(vec![ModelCapability::Text])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_strips_trailing_slash() {
        let server = LlamaCppServer::new("http://localhost:8080/", None);
        assert_eq!(server.url, "http://localhost:8080");
    }

    #[test]
    fn new_preserves_url_without_trailing_slash() {
        let server = LlamaCppServer::new("http://localhost:8080", Some("key".into()));
        assert_eq!(server.url, "http://localhost:8080");
        assert_eq!(server.api_key.as_deref(), Some("key"));
    }

    #[test]
    fn model_status_from_props_loaded() {
        let props = PropsResponse {
            role: String::new(),
            error: None,
            modalities: vec!["text".into()],
            is_sleeping: false,
            chat_template_caps: vec![],
            n_ctx: Some(8192),
        };
        assert!(props.error.is_none());
        assert!(!props.is_sleeping);
    }

    #[test]
    fn model_status_from_props_sleeping() {
        let props = PropsResponse {
            is_sleeping: true,
            ..default_props()
        };
        assert!(props.is_sleeping);
    }

    #[test]
    fn model_status_from_props_unauthorized() {
        let props = PropsResponse {
            error: Some(PropError {
                code: 401,
                message: "Unauthorized".into(),
            }),
            ..default_props()
        };
        assert_eq!(props.error.as_ref().unwrap().code, 401);
    }

    #[test]
    fn model_status_from_props_loading() {
        let props = PropsResponse {
            error: Some(PropError {
                code: 503,
                message: "Loading".into(),
            }),
            ..default_props()
        };
        assert_eq!(props.error.as_ref().unwrap().code, 503);
    }

    #[test]
    fn model_status_from_props_unloaded() {
        let props = PropsResponse {
            error: Some(PropError {
                code: 400,
                message: "model is not loaded".into(),
            }),
            ..default_props()
        };
        let err = props.error.as_ref().unwrap();
        assert_eq!(err.code, 400);
        assert!(err.message.contains("model is not loaded"));
    }

    fn default_props() -> PropsResponse {
        PropsResponse {
            role: String::new(),
            error: None,
            modalities: Vec::new(),
            is_sleeping: false,
            chat_template_caps: Vec::new(),
            n_ctx: None,
        }
    }

    #[test]
    fn server_mode_router() {
        let props = PropsResponse {
            role: "router".into(),
            ..default_props()
        };
        assert_eq!(props.role, "router");
    }

    #[test]
    fn server_mode_single() {
        let props = PropsResponse {
            role: String::new(),
            ..default_props()
        };
        assert_eq!(props.role, "");
    }

    #[test]
    fn model_capability_text_only() {
        let caps: Vec<ModelCapability> = vec![ModelCapability::Text];
        assert_eq!(caps.len(), 1);
        assert!(!caps.contains(&ModelCapability::Image));
    }

    #[test]
    fn model_capability_multimodal() {
        let caps: Vec<ModelCapability> = vec![ModelCapability::Text, ModelCapability::Image];
        assert_eq!(caps.len(), 2);
        assert!(caps.contains(&ModelCapability::Image));
    }

    #[test]
    fn model_status_enum_variants() {
        assert_eq!(ModelStatus::Loaded, ModelStatus::Loaded);
        assert_eq!(ModelStatus::Loading, ModelStatus::Loading);
        assert_eq!(ModelStatus::Failed, ModelStatus::Failed);
        assert_eq!(ModelStatus::Sleeping, ModelStatus::Sleeping);
        assert_eq!(ModelStatus::Unloaded, ModelStatus::Unloaded);
        assert_eq!(ModelStatus::Unauthorized, ModelStatus::Unauthorized);
        assert_ne!(ModelStatus::Loaded, ModelStatus::Unloaded);
    }

    #[test]
    fn server_mode_enum_variants() {
        assert_eq!(ServerMode::Router, ServerMode::Router);
        assert_eq!(ServerMode::Single, ServerMode::Single);
        assert_eq!(ServerMode::Legacy, ServerMode::Legacy);
    }

    #[test]
    fn fetch_models_parses_model_data() {
        let json = r#"{
            "data": [
                {
                    "id": "llama3.1-8b",
                    "aliases": ["llama3.1"],
                    "meta": {"n_ctx": 32768},
                    "status": "loaded"
                }
            ]
        }"#;
        let models: ModelsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(models.data.len(), 1);
        assert_eq!(models.data[0].model, "llama3.1-8b");
        assert_eq!(models.data[0].aliases, vec!["llama3.1"]);
        assert_eq!(models.data[0].meta.n_ctx, Some(32768));
        assert_eq!(models.data[0].status, "loaded");
    }

    #[test]
    fn fetch_models_parses_fallback_format() {
        let json = r#"{
            "models": [
                {"id": "model-1"},
                {"id": "model-2"}
            ]
        }"#;
        let models: ModelsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(models.models.len(), 2);
        assert_eq!(models.models[0].model, "model-1");
    }

    #[test]
    fn health_response_parses() {
        let json = r#"{"status": "ok"}"#;
        let health: HealthResponse = serde_json::from_str(json).unwrap();
        assert_eq!(health.status, "ok");
    }

    #[test]
    fn props_response_with_n_ctx() {
        let json = r#"{"n_ctx": 131072, "modalities": ["text", "image"]}"#;
        let props: PropsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(props.n_ctx, Some(131072));
        assert!(props.modalities.contains(&"image".to_string()));
    }
}

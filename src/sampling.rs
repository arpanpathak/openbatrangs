//! # Sampling for each model family
//!
//! Each model family is tuned for its own sampling settings, published in its
//! model card and in the `generation_config.json` that ships with the
//! weights. Gemma 4 ships with temperature 1.0, top-p 0.95 and top-k 64; at a
//! low temperature its reasoning tends to loop on phrases such as "wait, I
//! should". Requests therefore use the family's settings, and a temperature
//! the user sets replaces only the temperature.
//!
//! ## References
//!
//! - Gemma 4 `generation_config.json`: <https://huggingface.co/google/gemma-4-31B-it/blob/main/generation_config.json>
//! - Qwen3 best practices: <https://huggingface.co/Qwen/Qwen3-8B#best-practices>

use serde_json::{Value, json};

/// The sampling fields of one chat request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Sampling {
    /// Softmax temperature.
    pub(crate) temperature: f64,
    /// Nucleus sampling threshold, or `None` to leave the server's default.
    pub(crate) top_p: Option<f64>,
    /// Number of most likely tokens kept, or `None` to leave the server's
    /// default.
    pub(crate) top_k: Option<u32>,
}

/// Recommended sampling for each family, keyed by a substring of the
/// lowercase model name.
const FAMILY_SAMPLING: [(&str, Sampling); 2] = [
    (
        "gemma",
        Sampling {
            temperature: 1.0,
            top_p: Some(0.95),
            top_k: Some(64),
        },
    ),
    (
        "qwen",
        Sampling {
            temperature: 0.6,
            top_p: Some(0.95),
            top_k: Some(20),
        },
    ),
];

/// Sampling for a model outside [`FAMILY_SAMPLING`]: a low temperature, which
/// keeps tool calls well formed.
const OTHER_SAMPLING: Sampling = Sampling {
    temperature: 0.2,
    top_p: None,
    top_k: None,
};

impl Sampling {
    /// The family's sampling for `model`, with `temperature` in place of the
    /// family's when the user set one.
    pub(crate) fn for_model(model: &str, temperature: Option<f64>) -> Self {
        let name = model.to_lowercase();
        let family = FAMILY_SAMPLING
            .iter()
            .find(|(family, _)| name.contains(family))
            .map_or(OTHER_SAMPLING, |(_, sampling)| *sampling);

        Self {
            temperature: temperature.unwrap_or(family.temperature),
            ..family
        }
    }

    /// Writes the fields into `target`: an OpenAI request body or an Ollama
    /// `options` object, which use the same names.
    pub(crate) fn write_to(self, target: &mut Value) {
        target["temperature"] = json!(self.temperature);

        if let Some(top_p) = self.top_p {
            target["top_p"] = json!(top_p);
        }

        if let Some(top_k) = self.top_k {
            target["top_k"] = json!(top_k);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_family_gets_its_model_card_sampling() {
        let gemma = Sampling::for_model("Gemma-4-26B-A4B-NVFP4", None);
        let qwen = Sampling::for_model("qwen3.6:35b", None);
        let other = Sampling::for_model("llama3.2", None);

        assert_eq!((gemma.temperature, gemma.top_k), (1.0, Some(64)));
        assert_eq!((qwen.temperature, qwen.top_k), (0.6, Some(20)));
        assert_eq!(other, OTHER_SAMPLING);
    }

    #[test]
    fn a_user_temperature_replaces_only_the_temperature() {
        let sampling = Sampling::for_model("gemma4:31b", Some(0.4));

        assert_eq!(sampling.temperature, 0.4);
        assert_eq!(sampling.top_p, Some(0.95));
        assert_eq!(sampling.top_k, Some(64));
    }

    #[test]
    fn only_the_fields_that_are_set_are_written() {
        let mut gemma = json!({});
        let mut other = json!({});

        Sampling::for_model("gemma", None).write_to(&mut gemma);
        Sampling::for_model("llama", None).write_to(&mut other);

        assert_eq!(
            gemma,
            json!({"temperature": 1.0, "top_p": 0.95, "top_k": 64})
        );
        assert_eq!(other, json!({"temperature": 0.2}));
    }
}

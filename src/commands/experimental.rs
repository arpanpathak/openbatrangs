//! Experimental engine probe and benchmark commands.
//!
//! `experimental doctor` prints which engines are available. `experimental bench`
//! runs each selected engine through a fixed workload, samples board power with
//! `tegrastats`, and produces a markdown report with:
//!
//! - tokens/second
//! - average watts
//! - tokens/second per watt
//! - estimated USD per million tokens (energy + hardware amortization)

use crate::cli::BenchArgs;
use crate::constants::engine::{
    BENCH_DEFAULT_PROMPT, BENCH_DEVICE_DUTY_CYCLE, BENCH_DEVICE_LIFETIME_YEARS,
    VLLM_NOT_AVAILABLE_REASON,
};
use crate::engine::{create_backend, resolve_engine_kinds, BenchSample, EngineConfig, EngineKind};
use crate::model_select::calculate_memory_budget;
use crate::models;
use crate::ollama::OllamaClient;
use crate::perf::PowerSampler;
use anyhow::{bail, Context, Result};
use std::fmt::Write as _;
use std::path::PathBuf;

/// Seconds per year used for hardware amortization math.
const SECONDS_PER_YEAR: f64 = 365.0 * 24.0 * 60.0 * 60.0;

/// Probe which experimental engines are available on this machine.
pub(crate) async fn experimental_doctor(client: &OllamaClient) -> Result<()> {
    for kind in EngineKind::all() {
        let config = EngineConfig::new(*kind, &client.base_url, None);
        let backend = create_backend(&config)?;
        let name = backend.kind().display_name();
        match backend.is_available().await {
            true => println!("✅ {name} — available"),
            false => println!("❌ {name} — not available"),
        }
    }
    println!("\nvLLM: {VLLM_NOT_AVAILABLE_REASON}");
    Ok(())
}

/// Run the experimental benchmark harness.
pub(crate) async fn experimental_bench(client: &OllamaClient, args: BenchArgs) -> Result<()> {
    let kinds = resolve_engine_kinds(&args.engines)?;
    let mut outcomes = Vec::new();

    for kind in kinds {
        let config = EngineConfig {
            kind,
            ollama_url: client.base_url.clone(),
            model: args.model.clone(),
            trtexec_seq_len: args.seq_len,
            trtexec_avg_runs: args.avg_runs,
            trt_shapes: args.trt_shapes.clone(),
        };
        let backend = create_backend(&config)?;
        if !backend.is_available().await {
            outcomes.push(EngineOutcome::Skipped {
                kind,
                reason: "engine not available on this machine".to_string(),
            });
            continue;
        }

        let model = resolve_bench_model(client, kind, args.model.as_deref()).await?;
        let config = EngineConfig {
            model: Some(model.clone()),
            ..config
        };
        let backend = create_backend(&config)?;
        let prompt = args
            .prompt
            .clone()
            .unwrap_or_else(|| BENCH_DEFAULT_PROMPT.to_string());
        let mut samples = Vec::new();
        let mut power_readings = Vec::new();

        for iteration in 0..args.iterations {
            println!(
                "⚡ {}/{} {} ({})",
                iteration + 1,
                args.iterations,
                kind.display_name(),
                model
            );
            let sampler = PowerSampler::start();
            let sample = backend
                .bench_generate(&prompt, args.max_tokens)
                .await
                .with_context(|| format!("benchmark failed for {}", kind.as_str()))?;
            let watts = sampler.average_watts();
            drop(sampler);
            println!(
                "   {:.1} tok/s · {} W · {}",
                sample.tokens_per_second(0),
                watts
                    .map(|w| format!("{w:.1}"))
                    .unwrap_or_else(|| "n/a".to_string()),
                sample.notes
            );
            power_readings.push(watts);
            samples.push(sample);
        }

        outcomes.push(EngineOutcome::Benchmarked(EngineReport::from_samples(
            kind,
            model,
            samples,
            power_readings,
        )));
    }

    let markdown = render_report(&outcomes, &args)?;
    println!("\n{markdown}");
    if let Some(path) = args.output {
        write_report(&path, &markdown)?;
    }
    Ok(())
}

/// Pick the model for an engine when the user did not provide one.
async fn resolve_bench_model(
    client: &OllamaClient,
    kind: EngineKind,
    explicit: Option<&str>,
) -> Result<String> {
    match (kind, explicit) {
        (_, Some(model)) => Ok(model.to_string()),
        (EngineKind::Ollama, None) => {
            let tags = client.tags().await?;
            let mem_budget = calculate_memory_budget();
            let best = models::score_models(&tags, mem_budget, 8_192)
                .into_iter()
                .next()
                .context("no suitable Ollama model installed; pass --model")?;
            Ok(best.name)
        }
        (EngineKind::TensorRt, None) => {
            bail!("TensorRT benchmark requires --model pointing to an .onnx file")
        }
    }
}

/// Per-engine benchmark outcome.
enum EngineOutcome {
    Benchmarked(EngineReport),
    Skipped { kind: EngineKind, reason: String },
}

/// Aggregated benchmark results for one engine.
struct EngineReport {
    kind: EngineKind,
    model: String,
    samples: Vec<BenchSample>,
    avg_tokens_per_sec: f64,
    avg_power_watts: Option<f64>,
    tokens_per_sec_per_watt: Option<f64>,
    usd_per_million_tokens: Option<f64>,
}

impl EngineReport {
    fn from_samples(
        kind: EngineKind,
        model: String,
        samples: Vec<BenchSample>,
        power_readings: Vec<Option<f64>>,
    ) -> Self {
        let avg_tokens_per_sec = mean(samples.iter().map(|sample| sample.tokens_per_second(0)));
        let watts: Vec<f64> = power_readings.into_iter().flatten().collect();
        let avg_power_watts = mean_opt(watts.iter().copied());
        let tokens_per_sec_per_watt = avg_power_watts.map(|watts| avg_tokens_per_sec / watts);
        let usd_per_million_tokens = energy_cost_per_million(avg_tokens_per_sec, avg_power_watts)
            .map(|energy| {
                let hardware = hardware_cost_per_million(avg_tokens_per_sec);
                energy + hardware
            });
        Self {
            kind,
            model,
            samples,
            avg_tokens_per_sec,
            avg_power_watts,
            tokens_per_sec_per_watt,
            usd_per_million_tokens,
        }
    }
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let values: Vec<f64> = values.collect();
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn mean_opt(values: impl Iterator<Item = f64>) -> Option<f64> {
    let values: Vec<f64> = values.collect();
    if values.is_empty() {
        return None;
    }
    Some(values.iter().sum::<f64>() / values.len() as f64)
}

/// Energy cost in USD per million tokens at the given price/kWh.
fn energy_cost_per_million(tokens_per_sec: f64, watts: Option<f64>) -> Option<f64> {
    let watts = watts?;
    if tokens_per_sec <= 0.0 || watts <= 0.0 {
        return None;
    }
    let seconds_per_million = 1_000_000.0 / tokens_per_sec;
    let kwh = seconds_per_million * watts / 1_000.0 / 3_600.0;
    Some(kwh * crate::constants::engine::BENCH_DEFAULT_ELECTRICITY_USD_PER_KWH)
}

/// Hardware amortization cost in USD per million tokens.
fn hardware_cost_per_million(tokens_per_sec: f64) -> f64 {
    if tokens_per_sec <= 0.0 {
        return 0.0;
    }
    let total_tokens =
        tokens_per_sec * SECONDS_PER_YEAR * BENCH_DEVICE_LIFETIME_YEARS * BENCH_DEVICE_DUTY_CYCLE;
    crate::constants::engine::BENCH_DEFAULT_DEVICE_COST_USD / total_tokens * 1_000_000.0
}

/// Render the full markdown report.
fn render_report(outcomes: &[EngineOutcome], args: &BenchArgs) -> Result<String> {
    let mut markdown = String::new();
    writeln!(markdown, "# openBatarangs experimental benchmark\n")?;
    writeln!(
        markdown,
        "- Device price: ${:.0}\n- Electricity: ${:.2}/kWh\n- Lifetime: {:.0} years @ {:.0}% duty cycle",
        args.device_cost_usd,
        args.electricity_usd_per_kwh,
        BENCH_DEVICE_LIFETIME_YEARS,
        BENCH_DEVICE_DUTY_CYCLE * 100.0
    )?;
    writeln!(
        markdown,
        "- Iterations: {}\n- Max tokens (Ollama): {}\n",
        args.iterations, args.max_tokens
    )?;

    for outcome in outcomes {
        match outcome {
            EngineOutcome::Skipped { kind, reason } => {
                writeln!(markdown, "## {} — skipped\n", kind.display_name())?;
                writeln!(markdown, "{reason}\n")?;
            }
            EngineOutcome::Benchmarked(report) => {
                writeln!(
                    markdown,
                    "## {} ({})\n",
                    report.kind.display_name(),
                    report.model
                )?;
                writeln!(markdown, "| Metric | Value |")?;
                writeln!(markdown, "| --- | --- |")?;
                writeln!(
                    markdown,
                    "| Tokens/sec | {:.2} |",
                    report.avg_tokens_per_sec
                )?;
                match report.avg_power_watts {
                    Some(watts) => writeln!(markdown, "| Avg power (W) | {watts:.2} |")?,
                    None => writeln!(markdown, "| Avg power (W) | n/a |")?,
                }
                match report.tokens_per_sec_per_watt {
                    Some(value) => writeln!(markdown, "| Tokens/sec/W | {value:.4} |")?,
                    None => writeln!(markdown, "| Tokens/sec/W | n/a |")?,
                }
                match report.usd_per_million_tokens {
                    Some(value) => writeln!(markdown, "| USD / 1M tokens | {value:.4} |")?,
                    None => writeln!(markdown, "| USD / 1M tokens | n/a |")?,
                }
                writeln!(markdown)?;
                writeln!(markdown, "Notes:")?;
                for sample in &report.samples {
                    writeln!(markdown, "- {}", sample.notes)?;
                }
                writeln!(markdown)?;
            }
        }
    }
    Ok(markdown)
}

/// Write the markdown report to a file.
fn write_report(path: &PathBuf, markdown: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::write(path, markdown)
        .with_context(|| format!("failed to write {}", path.display()))?;
    println!("📄 Report written to {}", path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_of_empty_is_zero() {
        assert_eq!(mean(std::iter::empty()), 0.0);
        assert_eq!(mean_opt(std::iter::empty()), None);
    }

    #[test]
    fn mean_averages_values() {
        assert_eq!(mean(vec![10.0, 20.0, 30.0].into_iter()), 20.0);
        assert_eq!(mean_opt(vec![1.0, 3.0].into_iter()), Some(2.0));
    }

    #[test]
    fn energy_cost_scales_with_throughput_and_power() {
        // 100 tok/s at 10 W -> 1M tokens takes 10 000 s = 0.02778 kWh.
        let cost = energy_cost_per_million(100.0, Some(10.0)).unwrap();
        assert!((cost - 0.027_777_8 * 0.15).abs() < 1e-6);
    }

    #[test]
    fn energy_cost_returns_none_for_zero_throughput() {
        assert_eq!(energy_cost_per_million(0.0, Some(10.0)), None);
        assert_eq!(energy_cost_per_million(100.0, None), None);
    }

    #[test]
    fn hardware_cost_is_positive_for_positive_throughput() {
        assert!(hardware_cost_per_million(100.0) > 0.0);
        assert_eq!(hardware_cost_per_million(0.0), 0.0);
    }

    #[test]
    fn render_report_skips_unavailable_engine() {
        let outcomes = vec![EngineOutcome::Skipped {
            kind: EngineKind::TensorRt,
            reason: "missing".to_string(),
        }];
        let args = BenchArgs {
            engines: vec![],
            model: None,
            prompt: None,
            max_tokens: 128,
            iterations: 1,
            seq_len: 128,
            avg_runs: 20,
            trt_shapes: None,
            output: None,
            device_cost_usd: 699.0,
            electricity_usd_per_kwh: 0.15,
        };
        let report = render_report(&outcomes, &args).unwrap();
        assert!(report.contains("skipped"));
        assert!(report.contains("missing"));
    }
}

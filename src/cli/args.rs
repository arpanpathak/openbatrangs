//! Clap argument definitions and default values.

use crate::constants::agent::MAX_CONTEXT_TOKENS;
use crate::constants::cli::{DEFAULT_MAX_STEPS, DEFAULT_MIN_CONTEXT, DEFAULT_OLLAMA_URL};
use crate::constants::engine::{
    BENCH_DEFAULT_DEVICE_COST_USD, BENCH_DEFAULT_ELECTRICITY_USD_PER_KWH, BENCH_DEFAULT_ITERATIONS,
    BENCH_DEFAULT_MAX_TOKENS,
};
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

/// Command-line interface definition.
#[derive(Parser)]
#[command(
    name = "openbatrangs",
    version,
    about = "🦇 openBatarangs — interactive agentic coding CLI for local models via Ollama"
)]
pub(crate) struct Cli {
    /// Task for the coding agent, e.g. "fix the Rust build errors".
    /// With no task, openBatarangs starts an interactive TUI.
    #[arg(value_name = "TASK")]
    pub(crate) task: Vec<String>,

    /// Ollama server URL.
    #[arg(long, global = true, default_value = DEFAULT_OLLAMA_URL)]
    pub(crate) ollama_url: String,

    /// Inference engine for the agent (experimental; `ollama` or `tensorrt`).
    #[arg(long, global = true, default_value = "ollama", value_name = "NAME")]
    pub(crate) engine: String,

    /// Model to use; auto-discovered when omitted.
    #[arg(short, long, global = true)]
    pub(crate) model: Option<String>,

    /// Working directory for the agent.
    #[arg(long, global = true, default_value = ".")]
    pub(crate) cwd: PathBuf,

    /// Maximum agent steps.
    #[arg(long, global = true, default_value_t = DEFAULT_MAX_STEPS)]
    pub(crate) max_steps: usize,

    /// Read-only mode: no file writes or shell commands.
    #[arg(long = "read-only", global = true)]
    pub(crate) is_read_only: bool,

    /// Ask before each file write or shell command.
    #[arg(long = "confirm", global = true)]
    pub(crate) should_confirm: bool,

    /// Do not auto-pull a recommended model when none is suitable.
    #[arg(long = "no-auto-pull", global = true)]
    pub(crate) is_auto_pull_disabled: bool,

    /// Minimum context window for auto model selection.
    #[arg(long, global = true, default_value_t = DEFAULT_MIN_CONTEXT)]
    pub(crate) min_context: usize,

    /// Maximum context window sent to the model (lower uses less memory).
    #[arg(long = "max-ctx", global = true, default_value_t = MAX_CONTEXT_TOKENS)]
    pub(crate) max_ctx: u64,

    #[command(subcommand)]
    pub(crate) command: Option<Commands>,
}

/// Available subcommands.
#[derive(Subcommand)]
pub(crate) enum Commands {
    /// Run the coding agent (same as passing a task directly).
    Agent {
        /// Task description.
        #[arg(value_name = "TASK")]
        task: Vec<String>,
    },
    /// List locally installed Ollama models with agent scores.
    ListModels,
    /// Check Ollama connectivity and recommend the best model.
    Doctor,
    /// Install/start Ollama and pull a recommended coding model.
    Setup,
    /// Pull a model tag from the Ollama registry.
    Pull {
        /// Model tag, e.g. `samantha-mistral:7b`.
        #[arg(value_name = "MODEL")]
        model: String,
    },
    /// Experimental inference engines and benchmarks.
    Experimental {
        #[command(subcommand)]
        command: ExperimentalCommand,
    },
}

/// Experimental subcommands.
#[derive(Subcommand)]
pub(crate) enum ExperimentalCommand {
    /// Probe which experimental engines are available on this machine.
    Doctor,
    /// Benchmark engines and print a performance-per-watt-per-dollar report.
    Bench(BenchArgs),
}

/// Arguments for `experimental bench`.
#[derive(Args, Clone)]
pub(crate) struct BenchArgs {
    /// Engines to benchmark; defaults to every available engine.
    #[arg(short, long, value_delimiter = ',', value_name = "ENGINE")]
    pub(crate) engines: Vec<String>,

    /// Model tag (Ollama) or ONNX model path (TensorRT).
    #[arg(long, value_name = "MODEL")]
    pub(crate) model: Option<String>,

    /// Prompt used for Ollama benchmarks.
    #[arg(long, value_name = "PROMPT")]
    pub(crate) prompt: Option<String>,

    /// Maximum tokens for Ollama generation.
    #[arg(long, default_value_t = BENCH_DEFAULT_MAX_TOKENS)]
    pub(crate) max_tokens: usize,

    /// Number of benchmark iterations per engine.
    #[arg(long, default_value_t = BENCH_DEFAULT_ITERATIONS)]
    pub(crate) iterations: usize,

    /// TensorRT input sequence length used for prefill tokens/sec.
    #[arg(long = "seq-len", default_value_t = crate::constants::engine::TRTEXEC_DEFAULT_SEQ_LEN)]
    pub(crate) seq_len: usize,

    /// TensorRT repetitions per trtexec run.
    #[arg(long = "avg-runs", default_value_t = crate::constants::engine::TRTEXEC_DEFAULT_AVG_RUNS)]
    pub(crate) avg_runs: usize,

    /// Optional trtexec --shapes string for dynamic ONNX inputs.
    #[arg(long = "trt-shapes", value_name = "SHAPES")]
    pub(crate) trt_shapes: Option<String>,

    /// Markdown report output path (empty writes to stdout only).
    #[arg(long, value_name = "PATH")]
    pub(crate) output: Option<PathBuf>,

    /// Device purchase price in USD for amortization math.
    #[arg(long, default_value_t = BENCH_DEFAULT_DEVICE_COST_USD)]
    pub(crate) device_cost_usd: f64,

    /// Electricity price in USD per kWh.
    #[arg(long, default_value_t = BENCH_DEFAULT_ELECTRICITY_USD_PER_KWH)]
    pub(crate) electricity_usd_per_kwh: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::cli::{DEFAULT_MAX_STEPS, DEFAULT_MIN_CONTEXT, DEFAULT_OLLAMA_URL};

    #[test]
    fn parses_defaults_without_arguments() {
        let cli = Cli::parse_from(["openbatrangs"]);
        assert!(cli.task.is_empty());
        assert_eq!(cli.ollama_url, DEFAULT_OLLAMA_URL);
        assert_eq!(cli.engine, "ollama");
        assert_eq!(cli.max_steps, DEFAULT_MAX_STEPS);
        assert_eq!(cli.min_context, DEFAULT_MIN_CONTEXT);
        assert_eq!(cli.max_ctx, MAX_CONTEXT_TOKENS);
        assert_eq!(cli.cwd, PathBuf::from("."));
        assert!(!cli.is_read_only);
        assert!(!cli.should_confirm);
        assert!(!cli.is_auto_pull_disabled);
        assert!(cli.command.is_none());
    }

    #[test]
    fn parses_positional_task() {
        let cli = Cli::parse_from(["openbatrangs", "fix", "the", "bug"]);
        assert_eq!(cli.task, vec!["fix", "the", "bug"]);
    }

    #[test]
    fn parses_global_options_before_subcommand() {
        let cli = Cli::parse_from([
            "openbatrangs",
            "--model",
            "qwen2.5-coder:7b",
            "--max-steps",
            "5",
            "--read-only",
            "list-models",
        ]);
        assert_eq!(cli.model.as_deref(), Some("qwen2.5-coder:7b"));
        assert_eq!(cli.max_steps, 5);
        assert!(cli.is_read_only);
        assert!(matches!(cli.command, Some(Commands::ListModels)));
    }

    #[test]
    fn parses_agent_subcommand_with_task() {
        let cli = Cli::parse_from(["openbatrangs", "agent", "write", "tests"]);
        match cli.command {
            Some(Commands::Agent { task }) => assert_eq!(task, vec!["write", "tests"]),
            _ => panic!("expected agent subcommand"),
        }
    }

    #[test]
    fn parses_setup_subcommand() {
        let cli = Cli::parse_from(["openbatrangs", "setup"]);
        assert!(matches!(cli.command, Some(Commands::Setup)));
    }

    #[test]
    fn parses_doctor_subcommand() {
        let cli = Cli::parse_from(["openbatrangs", "doctor"]);
        assert!(matches!(cli.command, Some(Commands::Doctor)));
    }

    #[test]
    fn parses_pull_subcommand_with_model() {
        let cli = Cli::parse_from(["openbatrangs", "pull", "samantha-mistral:7b"]);
        match cli.command {
            Some(Commands::Pull { model }) => assert_eq!(model, "samantha-mistral:7b"),
            _ => panic!("expected pull subcommand"),
        }
    }

    #[test]
    fn parses_engine_flag() {
        let cli = Cli::parse_from(["openbatrangs", "--engine", "tensorrt", "list-models"]);
        assert_eq!(cli.engine, "tensorrt");
    }

    #[test]
    fn parses_experimental_bench_subcommand() {
        let cli = Cli::parse_from([
            "openbatrangs",
            "experimental",
            "bench",
            "--engines",
            "ollama,tensorrt",
            "--model",
            "model.onnx",
            "--iterations",
            "2",
        ]);
        match cli.command {
            Some(Commands::Experimental { command }) => match command {
                ExperimentalCommand::Bench(args) => {
                    assert_eq!(args.engines, vec!["ollama", "tensorrt"]);
                    assert_eq!(args.model.as_deref(), Some("model.onnx"));
                    assert_eq!(args.iterations, 2);
                }
                _ => panic!("expected bench subcommand"),
            },
            _ => panic!("expected experimental subcommand"),
        }
    }
}

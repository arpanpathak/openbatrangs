//! Ollama process/polling constants used by command implementations.

/// How many times to poll Ollama while waiting for it to start.
pub const OLLAMA_START_POLL_ATTEMPTS: u32 = 20;

/// How many times to poll Ollama during `setup` while waiting for it to start.
pub const SETUP_START_POLL_ATTEMPTS: u32 = 40;

/// Delay between Ollama availability checks, in milliseconds.
pub const OLLAMA_POLL_INTERVAL_MILLIS: u64 = 500;

/// Executable name of the Ollama server.
pub const OLLAMA_BINARY: &str = "ollama";

/// Official Ollama Linux install script URL.
pub const OLLAMA_INSTALL_SCRIPT_URL: &str = "https://ollama.com/install.sh";

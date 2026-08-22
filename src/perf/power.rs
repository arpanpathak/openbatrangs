//! Average GPU/board power sampling for benchmarks.
//!
//! [`PowerSampler`] starts `tegrastats` for the duration of one benchmark run,
//! collects `VDD_IN` readings, and reports the average in watts.

use super::gpu::parse_tegrastats;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;

/// Collects `tegrastats` power samples while alive.
pub struct PowerSampler {
    samples: Arc<Mutex<Vec<f64>>>,
    child: Option<Child>,
}

impl PowerSampler {
    /// Start sampling power in the background.
    pub fn start() -> Self {
        let samples = Arc::new(Mutex::new(Vec::new()));
        let child = start_tegrastats_sampler(samples.clone());
        Self { samples, child }
    }

    /// Average board power draw in watts, or `None` when no samples arrived.
    pub fn average_watts(&self) -> Option<f64> {
        let samples = self.samples.lock().ok()?;
        if samples.is_empty() {
            return None;
        }
        Some(samples.iter().sum::<f64>() / samples.len() as f64)
    }
}

impl Drop for PowerSampler {
    fn drop(&mut self) {
        if let Some(child) = &mut self.child {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Spawn `tegrastats` and push parsed `VDD_IN` watts into `samples`.
fn start_tegrastats_sampler(samples: Arc<Mutex<Vec<f64>>>) -> Option<Child> {
    let mut child = Command::new("tegrastats")
        .arg("--interval")
        .arg("500")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let stdout = child.stdout.take()?;
    thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines().map_while(Result::ok) {
            if let Some(stats) = parse_tegrastats(&line) {
                if let Some(watts) = stats.power_watts {
                    if let Ok(mut guard) = samples.lock() {
                        guard.push(watts);
                    }
                }
            }
        }
    });
    Some(child)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampler_with_no_samples_returns_none() {
        let sampler = PowerSampler {
            samples: Arc::new(Mutex::new(Vec::new())),
            child: None,
        };
        assert!(sampler.average_watts().is_none());
    }

    #[test]
    fn sampler_averages_samples() {
        let samples = Arc::new(Mutex::new(vec![10.0, 20.0, 30.0]));
        let sampler = PowerSampler {
            samples,
            child: None,
        };
        assert_eq!(sampler.average_watts(), Some(20.0));
    }
}

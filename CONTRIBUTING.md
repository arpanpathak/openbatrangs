# Contributing to openBatarangs

Thanks for wanting to help! PRs are welcome.

## Development setup

```sh
git clone https://github.com/arpanpathak/openbatrangs.git
cd openbatrangs
cargo build
cargo test
```

## What's useful

- Bug reports with the exact command you ran and the error output.
- Improvements to model auto-discovery/scoring.
- New agent tools or safety improvements.
- Better context/token budgeting for long agent sessions.
- Packaging help for prebuilt binaries and crates.io publishing.

## Guidelines

- Keep the CLI dependency-light and easy to build on Jetson-class hardware.
- Tools must stay safe by default: relative paths only, `--read-only` disables writes/commands, `--confirm` asks before mutating.
- Run `cargo fmt` and `cargo clippy` before submitting if possible.
- Add or update tests when changing core behavior.

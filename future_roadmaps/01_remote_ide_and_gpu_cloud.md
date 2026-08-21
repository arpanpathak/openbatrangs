# 01 - Remote IDE and GPU Cloud

**Status:** idea / draft

## Vision

Turn a low-power Jetson-class device into a personal AI coding cloud that people
can use from anywhere. No home-router port forwarding, no public IP required —
just tunnels, zero-trust auth, and an IDE in the browser.

## Why this attracts people

- "Run real local AI on my own hardware" is a compelling story.
- Remote access is the missing piece between "demo on my desk" and "share with friends".
- GPU/CUDA workloads on a small device are rare and interesting.
- It demonstrates practical edge-AI infrastructure, not just a toy CLI.

## Cool drafts

### 1. One-command remote setup

```bash
openbatrangs remote enable
```

Interactive wizard that:

- Detects the device (Jetson, x86, Apple Silicon)
- Offers Cloudflare Tunnel, Tailscale, or reverse SSH
- Sets up `code-server` / VS Code Web / Jupyter
- Prints a shareable URL or tailnet invite

### 2. Zero-trust access by default

- Cloudflare Access in front of the web IDE
- Allowlist by email / GitHub / Google identity
- Tailscale tailnet for trusted SSH access
- No password auth ever; SSH keys or short-lived certificates only

### 3. Shared GPU workspace

- Multi-user workspaces with per-user resource quotas (CPU, RAM, GPU)
- CUDA job queue: `openbatrangs job submit` with status via web
- Live `tegrastats`-style perf panel embedded in the web IDE
- Time-boxed sessions so one runaway job cannot kill the device

### 4. openBatarangs as a remote service

- Run the TUI inside a web terminal (ttyd / tmux + web shell)
- Share an agent session with read-only viewers
- Let collaborators request agent actions with approval workflow

## Roadmap sketch

| Phase | Deliverable |
|---|---|
| 1 | `remote enable` for Cloudflare Tunnel + code-server |
| 2 | Tailscale SSH setup and `openbatrangs remote status` |
| 3 | Per-user quotas and GPU job queue |
| 4 | Web terminal for openBatarangs + shareable sessions |
| 5 | Multi-user approval flow and audit logs |

## Open questions

- Should remote access be a paid cloud service or a self-hosted script?
- How to handle CUDA job isolation safely on a single GPU?
- Web terminal vs native IDE: which first?

# 04 - Security and Sandboxing

**Status:** idea / draft

## Vision

Make openBatarangs safe enough to share: sandboxed tools, resource limits,
multi-user isolation, and auditable agent actions. Security becomes a feature,
not a limitation.

## Why this attracts people

- "Can I safely let an AI agent touch my files and run commands?" is the first question.
- Multi-user support unlocks classrooms, labs, and community devices.
- Good security design is a differentiator that serious users care about.

## Cool drafts

### 1. Tool permission profiles

- Profiles: `read-only`, `ask`, `auto`, `sandbox`
- Per-tool permissions instead of global allow/deny
- `openbatrangs permissions edit` to review what the agent can do

### 2. Sandboxed command execution

- Optional container or `bubblewrap` execution for `run_command`
- Network blocked by default inside the sandbox
- Filesystem writes confined to the workspace

### 3. Resource budgets

- Per-task CPU/RAM/GPU limits via cgroups or systemd scopes
- Timeout hierarchy: command < step < task
- Watchdog that kills runaway jobs and reports the reason

### 4. Audit log

- Every tool call, file write, and command recorded in `~/.openbatrangs/audit.log`
- TUI command: `openbatrangs audit --last 20`
- Optional export to JSON for external SIEM

### 5. Multi-user device mode

- Each user gets a separate workspace, model quota, and audit trail
- Admin can revoke access instantly
- Remote access (see draft 01) uses zero-trust identity instead of shared passwords

## Roadmap sketch

| Phase | Deliverable |
|---|---|
| 1 | Permission profiles per tool |
| 2 | Audit log + `openbatrangs audit` |
| 3 | Resource budgets via systemd/cgroups |
| 4 | Sandboxed command execution |
| 5 | Multi-user device mode |

## Open questions

- Which sandbox backend should be default (bubblewrap vs Docker vs none)?
- Should audit logs be signed to prevent tampering?
- How do permissions interact with plugins from draft 02?

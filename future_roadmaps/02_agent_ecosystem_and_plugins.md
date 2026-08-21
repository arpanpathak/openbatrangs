# 02 - Agent Ecosystem and Plugins

**Status:** idea / draft

## Vision

Make openBatarangs an extensible agent platform. Users can add tools, connect
external services, compose agent teams, and share their setups like apps.

## Why this attracts people

- Plugins and marketplaces create a network effect: more plugins -> more users.
- MCP (Model Context Protocol) is trending and opens integration with a huge ecosystem.
- Multi-agent workflows are a headline feature that generates buzz.
- It turns a single CLI into a platform people can build on.

## Cool drafts

### 1. MCP support

- Native MCP client: connect the agent to GitHub, databases, browsers, file sync
- One-command registration: `openbatrangs mcp add <command>`
- Allowlist and permissions per MCP server

### 2. Plugin system

```bash
openbatrangs plugin install gh-issues
openbatrangs plugin create my-tool
```

- Plugin manifest with name, description, permissions, and version
- Sandboxed tools with explicit user approval
- Community plugin registry (GitHub-based, no central server required)

### 3. Agent teams

- `openbatrangs team create "reviewer + writer + tester"`
- Each agent gets a role, a different model, and a shared workspace
- Team leader aggregates results and resolves conflicts

### 4. Project memory and context packs

- Persistent project memory across sessions (files, decisions, TODO)
- Context packs: language-specific, framework-specific, or hardware-specific
- Auto-generated `CONTEXT.md` that shrinks with the project

## Roadmap sketch

| Phase | Deliverable |
|---|---|
| 1 | MCP client with allowlist permissions |
| 2 | Plugin manifest + `plugin install/create` |
| 3 | Agent teams with role definitions |
| 4 | Community registry and `openbatrangs plugin search` |
| 5 | Project memory and context packs |

## Open questions

- Which MCP servers should ship by default?
- How strict should sandboxing be for third-party plugins?
- Should teams use the same model or support heterogeneous models?

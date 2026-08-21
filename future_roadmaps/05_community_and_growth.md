# 05 - Community and Growth

**Status:** idea / draft

## Vision

Build a community around local, edge-native AI coding. Give people reasons to
contribute, compete, and show off what their low-power devices can do.

## Why this attracts people

- Leaderboards and benchmarks create friendly competition.
- "Runs on a $99 Jetson" is a strong contrast to cloud-only AI tools.
- Easy contribution paths lower the barrier for new developers.
- Visual, shareable artifacts (reports, badges, recordings) spread organically.

## Cool drafts

### 1. Model leaderboard

- Benchmark openBatarangs on common tasks with different local models
- Categories: code generation, refactoring, bug fixing, agent tool-use
- Hardware categories: 8 GB, 16 GB, 32 GB unified memory
- Auto-generated badges: "Best 3B model on 16 GB Jetson"

### 2. Recipe / template gallery

- `openbatrangs recipe apply jetson-cuda`
- Recipes combine model, context packs, plugins, and permissions
- Community can submit recipes as PRs to a `recipes/` directory

### 3. Contribution ladders

| Level | How to join |
|---|---|
| User | Run the CLI, report issues, share recordings |
| Contributor | Fix bugs, add tools, write docs |
| Recipe author | Submit model packs and project templates |
| Maintainer | Own a module, review PRs, shape roadmap |

### 4. Showcase page

- Weekly "built with openBatarangs" highlights
- Terminal recordings as shareable GIFs/WebMs
- Before/after refactor showcases (the codebase itself is a good example)

### 5. Hackathon / challenge packs

- "24-hour edge AI coding challenge" with prebuilt Docker images
- Datasets and evaluation scripts included
- Winners get their model/recipe featured in the repo

## Roadmap sketch

| Phase | Deliverable |
|---|---|
| 1 | Benchmark script + model leaderboard table |
| 2 | Recipe format and `recipe apply` |
| 3 | CONTRIBUTING ladders and issue templates |
| 4 | Showcase page with recordings |
| 5 | Challenge packs and community events |

## Open questions

- Should benchmarks run in CI on real Jetson hardware or emulated?
- Where should the leaderboard live: repo, GitHub Pages, or a small site?
- How do we keep community recipes safe to install?

# Vibe Orchestrator - Project Starmap

## Overview

Vibe is a Python CLI tool that uses GLM-4.7 as the brain/project manager and Claude Code as the worker. User talks to GLM, GLM delegates tasks to Claude, reviews output, and maintains project state.

```
User <-> GLM (brain) <-> Claude (worker)
              |
              v
        Memory-keeper + Starmap + Changelog
```

## Architecture

### Core Flow

1. **Startup**: Validate all systems, show project list, load context
2. **Conversation**: User talks to GLM
3. **Task Delegation**: GLM creates atomic tasks for Claude
4. **Execution**: Claude runs task via subprocess
5. **Review Gate**: GLM reviews Claude's output
6. **Accept/Reject**: Update project state or send feedback (max 3 retries)
7. **Loop**: Continue until goal complete

## Directory Structure

```
/home/brian/vibe/
├── vibe/
│   ├── __init__.py
│   ├── cli.py             # Entry point, rich UI, commands
│   ├── config.py          # Settings, projects.json loading
│   ├── exceptions.py      # Exception hierarchy
│   ├── state.py           # Session state machine
│   │
│   ├── glm/
│   │   ├── __init__.py
│   │   ├── client.py      # OpenRouter API wrapper for GLM-4.7
│   │   ├── prompts.py     # System prompts (Supervisor, Reviewer)
│   │   └── parser.py      # Parse GLM JSON responses
│   │
│   ├── claude/
│   │   ├── __init__.py
│   │   ├── executor.py    # Claude CLI subprocess with streaming
│   │   └── circuit.py     # Circuit breaker for failures
│   │
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── supervisor.py  # CORE: Main orchestration loop
│   │   ├── reviewer.py    # GLM review gate with retry logic
│   │   ├── task_queue.py  # Async task management
│   │   ├── task_enforcer.py # Tool requirements per task type
│   │   └── project_updater.py # Auto-update STARMAP/CHANGELOG
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   └── keeper.py      # Direct SQLite access to memory-keeper
│   │
│   └── integrations/
│       ├── __init__.py
│       ├── perplexity.py  # Research API client
│       └── github.py      # GitHub CLI wrapper
│
├── tests/
├── docs/
│   └── MCP_SERVERS.md     # Recommended MCP servers
├── CHANGELOG.md
├── STARMAP.md             # This file
├── pyproject.toml
└── README.md
```

## Key Components

### Supervisor (`orchestrator/supervisor.py`)
The CORE of Vibe. Coordinates the full GLM → Claude → GLM loop:
- Loads project context (STARMAP, CLAUDE.md, memory)
- Asks GLM to decompose user requests into atomic tasks
- Executes each task via Claude subprocess
- Sends Claude's output to GLM for review
- Handles retries with feedback injection (max 3 attempts)
- Persists results to memory-keeper

### Reviewer (`orchestrator/reviewer.py`)
GLM-powered code review gate:
- Evaluates task completion, code quality, scope adherence
- Tracks attempts per task for retry logic
- Builds retry context with previous rejection feedback
- Auto-approves if review crashes to avoid losing work

### ClaudeExecutor (`claude/executor.py`)
Subprocess integration with Claude Code CLI:
- Streaming JSON output parsing
- Tool call tracking (Read, Write, Edit, Bash, Grep, Glob)
- File change detection from Edit/Write tools
- Timeout protection with configurable tiers
- Clean environment (removes API keys from subprocess)

### GLMClient (`glm/client.py`)
OpenRouter API wrapper for GLM-4.7:
- Task decomposition into atomic tasks
- Code review with structured JSON output
- Clarification question detection
- Streaming responses for real-time display

### VibeMemory (`memory/keeper.py`)
Direct SQLite access to memory-keeper database:
- Session management with project isolation
- Context persistence (decisions, tasks, progress)
- Checkpoint creation before risky operations
- Convention storage for cross-project rules

## CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/status` | Show session status |
| `/usage` | Show GLM token usage |
| `/memory` | Show memory statistics |
| `/research <query>` | Research via Perplexity |
| `/github` | Show GitHub repo info |
| `/issues` | List open issues |
| `/prs` | List open pull requests |
| `/convention` | Manage global conventions |
| `/debug start` | Start debug session |
| `/quit` | Exit gracefully |

## Configuration

Projects are registered in `~/.config/vibe/projects.json`:
```json
{
  "projects": [
    {
      "name": "athena",
      "path": "/home/brian/athena/v2/",
      "description": "Personal AI Research Hub",
      "starmap": "STARMAP.md",
      "claude_md": "CLAUDE.md",
      "test_command": "pytest -v"
    }
  ]
}
```

## Recent Changes

- **2026-01-11**: Core implementation complete (Supervisor, Reviewer)
- **2026-01-11**: Robustness fixes (context overflow, state bleed, review fallback)
- **2026-01-11**: Task enforcer, conventions, debug sessions
- **2026-01-10**: Foundation, GLM client, Claude executor

## GitHub

Repository: https://github.com/SEO-Geek/vibe-orchestrator

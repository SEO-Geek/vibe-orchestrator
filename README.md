# Vibe Orchestrator

GLM-4.7 as brain, Claude Code as worker - AI pair programming orchestrator.

## Overview

Vibe is a CLI tool that uses GLM-4.7 (via OpenRouter) as a project manager to supervise Claude Code. User talks to GLM, GLM decomposes tasks, delegates to Claude, reviews output, and maintains project state.

```
User <-> GLM (brain) <-> Claude (worker)
              |
              v
        Memory-keeper + Starmap + Changelog
```

## Installation

```bash
# Clone and install
cd /home/brian/vibe
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Create symlink for easy access
ln -sf $(pwd)/.venv/bin/vibe ~/.local/bin/vibe
```

## Usage

```bash
# Start Vibe
vibe

# Add a new project
vibe add myproject /path/to/project --description "My project"

# List projects
vibe list

# Remove a project
vibe remove myproject
```

## Requirements

- Python 3.11+
- Claude Code CLI installed
- OpenRouter API key (OPENROUTER_API_KEY)
- Memory-keeper MCP running

## Configuration

Projects are stored in `~/.config/vibe/projects.json`:

```json
{
  "projects": [
    {
      "name": "myproject",
      "path": "/path/to/project",
      "starmap": "STARMAP.md",
      "claude_md": "CLAUDE.md",
      "test_command": "pytest -v"
    }
  ]
}
```

## Architecture

- **GLM Client**: OpenRouter API wrapper for GLM-4.7
- **Claude Executor**: Subprocess management with circuit breaker
- **Supervisor**: Orchestrates user -> GLM -> Claude -> GLM review flow
- **Memory**: Direct SQLite access to memory-keeper database
- **State Machine**: Tracks session state (IDLE, EXECUTING, REVIEWING, etc.)

## License

MIT

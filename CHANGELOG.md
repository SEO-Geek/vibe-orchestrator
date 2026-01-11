# Changelog

All notable changes to Vibe Orchestrator will be documented in this file.

## [Unreleased]

### Added

#### Phase 4: Claude Code Executor (2026-01-11)

- **Claude CLI Integration**: Full subprocess integration with Claude Code CLI
  - Uses `--output-format stream-json --verbose` for structured output parsing
  - Tracks tool calls (Read, Write, Edit, Bash, Grep, Glob) in real-time
  - Monitors file modifications from Edit/Write tools
  - Timeout tiers: quick (30s), code (120s), debug (180s), research (300s)

- **Review Gate**: GLM reviews Claude's changes before acceptance
  - Gets git diff of modified files
  - Evaluates against task requirements
  - Approves or rejects with feedback
  - Displays clear APPROVED/REJECTED status

- **Task Execution Flow**:
  - User confirmation before executing task list
  - Progress display with tool call notifications
  - Cost and duration tracking per task
  - Session summary with completed/failed counts

- **Security Improvements**:
  - Environment cleaning with pattern-based filtering
  - Removes API keys, tokens, secrets from subprocess environment
  - Proper subprocess cleanup on timeout and exceptions

- **State Machine Updates**:
  - Added `add_error()` for error tracking without current task
  - Added `add_completed_task()` for simple task recording
  - Full state transitions: IDLE → EXECUTING → REVIEWING → IDLE

#### Phase 2-3: GLM Client & Conversation Loop (2026-01-10)

- **GLM Client**: OpenRouter API wrapper for GLM-4.7
  - Async chat completions with streaming support
  - Task decomposition into atomic tasks
  - Clarification question detection
  - Code review evaluation

- **Conversation Loop**: Interactive CLI with Rich UI
  - GLM-driven conversation with context awareness
  - Project context loading (STARMAP.md, CLAUDE.md)
  - Commands: /help, /status, /usage, /quit

#### Phase 1: Foundation (2026-01-10)

- **Project Structure**: Clean modular architecture
  - `vibe/glm/` - GLM client, prompts, parser
  - `vibe/claude/` - Executor, circuit breaker
  - `vibe/orchestrator/` - Supervisor, task queue, reviewer
  - `vibe/memory/` - Memory-keeper integration

- **Exception Hierarchy**: Structured error handling
  - VibeError base class with context
  - Specialized: ConfigError, GLMError, ClaudeError, VibeMemoryError

- **Configuration**: JSON-based project registry
  - `~/.config/vibe/projects.json` for project list
  - Project model with path, description, starmap

- **Startup Validation**: System checks on launch
  - OpenRouter API ping
  - Claude Code CLI detection
  - Memory-keeper database check
  - GitHub CLI authentication

## [0.1.0] - 2026-01-10

- Initial project structure
- GitHub repository: https://github.com/SEO-Geek/vibe-orchestrator

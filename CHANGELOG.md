# Changelog

All notable changes to Vibe Orchestrator will be documented in this file.

## [Unreleased]

### Added

#### Phase 7: Project Updates (2026-01-11)

- **ProjectUpdater Class**: Automatic documentation updates
  - STARMAP.md updates when new files are created
  - CHANGELOG.md entries generated after task completion
  - Uses GLM for intelligent update text (with fallback)
  - Git-aware file change detection (created/modified/deleted)

- **Auto-Update Integration**:
  - Triggers after successful task execution
  - Collects all file changes across tasks
  - Updates docs only when files were actually changed
  - Shows update status in session summary

- **Changelog Features**:
  - Proper [Unreleased] section management
  - Category inference (Added/Changed/Fixed/Removed)
  - Maintains existing changelog format
  - Creates new changelog if none exists

- **STARMAP Features**:
  - Adds new files to "Recent Files Added" section
  - GLM-powered section updates (optional)
  - Preserves existing format and style

#### Phase 6: Memory Integration (2026-01-11)

- **VibeMemory Class**: Direct SQLite access to memory-keeper database
  - Session start/end tracking with project isolation (channel = project_name)
  - Context items: save, load, search with category/priority filters
  - Decisions and task results persistence
  - Journal entries with mood and tags

- **Checkpoint System**: Git-aware checkpoints before risky operations
  - Captures git branch and status
  - Links context items to checkpoint for recovery
  - Created automatically before task execution

- **CLI Memory Integration**:
  - Session starts when project is loaded
  - Memory items count shown on project load
  - `/memory` command shows stats (total items, categories, sessions)
  - Task results saved after each execution
  - Session ends gracefully on `/quit` or signal

- **Memory Features**:
  - `list_recent_sessions()` - View past sessions for project
  - `continue_session()` - Resume existing session
  - `search()` - Find items by text and category
  - `save_decision()` - Record decisions with reasoning
  - `save_task_result()` - Track task completion
  - `get_stats()` - Memory statistics per project

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

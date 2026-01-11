# Changelog

All notable changes to Vibe Orchestrator will be documented in this file.

## [Unreleased]

### Added

#### Phase 8: Polish (2026-01-11)

- **Task Enforcer System**:
  - Automatic task type detection from description
  - Required/preferred/forbidden tools per task type
  - Completion checklists generated for each task type
  - Tool usage verification after Claude execution
  - Prevents Claude from using curl for browser testing

- **Global Conventions**:
  - Cross-project "we always do X" rules stored in memory-keeper
  - `/convention` command to add/list/delete conventions
  - Loaded automatically for every Claude task
  - Enforced alongside task-specific tool requirements

- **Perplexity Research Integration**:
  - PerplexityClient using OpenAI-compatible API
  - `/research` command for technical documentation lookup
  - Research for tasks, error lookup, library documentation
  - Citations extracted from responses

- **GitHub Operations**:
  - GitHubOps class using `gh` CLI for all operations
  - Issue management: list, get, create, close
  - Pull request operations: list, get, create, merge
  - Branch creation and repo sync
  - `/github`, `/issues`, `/prs` commands in CLI

- **MCP Server Installations** (13 total):
  - Installed `sequential-thinking` for step-by-step reasoning
  - Installed `puppeteer` as backup browser automation
  - Installed `fetch` (uvx) for HTTP API testing
  - Installed `sqlite` (uvx) for database operations
  - Installed `uv` package manager for Python MCP servers
  - All servers verified working via `claude mcp list`

- **MCP Server Documentation**:
  - Recommended MCP servers for vibe coding (docs/MCP_SERVERS.md)
  - Browser automation, code analysis, Docker, database tools
  - Tool requirements mapped to task types

- **Debug Session Tracking System**:
  - `DebugSession` class tracks debugging attempts with pass/fail and reasoning
  - `DebugAttempt` dataclass for individual fix attempts with hypotheses
  - `AttemptResult` enum: PENDING, SUCCESS, PARTIAL, FAILED, MADE_WORSE
  - Feature preservation checklist (must_preserve list)
  - Git checkpoint creation before each attempt for rollback
  - Context injection into Claude prompts to prevent repeated failures
  - Prevents Claude from losing context on what was already tried
  - Prevents "bush fixes" that break existing functionality

- **CLI Commands**:
  - `/research <query>` - Research via Perplexity API
  - `/github` - Show GitHub repo info
  - `/issues` - List open issues
  - `/prs` - List open pull requests
  - `/convention` - Manage global conventions
  - `/debug start <problem>` - Start a debug session
  - `/debug preserve <feature>` - Add feature that must not break
  - `/debug hypothesis <text>` - Set current working hypothesis
  - `/debug attempt <description>` - Start a new fix attempt
  - `/debug fail <reason>` - Mark attempt as failed
  - `/debug partial <result>` - Mark attempt as partially worked
  - `/debug success` - Mark attempt as successful
  - `/debug status` - Show debug session status
  - `/debug context` - Show what gets injected into Claude
  - `/debug end` - End the debug session
  - `/rollback <attempt_id>` - Rollback to before a specific attempt
  - `/rollback start` - Rollback to session start state

- **New Exceptions**:
  - `ResearchError` for Perplexity API errors
  - `GitHubError` for GitHub CLI errors

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

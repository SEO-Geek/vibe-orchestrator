# Changelog

All notable changes to Vibe Orchestrator will be documented in this file.

## [Unreleased]

### Fixed

#### Rock-Solid Clarification System (2026-01-11)

**Problem**: GLM asked 3+ clarification questions instead of delegating to Claude for investigation tasks, causing frustrating delays.

**Solution**: Multi-layered defense system to ensure fast delegation:

1. **Keyword Detection** (`INVESTIGATION_KEYWORDS`):
   - Instant delegation for: check, debug, investigate, find, review, analyze, test, verify, examine, diagnose, troubleshoot, explore, trace, profile, audit, scan, monitor
   - NO API call needed - detects in code before asking GLM
   - Helper function: `is_investigation_request(text)`

2. **Hard Clarification Limit**:
   - Maximum 1 clarification question per request
   - `clarification_count` tracked in `SessionContext`
   - After 1 question, force delegation regardless of GLM response

3. **API Timeout with Auto-Delegation**:
   - 30-second timeout on GLM API calls
   - On timeout → delegate to Claude (fail-safe)
   - Never hang waiting for slow API

4. **Retry Logic with Exponential Backoff**:
   - Max 2 retries (3 total attempts)
   - Delays: 1s, 3s between retries
   - On all failures → delegate to Claude

5. **Circuit Breaker**:
   - Opens after 3 consecutive GLM failures
   - Skips GLM entirely for 60 seconds
   - Auto-resets after cooldown period
   - Prevents cascading failures

6. **Improved SUPERVISOR_SYSTEM_PROMPT**:
   - "CRITICAL: Delegate First, Ask Questions Sparingly" section
   - Explicit rules: only ask when user must make a DECISION
   - Never ask about file locations, errors, state - Claude can find these

7. **Dynamic Timeout Tiers for Claude Tasks**:
   - Investigation tasks auto-detect and use "research" tier (300s)
   - Normal coding tasks use "code" tier (120s)
   - `timeout_tier` parameter added to `executor.execute()`
   - Task panel shows tier label for research tasks
   - Keywords detected: investigate, review, analyze, examine, etc.

**Files Changed**:
- `vibe/glm/client.py` - Keyword detection, timeout, retry, circuit breaker, `is_investigation_request()`
- `vibe/glm/prompts.py` - Updated SUPERVISOR_SYSTEM_PROMPT
- `vibe/state.py` - Added `clarification_count` field
- `vibe/cli.py` - Track clarification count, dynamic timeout tier per task
- `vibe/claude/executor.py` - Added `timeout_tier` parameter to `execute()`

#### Task History Context for GLM (2026-01-11)

**Problem**: GLM had "ZERO clue" about previous tasks when user said "redo the tasks". Memory system was not passing task history to GLM context.

**Solution**: Connect memory system to GLM context loading:

1. **`load_project_context()` now accepts memory parameter**:
   - Loads recent task results (key: `task-*`, category: `task`)
   - Loads recent user requests (key: `request-*`)
   - Formats as "Recent Tasks Executed" section in context
   - Includes hint: "(User may refer to these tasks with 'redo', 'retry', or 'the tasks')"

2. **`process_user_request()` passes memory to context loader**:
   - `load_project_context(project, memory=memory)` instead of `load_project_context(project)`
   - Memory is already available in the function scope

3. **Updated TASK_DECOMPOSITION_PROMPT**:
   - Explicitly tells GLM to check "Recent Tasks Executed" section
   - Instructions to use history when user says "redo", "retry", "the tasks"

**Files Changed**:
- `vibe/cli.py` - `load_project_context()` accepts memory, loads task history, type hint fixed
- `vibe/glm/prompts.py` - TASK_DECOMPOSITION_PROMPT updated with history instructions

#### TaskHistory Refactor - Bulletproof In-Memory Tracking (2026-01-11)

**Problem**: Memory integration was fragile and complex. Too many failure points.

**Solution**: New `TaskHistory` class with in-memory storage + optional database backup:

1. **`vibe/memory/task_history.py`** - New module:
   - `TaskRecord` and `RequestRecord` dataclasses
   - `TaskHistory` class with class-level storage (survives across calls)
   - `add_task()`, `add_request()` - Always work, no exceptions
   - `get_context_for_glm()` - Formatted history for GLM
   - `load_from_memory()` - Load from database at startup
   - `get_stats()` - Task statistics
   - History trimmed to max 50 items (bounded memory)

2. **`vibe/cli.py`** refactored:
   - `load_project_context()` simplified - uses `get_context_for_glm()`
   - Task completion calls `add_task()` (in-memory) + `save_task_result()` (database)
   - Request handling calls `add_request()` at start
   - `TaskHistory.load_from_memory(_memory)` at session start
   - `/history` command shows task history and GLM context
   - `exit`, `quit`, `q` work without slash (like Claude Code)

3. **Dual-write pattern for reliability**:
   - TaskHistory (in-memory) - ALWAYS works
   - VibeMemory (database) - Backup, may fail silently

**Files Changed**:
- `vibe/memory/task_history.py` - NEW: In-memory task history
- `vibe/cli.py` - Refactored to use TaskHistory, added /history command, exit without slash

### Added

#### Core Implementation Complete (2026-01-11)

- **Supervisor Class** (`vibe/orchestrator/supervisor.py`):
  - Full orchestration loop: GLM → Claude → GLM review
  - Task decomposition via GLM
  - Retry logic with feedback injection (max 3 attempts)
  - Project context loading (STARMAP.md, CLAUDE.md, memory)
  - Checkpoint creation before risky operations
  - Cost tracking across all tasks

- **Reviewer Class** (`vibe/orchestrator/reviewer.py`):
  - GLM-powered code review gate
  - ReviewResult dataclass with approval status, issues, feedback
  - Attempt tracking per task
  - Retry context builder for failed attempts
  - Statistics tracking (approved/rejected counts)

### Fixed

#### Robustness Fixes (2026-01-11)

- **Context Overflow**: Truncate retry feedback to 500 chars max
- **State Bleed**: Use `copy.deepcopy()` for task dicts to prevent mutable constraint sharing
- **Review Crash Fallback**: Auto-approve if GLM review fails to avoid losing Claude's work
- **Decomposition Validation**: Reject empty task lists and empty descriptions in parser
- **Empty Feedback Handling**: Provide meaningful default when GLM returns no feedback
- **Timeout Protection**: Verified in executor with `asyncio.wait_for`

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

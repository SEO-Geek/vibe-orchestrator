# Vibe Orchestrator - Project Starmap

## Overview

Vibe is a Python CLI tool that orchestrates AI models to accomplish software engineering tasks.

**Architecture:**
- **Gemini (Brain/Orchestrator)**: Understands user intent, decomposes into atomic tasks
- **Claude (Worker)**: Executes tasks via subprocess
- **GLM (Code Reviewer)**: Reviews Claude's output for quality and correctness

```
User → Gemini (brain/orchestrator) → Claude (worker)
                  ↓                        ↓
              GLM (code review/verification)
                  ↓
        Memory-keeper + Starmap + Changelog
```

## Architecture

### Core Flow

1. **Startup**: Validate all systems (Gemini, GLM, Claude), show project list, load context
2. **Conversation**: User provides request
3. **Clarification** (if needed): Gemini checks if clarification is required
4. **Task Decomposition**: Gemini breaks request into atomic tasks
5. **Execution**: Claude runs each task via subprocess
6. **Review Gate**: GLM reviews Claude's output for quality/correctness
7. **Accept/Reject**: Update project state or send feedback (max 3 retries)
8. **Loop**: Continue until all tasks complete

## Directory Structure

```
/home/brian/vibe/
├── bin/
│   └── vibe-split           # tmux launcher (vibe left, claude right)
├── vibe/
│   ├── __init__.py
│   ├── cli.py             # Backward-compatibility shim (imports from cli/)
│   ├── cli/               # Refactored CLI (8 focused modules)
│   │   ├── __init__.py    # Exports all CLI components
│   │   ├── startup.py     # Startup validation and system checks
│   │   ├── project.py     # Project selection and context loading
│   │   ├── debug.py       # Debug workflow with GLM/Claude
│   │   ├── execution.py   # Task execution and GLM review
│   │   ├── commands.py    # Slash command handlers (/help, /debug, etc.)
│   │   ├── interactive.py # Main conversation loop
│   │   ├── typer_commands.py # CLI entry points (add, remove, list)
│   │   └── prompt.py      # Enhanced prompt with history/completion
│   ├── config.py          # Settings, projects.json loading, hook configs
│   ├── pricing.py         # Cost calculation for GLM/Claude API calls
│   ├── exceptions.py      # Exception hierarchy
│   ├── state.py           # Session state machine
│   │
│   ├── gemini/           # Brain/Orchestrator (task decomposition)
│   │   ├── __init__.py
│   │   ├── client.py      # OpenRouter API wrapper for Gemini 2.0 Flash
│   │   └── prompts.py     # Orchestrator system prompts
│   │
│   ├── glm/              # Code Reviewer (quality verification)
│   │   ├── __init__.py
│   │   ├── client.py      # OpenRouter API wrapper for GLM-4.7
│   │   ├── prompts.py     # Code review prompts only
│   │   ├── parser.py      # Parse GLM JSON responses
│   │   └── debug_state.py # Debug session state tracking
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
│   │   ├── task_enforcer.py # Tool requirements + SmartTaskDetector
│   │   ├── task_routing.py # Task-type-aware routing configuration
│   │   ├── task_board.py  # Kanban-style task tracking
│   │   ├── project_updater.py # Auto-update STARMAP/CHANGELOG
│   │   └── workflows/      # Intelligent task orchestration
│   │       ├── __init__.py
│   │       ├── templates.py  # WorkflowPhase, WorkflowTemplate
│   │       ├── injector.py   # SubTaskInjector, InjectionRule
│   │       └── engine.py     # WorkflowEngine
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── keeper.py      # Direct SQLite access to memory-keeper
│   │   ├── compaction.py  # GLM-powered context summarization
│   │   ├── task_history.py # In-memory task tracking
│   │   ├── debug_session.py # Debug session management
│   │   └── pattern_learning.py # Task pattern recording & learning
│   │
│   ├── persistence/        # Unified persistence layer
│   │   ├── __init__.py
│   │   ├── schema.sql     # SQLite schema (18 tables)
│   │   ├── models.py      # Dataclasses for all entities
│   │   └── repository.py  # VibeRepository database access
│   │
│   ├── logging/           # Structured logging system
│   │   ├── __init__.py
│   │   ├── config.py      # LogConfig with env overrides
│   │   ├── entries.py     # Log entry types (Gemini, GLM, Claude, session)
│   │   ├── handlers.py    # File handlers with rotation
│   │   └── viewer.py      # Log viewing and analysis
│   │
│   └── integrations/
│       ├── __init__.py
│       ├── research.py    # Perplexity research API client
│       └── github_ops.py  # GitHub CLI wrapper
│
├── tests/                 # 141 tests (unit + E2E integration)
│   ├── test_state.py      # State machine tests
│   ├── test_circuit.py    # Circuit breaker tests
│   ├── test_config.py     # Configuration tests
│   ├── test_exceptions.py # Exception hierarchy tests
│   ├── test_reviewer.py   # Reviewer gate tests
│   ├── test_supervisor.py # Investigation detection tests
│   └── test_e2e_supervisor.py # Full workflow E2E tests
│
├── .github/
│   └── workflows/
│       └── ci.yml         # GitHub Actions CI/CD pipeline
│
├── docs/
│   └── MCP_SERVERS.md     # Recommended MCP servers
├── CHANGELOG.md
├── STARMAP.md             # This file
├── pyproject.toml
└── README.md
```

## Key Components

### Supervisor (`orchestrator/supervisor.py`)
The CORE of Vibe. Coordinates the full Gemini → Claude → GLM loop:
- Uses **Gemini** (brain) for:
  - Task decomposition into atomic tasks
  - Clarification checks when requests are ambiguous
- Uses **Claude** (worker) for:
  - Task execution via subprocess
- Uses **GLM** (reviewer) for:
  - Code review and quality verification only
- **WorkflowEngine Integration**: Auto-expands tasks into multi-phase workflows
- Handles retries with feedback injection (max 3 attempts)
- Persists results to memory-keeper
- **Circuit Breaker Integration**: Prevents cascading failures
  - Checks `can_execute()` before each task
  - Records success/failure to track failure rate
  - Rejects tasks when circuit is OPEN (fails fast)
- **MCP Routing Table**: Task-type-aware tool recommendations
  - 6 task types: debug, code_write, ui_test, research, code_refactor, database
  - Recommended MCP tools injected into task context
  - Hints guide Claude to use appropriate tools

### Reviewer (`orchestrator/reviewer.py`)
GLM-powered code review gate:
- Evaluates task completion, code quality, scope adherence
- Tracks attempts per task for retry logic
- Builds retry context with previous rejection feedback
- Fails tasks if review crashes (never auto-approve unreviewed code)
- **Memory Management**: Prevents unbounded dict growth
  - `cleanup_completed_task()` clears tracking data after task completion
  - `cleanup_stale_tasks()` removes idle tasks older than 1 hour

### ClaudeExecutor (`claude/executor.py`)
Subprocess integration with Claude Code CLI:
- Streaming JSON output parsing
- Tool call tracking (Read, Write, Edit, Bash, Grep, Glob)
- File change detection from Edit/Write tools
- Timeout protection with configurable tiers
- Clean environment (removes API keys from subprocess)
- **Timeout Checkpointing**: Saves partial work before timeout for recovery
  - `TimeoutCheckpoint` dataclass stores tool calls, file changes, partial output
  - `save_checkpoint()` persists to `~/.config/vibe/checkpoints/`
  - `get_last_checkpoint()` retrieves for retry context

### GeminiClient (`gemini/client.py`)
OpenRouter API wrapper for Gemini 2.0 Flash (brain/orchestrator):
- Task decomposition into atomic tasks
- Clarification question detection
- Streaming responses for real-time display
- Usage tracking and statistics

### GLMClient (`glm/client.py`)
OpenRouter API wrapper for GLM-4.7 (code reviewer only):
- Code review with structured JSON output
- Reviews Claude's work for quality and correctness
- Scope verification (did Claude stay on task?)
- Streaming responses for real-time display

### VibeMemory (`memory/keeper.py`)
Direct SQLite access to memory-keeper database:
- Session management with project isolation
- Context persistence (decisions, tasks, progress)
- Checkpoint creation before risky operations
- Convention storage for cross-project rules
- **Connection Pooling**: Thread-local SQLite connections for efficiency
  - `ConnectionPool` class with automatic connection reuse
  - Validates connections before returning (handles stale)
  - WAL mode and foreign keys enabled per connection

### VibeRepository (`persistence/repository.py`)
Unified persistence layer - single source of truth:
- SQLite database at `~/.config/vibe/vibe.db`
- 18 tables: projects, sessions, messages, tasks, attempts, reviews, etc.
- Crash recovery via heartbeat-based orphan detection
- Full task lifecycle with audit trail
- Conversation persistence for all messages
- WAL mode for concurrent reads, foreign keys enforced
- Atomic operations with transaction support

### Pricing (`pricing.py`)
Cost calculation utilities:
- GLM_COSTS, CLAUDE_COSTS dictionaries (per 1K tokens)
- CostTracker class for session-wide accumulation
- format_cost() for display formatting

### Context Compaction (`memory/compaction.py`)
GLM-powered context summarization:
- Summarizes old items (>24h) to reduce token usage
- Groups by category, preserves recent context
- Safety-first: saves summary before deleting originals

### Pre/Post Task Hooks (`config.py`, `supervisor.py`)
Custom scripts before/after Claude tasks:
- Configured in projects.json: pre_task_hooks, post_task_hooks
- Security: path traversal prevention, executable check, 60s timeout
- Runs in project directory with captured stdout/stderr

### Intelligent Task Orchestration (`orchestrator/workflows/`)
Multi-phase workflow system for smarter task decomposition:
- **WorkflowTemplate**: Multi-phase pipelines per task type (DEBUG, CODE_WRITE, REFACTOR)
- **SubTaskInjector**: Automatically injects relevant sub-tasks based on content
- **SmartTaskDetector**: Intent pattern matching with confidence scoring (0.0-1.0)
- **WorkflowEngine**: Expands GLM tasks into workflow phases

Workflow patterns:
| Task Type | Phases |
|-----------|--------|
| CODE_WRITE | analyze → implement → document → verify |
| DEBUG | reproduce → investigate → fix → verify_fix → add_test |
| CODE_REFACTOR | analyze_dependencies → refactor → verify_behavior |
| RESEARCH | gather_info → summarize |

Injection rules auto-add tasks:
- Writing code → "Add inline comments", "Run tests"
- Fixing bugs → "Verify fix", "Check regressions"
- Refactoring → "Analyze usages first", "Update references"

## CLI Commands

### Interactive Commands
| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/status` | Show session status |
| `/usage` | Show Gemini/GLM token usage |
| `/memory` | Show memory statistics |
| `/research <query>` | Research via Perplexity |
| `/gemini <message>` | Direct chat with Gemini (no task execution) |
| `/github` | Show GitHub repo info |
| `/issues` | List open issues |
| `/prs` | List open pull requests |
| `/convention` | Manage global conventions |
| `/debug start` | Start debug session |
| `/quit` | Exit gracefully |
| `ESC` | Cancel running task (keeps session alive) |

### Subcommands
| Command | Description |
|---------|-------------|
| `vibe` | Start interactive session (single terminal) |
| `vibe-split` | **Recommended**: Split terminal (vibe left, claude right) |
| `vibe list` | List all registered projects |
| `vibe add <name> <path>` | Add a new project |
| `vibe remove <name>` | Remove a project |
| `vibe ping` | Test Gemini and GLM API connectivity |
| `vibe restore` | List/recover crashed sessions |
| `vibe logs` | View and analyze logs |

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

- **2026-01-18**: Performance Improvements - Adaptive Learning & Recovery
  - Task-type specific GLM reviews (CODE_WRITE, TEST, ANALYZE prompts)
  - Unified timeout tiers: QUICK (120s), CODE (300s), DEBUG (600s), RESEARCH (900s)
  - Persistent checkpoint recovery: saves to `~/.config/vibe/checkpoints/`
  - Gemini pattern-aware decomposition: historical patterns inform task planning
  - get_decomposition_hints() provides success/failure rates to Gemini
- **2026-01-18**: Direct Gemini Chat, ESC Cancellation & Claim Verification
  - Added `/gemini <msg>` command for direct orchestrator conversation (strategy, clarification)
  - Added ESC key support to cancel running tasks without killing session
  - Implemented KeyboardMonitor class with Unix termios/tty support
  - GLM prompts now include claim verification (detects Claude lying about changes)
  - Gemini prompts enhanced with anti-over-engineering rules
  - Cost display shows Gemini/GLM usage (Claude is fixed price)
  - Comprehensive logging with GeminiLogEntry for orchestrator tracking
- **2026-01-17**: Task-Type Routing, Kanban Board & Pattern Learning
  - Added `task_routing.py`: Intelligent per-task-type configuration (timeout tiers, review skipping, retries)
  - Added `task_board.py`: Kanban-style task tracking (BACKLOG → IN_PROGRESS → REVIEW → DONE)
  - Added `pattern_learning.py`: Records success/failure patterns, injects learnings into prompts
  - Supervisor now uses task-aware routing for timeouts and review decisions
  - All 141 tests pass
- **2026-01-17**: Gemini/GLM Architecture Split
  - **Gemini** is now the brain/orchestrator (task decomposition, clarification)
  - **GLM** is now code reviewer only (quality verification)
  - Added `vibe/gemini/` module with GeminiClient and prompts
  - Updated Supervisor to use both Gemini and GLM clients
  - Updated all tests to reflect new architecture
  - Updated startup to validate both Gemini and GLM connections
  - Updated `/usage` and `vibe ping` to show both models
- **2026-01-17**: Buffer Overflow Fix
  - Fixed `LimitOverrunError` when Claude outputs large JSON lines (>64KB)
  - Replaced `readline()` with chunked `read()` in executor
  - Removed TUI module (replaced by split terminal workflow)
- **2026-01-13**: Split Terminal Mode & Installation
  - `vibe-split` command: tmux launcher with left=vibe, right=claude output
  - Real-time Claude output streaming to `~/.config/vibe/claude-live.log`
  - Executor streams line-by-line for live display
  - `pip install -e .` installs `vibe` command
- **2026-01-13**: Comprehensive Inline Comments & Final Linting
  - 81 inline comments added to 4 key modules (supervisor, executor, reviewer, client)
  - Zero linting errors (all E501, F841, E741, N806 issues resolved)
  - Comments focus on "why" not "what" - design decisions documented
  - All 141 tests pass (100% pass rate, 2.9s runtime)
- **2026-01-13**: Code Quality & Security Fixes
  - Fixed security issue: review failures now reject tasks (never auto-approve)
  - Ruff linting fixes (104 auto-fixed issues)
  - Code formatting standardized across 34 files
  - Exception docstrings updated for reserved exceptions
- **2026-01-13**: CLI Module Refactoring
  - Split 2268-line `cli.py` into 8 focused modules
  - `cli/startup.py`: Startup validation
  - `cli/project.py`: Project management
  - `cli/debug.py`: Debug workflow
  - `cli/execution.py`: Task execution
  - `cli/commands.py`: Slash commands
  - `cli/interactive.py`: Conversation loop
  - `cli/typer_commands.py`: CLI entry points
  - `cli/prompt.py`: Enhanced prompt
- **2026-01-13**: CI/CD Pipeline
  - GitHub Actions workflow (Python 3.11/3.12)
  - pytest-cov with Codecov integration
  - Ruff linting and mypy type checking
- **2026-01-13**: E2E Integration Tests
  - 10 comprehensive end-to-end tests
  - Full workflow coverage (decomposition → execution → review)
  - 141 total tests passing
- **2026-01-13**: GLM Context & Memory Improvements
  - ExecutionDetails model for full task execution records
  - execution_details table with compressed diff storage
  - ContextSettings: configurable limits (100K diff, 25 memory items)
  - Smarter get_git_diff: filters noise, returns truncation flag
  - Preventive warnings for large-scope tasks
  - Truncation warnings prepended to GLM review when diff exceeds limit
- **2026-01-13**: Performance & Reliability Optimizations
  - MCP routing table with 6 task types (auto-injects tool hints)
  - Circuit breaker integration in Supervisor (fail-fast)
  - Connection pooling for memory keeper (thread-local reuse)
  - Timeout checkpointing (saves partial work for recovery)
  - Memory leak fix in Reviewer (cleanup methods)
  - Cached SmartTaskDetector singleton (avoids recreation)
  - Fixed UPSERT bug in memory keeper (preserves IDs)
  - WorkflowEngine integration in Supervisor
- **2026-01-13**: Intelligent GLM Orchestration (WorkflowEngine, SubTaskInjector, SmartTaskDetector)
- **2026-01-13**: Critical bug fixes (executor.py double-prompt, method name typo)
- **2026-01-11**: Unified persistence layer (VibeRepository, crash recovery, full task tracking)
- **2026-01-11**: Core implementation complete (Supervisor, Reviewer)
- **2026-01-11**: Robustness fixes (context overflow, state bleed, review fallback)
- **2026-01-11**: Task enforcer, conventions, debug sessions
- **2026-01-10**: Foundation, GLM client, Claude executor

## GitHub

Repository: https://github.com/SEO-Geek/vibe-orchestrator

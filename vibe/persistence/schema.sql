-- Vibe Orchestrator - Comprehensive SQLite Schema
-- Version: 1.0.0
--
-- Design Principles:
-- 1. Single source of truth - no in-memory caches needed
-- 2. Full history preservation - never delete, only mark inactive
-- 3. Crash recovery support - detect orphaned active sessions
-- 4. Project isolation - multi-project support with foreign keys
-- 5. Efficient queries - indexes on common access patterns
--
-- Schema version tracking for migrations
-- ============================================================================

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now')),
    description TEXT
);

INSERT OR IGNORE INTO schema_version (version, description) VALUES (1, 'Initial schema');

-- ============================================================================
-- PROJECTS - Root entity, all data is project-scoped
-- ============================================================================

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,                          -- UUID
    name TEXT NOT NULL UNIQUE,                    -- Human-readable name (e.g., "athena")
    path TEXT NOT NULL,                           -- Absolute filesystem path
    starmap TEXT DEFAULT 'STARMAP.md',           -- Path to starmap file relative to project
    claude_md TEXT DEFAULT 'CLAUDE.md',          -- Path to CLAUDE.md relative to project
    test_command TEXT DEFAULT 'pytest -v',        -- Default test command
    description TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_accessed_at TEXT,                        -- For cleanup/LRU
    is_active INTEGER NOT NULL DEFAULT 1          -- Soft delete
);

CREATE INDEX IF NOT EXISTS idx_projects_name ON projects(name);
CREATE INDEX IF NOT EXISTS idx_projects_active ON projects(is_active);

-- ============================================================================
-- SESSIONS - Vibe orchestrator session tracking
-- ============================================================================

-- Session status enum values:
-- 'initializing' - Session starting up
-- 'active' - Normal operation
-- 'crashed' - Detected orphan (process died)
-- 'completed' - Graceful shutdown
-- 'error' - Ended due to error

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,                          -- UUID
    project_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'initializing',
    pid INTEGER,                                  -- OS process ID for orphan detection
    hostname TEXT,                                -- Machine hostname for multi-host detection
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at TEXT,
    last_heartbeat_at TEXT,                       -- Updated periodically for crash detection
    summary TEXT,                                 -- End-of-session summary
    error_message TEXT,                           -- If status = 'error'
    total_tasks_completed INTEGER DEFAULT 0,
    total_tasks_failed INTEGER DEFAULT 0,
    total_cost_usd REAL DEFAULT 0.0,

    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_heartbeat ON sessions(last_heartbeat_at);

-- ============================================================================
-- CONVERSATIONS - Chat history between user, GLM, and system
-- ============================================================================

-- Message roles:
-- 'user' - User input
-- 'glm' - GLM (supervisor) response
-- 'system' - System messages (errors, status updates)
-- 'assistant' - Parsed GLM outputs (tasks, reviews)

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,                          -- UUID
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,                           -- user, glm, system, assistant
    content TEXT NOT NULL,                        -- Message content (can be large)
    message_type TEXT,                            -- clarification, decomposition, review, etc.
    parent_message_id TEXT,                       -- For threading/context
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    tokens_used INTEGER,                          -- Token count if from LLM
    cost_usd REAL,                                -- Cost if from LLM
    metadata TEXT,                                -- JSON blob for extra data

    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (parent_message_id) REFERENCES messages(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_messages_type ON messages(message_type);
CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);

-- ============================================================================
-- TASKS - Task lifecycle tracking
-- ============================================================================

-- Task status enum values:
-- 'pending' - Created but not started
-- 'queued' - In the task queue
-- 'executing' - Claude is working on it
-- 'reviewing' - GLM is reviewing output
-- 'completed' - Successfully finished
-- 'failed' - Failed after all retries
-- 'cancelled' - User cancelled
-- 'skipped' - Skipped due to dependency failure

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,                          -- UUID
    session_id TEXT NOT NULL,
    parent_task_id TEXT,                          -- For subtask relationships
    sequence_num INTEGER NOT NULL,                -- Order within session/parent
    description TEXT NOT NULL,
    files TEXT,                                   -- JSON array of file paths
    constraints TEXT,                             -- JSON array of constraints
    success_criteria TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    priority INTEGER DEFAULT 0,                   -- Higher = more important
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    started_at TEXT,
    completed_at TEXT,
    created_by TEXT,                              -- 'glm_decomposition', 'user', 'retry'
    original_request TEXT,                        -- Original user request this came from

    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (parent_task_id) REFERENCES tasks(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_parent ON tasks(parent_task_id);
CREATE INDEX IF NOT EXISTS idx_tasks_sequence ON tasks(session_id, sequence_num);

-- ============================================================================
-- TASK STATUS TRANSITIONS - Full audit trail of status changes
-- ============================================================================

CREATE TABLE IF NOT EXISTS task_status_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    from_status TEXT,                             -- NULL for initial creation
    to_status TEXT NOT NULL,
    reason TEXT,                                  -- Why the transition happened
    transitioned_at TEXT NOT NULL DEFAULT (datetime('now')),
    triggered_by TEXT,                            -- 'glm', 'claude', 'user', 'system'

    FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_transitions_task ON task_status_transitions(task_id);
CREATE INDEX IF NOT EXISTS idx_transitions_time ON task_status_transitions(transitioned_at);

-- ============================================================================
-- TASK ATTEMPTS - Each Claude execution attempt for a task
-- ============================================================================

-- Attempt result enum:
-- 'success' - Task completed successfully
-- 'failed' - Execution error
-- 'timeout' - Timed out
-- 'rejected' - GLM rejected the output
-- 'partial' - Partially successful

CREATE TABLE IF NOT EXISTS task_attempts (
    id TEXT PRIMARY KEY,                          -- UUID
    task_id TEXT NOT NULL,
    attempt_num INTEGER NOT NULL,                 -- 1, 2, 3...

    -- Input
    prompt TEXT NOT NULL,                         -- Full prompt sent to Claude
    timeout_tier TEXT DEFAULT 'code',             -- quick, code, debug, research
    allowed_tools TEXT,                           -- JSON array of allowed tools

    -- Output
    result TEXT NOT NULL DEFAULT 'pending',       -- success, failed, timeout, rejected, partial
    response_text TEXT,                           -- Claude's full response (can be very large)
    error_message TEXT,                           -- If failed
    summary TEXT,                                 -- Brief summary of what was done

    -- Metrics
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    duration_ms INTEGER,
    cost_usd REAL DEFAULT 0.0,
    tokens_used INTEGER,
    num_turns INTEGER DEFAULT 0,
    claude_session_id TEXT,                       -- Claude's internal session ID

    -- Tool tracking
    tool_calls TEXT,                              -- JSON array of tool calls made

    FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_attempts_task ON task_attempts(task_id);
CREATE INDEX IF NOT EXISTS idx_attempts_result ON task_attempts(result);
CREATE INDEX IF NOT EXISTS idx_attempts_started ON task_attempts(started_at);

-- ============================================================================
-- FILE CHANGES - Track all file modifications per attempt
-- ============================================================================

-- Change type enum:
-- 'create' - New file
-- 'modify' - Edited existing file
-- 'delete' - Removed file
-- 'rename' - Renamed (old_path -> file_path)

CREATE TABLE IF NOT EXISTS file_changes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_id TEXT NOT NULL,
    file_path TEXT NOT NULL,                      -- Absolute path
    change_type TEXT NOT NULL,                    -- create, modify, delete, rename
    old_path TEXT,                                -- For renames
    diff_content TEXT,                            -- Git diff or content preview
    lines_added INTEGER,
    lines_removed INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),

    FOREIGN KEY (attempt_id) REFERENCES task_attempts(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_file_changes_attempt ON file_changes(attempt_id);
CREATE INDEX IF NOT EXISTS idx_file_changes_path ON file_changes(file_path);

-- ============================================================================
-- REVIEWS - GLM review of task attempts
-- ============================================================================

CREATE TABLE IF NOT EXISTS reviews (
    id TEXT PRIMARY KEY,                          -- UUID
    attempt_id TEXT NOT NULL,
    approved INTEGER NOT NULL,                    -- 0 or 1
    issues TEXT,                                  -- JSON array of issues found
    feedback TEXT NOT NULL,
    suggested_next_steps TEXT,                    -- JSON array
    reviewed_at TEXT NOT NULL DEFAULT (datetime('now')),
    review_duration_ms INTEGER,
    tokens_used INTEGER,

    FOREIGN KEY (attempt_id) REFERENCES task_attempts(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_reviews_attempt ON reviews(attempt_id);
CREATE INDEX IF NOT EXISTS idx_reviews_approved ON reviews(approved);

-- ============================================================================
-- DEBUG SESSIONS - Extended debugging workflow tracking
-- ============================================================================

CREATE TABLE IF NOT EXISTS debug_sessions (
    id TEXT PRIMARY KEY,                          -- UUID
    session_id TEXT NOT NULL,                     -- Parent orchestrator session
    problem TEXT NOT NULL,                        -- The problem being debugged
    hypothesis TEXT,                              -- Current working hypothesis
    must_preserve TEXT,                           -- JSON array of features to preserve
    is_active INTEGER NOT NULL DEFAULT 1,
    is_solved INTEGER NOT NULL DEFAULT 0,
    initial_git_commit TEXT,                      -- Git state when session started
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    resolved_at TEXT,

    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_debug_sessions_session ON debug_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_debug_sessions_active ON debug_sessions(is_active);

-- ============================================================================
-- DEBUG ITERATIONS - Each iteration within a debug session
-- ============================================================================

CREATE TABLE IF NOT EXISTS debug_iterations (
    id TEXT PRIMARY KEY,                          -- UUID
    debug_session_id TEXT NOT NULL,
    iteration_num INTEGER NOT NULL,

    -- Task given to Claude
    task_description TEXT NOT NULL,
    starting_points TEXT,                         -- JSON array
    what_to_look_for TEXT,
    success_criteria TEXT,

    -- Claude's output
    output TEXT,                                  -- Full response
    files_examined TEXT,                          -- JSON array
    files_changed TEXT,                           -- JSON array
    structured_findings TEXT,                     -- JSON object

    -- GLM review
    review_approved INTEGER,
    review_is_solved INTEGER,
    review_feedback TEXT,
    review_next_task TEXT,

    -- Metrics
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    duration_ms INTEGER,

    -- Rollback support
    git_checkpoint TEXT,                          -- Git commit before this iteration

    FOREIGN KEY (debug_session_id) REFERENCES debug_sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_debug_iters_session ON debug_iterations(debug_session_id);
CREATE INDEX IF NOT EXISTS idx_debug_iters_num ON debug_iterations(debug_session_id, iteration_num);

-- ============================================================================
-- CONTEXT ITEMS - Key-value storage for project context
-- ============================================================================

-- Category enum:
-- 'task' - Task completion record
-- 'decision' - Design/architecture decision
-- 'progress' - Progress marker
-- 'note' - General note
-- 'error' - Error record
-- 'warning' - Warning record
-- 'convention' - Coding convention

CREATE TABLE IF NOT EXISTS context_items (
    id TEXT PRIMARY KEY,                          -- UUID
    project_id TEXT NOT NULL,
    session_id TEXT,                              -- Optional, NULL for global items
    key TEXT NOT NULL,
    value TEXT NOT NULL,                          -- Can be large (JSON, text)
    category TEXT NOT NULL DEFAULT 'note',
    priority TEXT NOT NULL DEFAULT 'normal',      -- high, normal, low
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT,                              -- Optional expiration

    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_context_project ON context_items(project_id);
CREATE INDEX IF NOT EXISTS idx_context_session ON context_items(session_id);
CREATE INDEX IF NOT EXISTS idx_context_key ON context_items(key);
CREATE INDEX IF NOT EXISTS idx_context_category ON context_items(category);
CREATE UNIQUE INDEX IF NOT EXISTS idx_context_unique_key ON context_items(project_id, key);

-- ============================================================================
-- CONVENTIONS - Global conventions across projects
-- ============================================================================

CREATE TABLE IF NOT EXISTS conventions (
    id TEXT PRIMARY KEY,                          -- UUID
    key TEXT NOT NULL UNIQUE,                     -- e.g., "browser-testing"
    convention TEXT NOT NULL,                     -- The rule text
    applies_to TEXT DEFAULT 'all',                -- all, python, javascript, etc.
    created_by_project TEXT,                      -- Which project added it
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_conventions_applies ON conventions(applies_to);
CREATE INDEX IF NOT EXISTS idx_conventions_active ON conventions(is_active);

-- ============================================================================
-- CHECKPOINTS - Recovery checkpoints with git state
-- ============================================================================

CREATE TABLE IF NOT EXISTS checkpoints (
    id TEXT PRIMARY KEY,                          -- UUID
    session_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    git_branch TEXT,
    git_commit TEXT,
    git_status TEXT,                              -- Short status output
    created_at TEXT NOT NULL DEFAULT (datetime('now')),

    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_session ON checkpoints(session_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_created ON checkpoints(created_at);

-- ============================================================================
-- CHECKPOINT CONTEXT SNAPSHOT - Links checkpoints to context state
-- ============================================================================

CREATE TABLE IF NOT EXISTS checkpoint_context (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checkpoint_id TEXT NOT NULL,
    context_item_id TEXT NOT NULL,

    FOREIGN KEY (checkpoint_id) REFERENCES checkpoints(id) ON DELETE CASCADE,
    FOREIGN KEY (context_item_id) REFERENCES context_items(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_checkpoint_ctx_cp ON checkpoint_context(checkpoint_id);

-- ============================================================================
-- TOOL USAGE METRICS - Aggregate tool usage statistics
-- ============================================================================

CREATE TABLE IF NOT EXISTS tool_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    invocation_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    total_duration_ms INTEGER DEFAULT 0,
    last_used_at TEXT,

    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tool_usage_session ON tool_usage(session_id);
CREATE INDEX IF NOT EXISTS idx_tool_usage_tool ON tool_usage(tool_name);
CREATE UNIQUE INDEX IF NOT EXISTS idx_tool_usage_unique ON tool_usage(session_id, tool_name);

-- ============================================================================
-- REQUEST LOG - User request history for context
-- ============================================================================

CREATE TABLE IF NOT EXISTS requests (
    id TEXT PRIMARY KEY,                          -- UUID
    session_id TEXT NOT NULL,
    request_text TEXT NOT NULL,
    result_summary TEXT,
    tasks_created INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending',                -- pending, completed, failed, cancelled
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,

    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_requests_session ON requests(session_id);
CREATE INDEX IF NOT EXISTS idx_requests_created ON requests(created_at);

-- ============================================================================
-- VIEWS - Useful aggregations
-- ============================================================================

-- Active sessions that may be orphaned (no heartbeat in 5 minutes)
CREATE VIEW IF NOT EXISTS orphaned_sessions AS
SELECT
    s.*,
    p.name as project_name,
    p.path as project_path,
    ROUND((julianday('now') - julianday(s.last_heartbeat_at)) * 24 * 60, 1) as minutes_since_heartbeat
FROM sessions s
JOIN projects p ON s.project_id = p.id
WHERE s.status = 'active'
  AND s.last_heartbeat_at IS NOT NULL
  AND datetime(s.last_heartbeat_at) < datetime('now', '-5 minutes');

-- Session summary with task counts
CREATE VIEW IF NOT EXISTS session_summary AS
SELECT
    s.id,
    s.project_id,
    p.name as project_name,
    s.status,
    s.started_at,
    s.ended_at,
    s.total_cost_usd,
    COUNT(DISTINCT t.id) as total_tasks,
    SUM(CASE WHEN t.status = 'completed' THEN 1 ELSE 0 END) as completed_tasks,
    SUM(CASE WHEN t.status = 'failed' THEN 1 ELSE 0 END) as failed_tasks,
    COUNT(DISTINCT m.id) as message_count
FROM sessions s
JOIN projects p ON s.project_id = p.id
LEFT JOIN tasks t ON t.session_id = s.id
LEFT JOIN messages m ON m.session_id = s.id
GROUP BY s.id;

-- Recent task history for GLM context
CREATE VIEW IF NOT EXISTS recent_task_history AS
SELECT
    t.id,
    t.session_id,
    t.description,
    t.status,
    t.completed_at,
    ta.summary,
    ta.result,
    GROUP_CONCAT(DISTINCT fc.file_path) as files_changed
FROM tasks t
LEFT JOIN task_attempts ta ON ta.task_id = t.id AND ta.result = 'success'
LEFT JOIN file_changes fc ON fc.attempt_id = ta.id
WHERE t.status IN ('completed', 'failed')
GROUP BY t.id
ORDER BY t.completed_at DESC
LIMIT 50;

-- ============================================================================
-- TRIGGERS - Automatic housekeeping
-- ============================================================================

-- Update session heartbeat when tasks change
CREATE TRIGGER IF NOT EXISTS update_session_heartbeat
AFTER INSERT ON tasks
BEGIN
    UPDATE sessions
    SET last_heartbeat_at = datetime('now')
    WHERE id = NEW.session_id AND status = 'active';
END;

-- Track task status transitions automatically
CREATE TRIGGER IF NOT EXISTS track_task_status_change
AFTER UPDATE OF status ON tasks
WHEN OLD.status != NEW.status
BEGIN
    INSERT INTO task_status_transitions (task_id, from_status, to_status, triggered_by)
    VALUES (NEW.id, OLD.status, NEW.status, 'system');
END;

-- Update project last_accessed when session starts
CREATE TRIGGER IF NOT EXISTS update_project_access
AFTER INSERT ON sessions
BEGIN
    UPDATE projects
    SET last_accessed_at = datetime('now')
    WHERE id = NEW.project_id;
END;

-- Update debug session timestamp on iteration
CREATE TRIGGER IF NOT EXISTS update_debug_session_timestamp
AFTER INSERT ON debug_iterations
BEGIN
    UPDATE debug_sessions
    SET updated_at = datetime('now')
    WHERE id = NEW.debug_session_id;
END;

-- Update context item timestamp on update
CREATE TRIGGER IF NOT EXISTS update_context_timestamp
AFTER UPDATE ON context_items
BEGIN
    UPDATE context_items
    SET updated_at = datetime('now')
    WHERE id = NEW.id;
END;

-- ============================================================================
-- EXECUTION DETAILS - Full task execution records for debugging and retry
-- ============================================================================

-- Stores comprehensive execution data including full Claude output, tool calls,
-- git diffs, and review results. Diffs are stored as compressed BLOBs when >50KB.
-- Used for: debugging failed tasks, providing retry context, analyzing patterns.

CREATE TABLE IF NOT EXISTS execution_details (
    id TEXT PRIMARY KEY,                          -- UUID
    task_id TEXT NOT NULL,                        -- Reference to task
    session_id TEXT NOT NULL,                     -- Reference to session
    task_description TEXT NOT NULL,

    -- Claude's full output (NOT truncated)
    claude_response TEXT,                         -- Full response text
    tool_calls TEXT,                              -- JSON array of tool calls
    files_changed TEXT,                           -- JSON array of file paths

    -- Git diff (compressed BLOB for large diffs)
    diff_content BLOB,                            -- Full diff, gzip if >50KB
    diff_chars INTEGER DEFAULT 0,                 -- Original diff size
    diff_was_truncated INTEGER DEFAULT 0,         -- Whether review saw truncated

    -- Review results
    review_approved INTEGER,                      -- NULL=pending, 0=rejected, 1=approved
    review_issues TEXT,                           -- JSON array of issues
    review_feedback TEXT,                         -- Full feedback text

    -- Execution metrics
    cost_usd REAL DEFAULT 0.0,
    duration_ms INTEGER DEFAULT 0,
    attempt_number INTEGER DEFAULT 1,

    created_at TEXT NOT NULL DEFAULT (datetime('now')),

    FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_execution_details_task ON execution_details(task_id);
CREATE INDEX IF NOT EXISTS idx_execution_details_session ON execution_details(session_id);
CREATE INDEX IF NOT EXISTS idx_execution_details_created ON execution_details(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_execution_details_approved ON execution_details(review_approved);

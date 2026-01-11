# Recommended MCP Servers for Vibe Coding

This document lists MCP (Model Context Protocol) servers that are highly relevant for vibe coding workflows. GLM can instruct Claude to use these tools appropriately based on task type.

## Currently Installed (Brian's Setup)

All 13 MCP servers are installed and verified working:

### Core Development Tools

| Server | Package | Purpose |
|--------|---------|---------|
| **memory** | @modelcontextprotocol/server-memory | Knowledge graph persistence |
| **memory-keeper** | mcp-memory-keeper | Session checkpoint/recovery |
| **filesystem** | @modelcontextprotocol/server-filesystem | Safe file operations |
| **github** | @modelcontextprotocol/server-github | GitHub API operations |
| **git** | mcp-git | Repository operations |

### Browser Automation (2 options)

| Server | Package | Purpose |
|--------|---------|---------|
| **playwright** | @executeautomation/playwright-mcp-server | Primary browser automation and testing |
| **puppeteer** | @modelcontextprotocol/server-puppeteer | Backup browser automation (stealth mode) |
| **chrome-devtools** | chrome-devtools-mcp | Control Chrome browser (requires --remote-debugging-port=9222) |

### Research & Documentation

| Server | Package | Purpose |
|--------|---------|---------|
| **perplexity** | perplexity-mcp | Deep research & web search |
| **context7** | @upstash/context7-mcp | Fetch current library documentation |

### Reasoning & API Testing

| Server | Package | Purpose |
|--------|---------|---------|
| **sequential-thinking** | @modelcontextprotocol/server-sequential-thinking | Step-by-step reasoning for complex tasks |
| **fetch** | mcp-server-fetch (uvx) | HTTP requests for API testing |

### Database Tools

| Server | Package | Purpose |
|--------|---------|---------|
| **sqlite** | mcp-server-sqlite (uvx) | SQLite database operations |

## Optional Additional Servers

### Code Analysis & Linting

| Server | Install Command | Purpose |
|--------|-----------------|---------|
| **semgrep** | See [repo](https://github.com/semgrep/mcp) | Security and code quality scanning |
| **sonarqube** | See [repo](https://github.com/SonarSource/sonarqube-mcp-server) | Code quality metrics |

### Docker & Container Management

| Server | Install Command | Purpose |
|--------|-----------------|---------|
| **docker-mcp** | See [repo](https://github.com/QuantGeekDev/docker-mcp) | Docker container management |
| **kubernetes** | See [repo](https://github.com/Flux159/mcp-server-kubernetes) | K8s cluster operations |

### Advanced Database

| Server | Install Command | Purpose |
|--------|-----------------|---------|
| **postgres** | `uvx mcp-server-postgres` | PostgreSQL operations |
| **neo4j** | See [repo](https://github.com/neo4j-contrib/mcp-neo4j/) | Graph database |

### Debugging

| Server | Install Command | Purpose |
|--------|-----------------|---------|
| **gdb** | See [repo](https://github.com/pansila/mcp_server_gdb) | GDB debugger control |

## Tool Enforcement by Task Type

Vibe's TaskEnforcer automatically detects task types and enforces appropriate tool usage:

### UI Testing Tasks
**Keywords**: "test ui", "browser test", "e2e test", "verify in browser"

**Required Tools**:
- `mcp__playwright__playwright_navigate`
- `mcp__playwright__playwright_screenshot`

**Preferred Tools**:
- `mcp__playwright__playwright_console_logs`
- `mcp__chrome-devtools__take_screenshot`
- `mcp__chrome-devtools__list_console_messages`

**Forbidden**:
- `curl`, `wget`, `httpie`
- Direct HTTP requests when browser interaction is needed

### Research Tasks
**Keywords**: "research", "look up", "how to", "documentation"

**Preferred Tools**:
- `mcp__perplexity__search`
- `mcp__perplexity__reason`
- `mcp__context7__resolve-library-id`
- `mcp__context7__query-docs`

### Debug Tasks
**Keywords**: "debug", "fix bug", "troubleshoot", "not working"

**Preferred Tools**:
- `mcp__playwright__playwright_console_logs`
- `mcp__chrome-devtools__list_console_messages`
- `mcp__chrome-devtools__list_network_requests`

## Adding New MCP Servers

1. Install the server via npm or from source
2. Add configuration to `~/.config/claude/mcp_servers.json`
3. Update `TaskEnforcer` in `vibe/orchestrator/task_enforcer.py` if needed
4. Add task type keywords and tool requirements

## Resources

- **Official MCP Servers**: https://github.com/modelcontextprotocol/servers
- **Awesome MCP Servers**: https://github.com/wong2/awesome-mcp-servers
- **GitHub MCP Registry**: https://github.com/github/mcp-registry
- **MCP Inspector**: https://github.com/modelcontextprotocol/inspector (for debugging)

## Global Conventions

Use the `/convention` command in Vibe to set rules that apply across all projects:

```
/convention add browser-testing Always use Playwright for browser testing, never curl
/convention add code-comments Add inline comments for complex logic
/convention add verify-screenshot Always take screenshot before marking UI task complete
```

These conventions are stored in memory-keeper and loaded for every Claude task.

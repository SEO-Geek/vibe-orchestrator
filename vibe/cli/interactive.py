"""
Vibe CLI - Interactive Conversation Loop

The main interactive conversation loop where users interact with GLM/Claude.
"""

import asyncio
import copy
import json as json_module
import logging
import subprocess
import sys
from datetime import datetime

from rich.console import Console
from rich.panel import Panel

from vibe.claude.executor import ClaudeExecutor
from vibe.cli import commands
from vibe.cli.debug import execute_debug_workflow
from vibe.cli.execution import execute_task_with_claude, review_with_glm, show_task_result
from vibe.cli.project import load_project_context
from vibe.config import Project, VibeConfig
from vibe.exceptions import ClaudeError
from vibe.glm.client import GLMClient
from vibe.integrations import GitHubOps, PerplexityClient
from vibe.logging import SessionLogEntry, get_session_id, now_iso, session_logger
from vibe.memory.debug_session import DebugSession
from vibe.memory.keeper import VibeMemory
from vibe.memory.task_history import add_request, add_task
from vibe.orchestrator.project_updater import ProjectUpdater
from vibe.persistence.models import MessageRole, MessageType, TaskStatus
from vibe.persistence.repository import VibeRepository
from vibe.state import SessionContext, SessionState

logger = logging.getLogger(__name__)
console = Console()


def persist_message(
    repository: VibeRepository | None,
    session_id: str,
    role: MessageRole,
    content: str,
    message_type: MessageType = MessageType.CHAT,
) -> None:
    """
    Persist a message to the unified persistence layer.

    Fails silently if repository is not available.
    """
    if repository and session_id:
        try:
            repository.add_message(session_id, role, content, message_type)
        except Exception as e:
            logger.debug(f"Failed to persist message: {e}")


async def execute_tasks(
    glm_client: GLMClient,
    executor: ClaudeExecutor,
    project: Project,
    tasks: list[dict],
    memory: VibeMemory | None,
    context: SessionContext,
    user_request: str,
    repository: VibeRepository | None = None,
    debug_session: DebugSession | None = None,
) -> None:
    """
    Execute a list of tasks with Claude and GLM review.

    This is the core task execution loop used by both process_user_request
    and /redo command.
    """
    MAX_RETRIES = 3
    completed = 0
    failed = 0
    all_file_changes: list[str] = []
    all_summaries: list[str] = []
    task_ids: dict[int, str] = {}

    # Create tasks in persistence before execution
    if repository and context.repo_session_id:
        for idx, t in enumerate(tasks, 1):
            try:
                repo_task = repository.create_task(
                    session_id=context.repo_session_id,
                    description=t.get("description", ""),
                    files=t.get("files", []),
                    constraints=t.get("constraints", []),
                    original_request=user_request[:200] if user_request else None,
                )
                task_ids[idx] = repo_task.id
            except Exception as e:
                logger.debug(f"Failed to create task in persistence: {e}")

    context.transition_to(SessionState.EXECUTING)

    for i, task in enumerate(tasks, 1):
        attempt = 0
        task_completed = False
        previous_feedback = ""

        while attempt < MAX_RETRIES and not task_completed:
            attempt += 1
            try:
                # Get debug context if in debug session
                debug_ctx = debug_session.get_context_for_claude() if debug_session else None

                # Add previous rejection feedback to constraints for retry
                task_with_feedback = copy.deepcopy(task)
                if previous_feedback:
                    truncated_feedback = previous_feedback[:500]
                    if len(previous_feedback) > 500:
                        truncated_feedback += "... [truncated]"

                    existing_constraints = task_with_feedback.get("constraints", []) or []
                    retry_constraints = [
                        f"PREVIOUS ATTEMPT REJECTED: {truncated_feedback}",
                        "Address the feedback above before proceeding.",
                    ]
                    task_with_feedback["constraints"] = existing_constraints + retry_constraints
                    console.print(
                        f"\n  [yellow]Retrying task (attempt {attempt}/{MAX_RETRIES})...[/yellow]"
                    )

                # Execute with Claude
                result, tool_verification = await execute_task_with_claude(
                    executor=executor,
                    task=task_with_feedback,
                    task_num=i,
                    total_tasks=len(tasks),
                    debug_context=debug_ctx,
                )

                if not result.success:
                    console.print(f"[red]Task failed: {result.error}[/red]")
                    failed += 1
                    context.add_error(result.error or "Unknown error")
                    add_task(
                        task.get("description", ""),
                        success=False,
                        summary=result.error or "Unknown error",
                    )
                    if memory:
                        memory.save_task_result(
                            task_description=task.get("description", ""),
                            success=False,
                            summary=result.error or "Unknown error",
                        )
                    break  # Don't retry on execution failure

                # Check tool verification
                if not tool_verification["passed"] and tool_verification.get("missing_required"):
                    console.print(
                        "[yellow]Warning: Task may be incomplete - missing required tools[/yellow]"
                    )

                # Review with GLM
                context.transition_to(SessionState.REVIEWING)
                try:
                    review = await review_with_glm(
                        glm_client=glm_client,
                        task=task,
                        result=result,
                        project_path=project.path,
                    )
                except Exception as review_error:
                    # Review crashed - FAIL the task (never auto-approve unreviewed code)
                    console.print(
                        f"[red]Review failed ({review_error}) - rejecting to ensure code quality[/red]"
                    )
                    review = {
                        "approved": False,
                        "issues": [f"Review system error: {str(review_error)[:100]}"],
                        "feedback": "Task rejected due to review error. Code changes preserved but not approved.",
                    }

                # Persist GLM review response
                review_content = json_module.dumps(
                    {
                        "task": task.get("description", "")[:100],
                        "approved": review.get("approved", False),
                        "issues": review.get("issues", []),
                        "feedback": review.get("feedback", "")[:500],
                    }
                )
                persist_message(
                    repository,
                    context.repo_session_id,
                    MessageRole.GLM,
                    review_content,
                    MessageType.REVIEW,
                )

                # If tool verification failed, add that to review issues
                if not tool_verification["passed"]:
                    existing_issues = review.get("issues", [])
                    existing_issues.append(f"Tool verification: {tool_verification['feedback']}")
                    review["issues"] = existing_issues

                # Show result
                show_task_result(result, review)

                if review.get("approved"):
                    completed += 1
                    task_completed = True
                    context.add_completed_task(task.get("description", ""))
                    all_file_changes.extend(result.file_changes)
                    if result.result:
                        all_summaries.append(result.result)

                    add_task(
                        task.get("description", ""),
                        success=True,
                        summary=result.result or "",
                        files_changed=result.file_changes,
                    )
                    if memory:
                        memory.save_task_result(
                            task_description=task.get("description", ""),
                            success=True,
                            summary=result.result or "",
                            files_changed=result.file_changes,
                        )

                    # Update task status in persistence
                    if repository and i in task_ids:
                        try:
                            repository.update_task_status(
                                task_ids[i], TaskStatus.COMPLETED, reason="Approved by GLM review"
                            )
                        except Exception as e:
                            logger.debug(f"Failed to update task status: {e}")

                    # Git commit after approval
                    if result.file_changes:
                        try:
                            subprocess.run(
                                ["git", "add", "--"] + result.file_changes,
                                cwd=project.path,
                                capture_output=True,
                                timeout=30,
                            )
                            task_desc = task.get("description", "Task completed")[:72]
                            commit_msg = f"vibe: {task_desc}\n\nApproved by GLM review gate."
                            subprocess.run(
                                ["git", "commit", "-m", commit_msg],
                                cwd=project.path,
                                capture_output=True,
                                timeout=30,
                            )
                            console.print(
                                f"  [dim]Git: committed {len(result.file_changes)} file(s)[/dim]"
                            )
                        except Exception as git_err:
                            console.print(f"  [yellow]Git commit skipped: {git_err}[/yellow]")
                else:
                    # Task rejected - prepare for retry
                    feedback_text = review.get("feedback", "")
                    issues_text = "; ".join(review.get("issues", []))
                    previous_feedback = (
                        feedback_text or issues_text or "Task did not meet quality standards."
                    )

                    if memory:
                        memory.save(
                            key=f"rejection-{task.get('id', i)}-{attempt}",
                            value=f"Task: {task.get('description', '')}\nFeedback: {previous_feedback}",
                            category="warning",
                            priority="high",
                        )

                    if attempt >= MAX_RETRIES:
                        failed += 1
                        console.print(f"\n  [red]Task failed after {MAX_RETRIES} attempts[/red]")
                        add_task(
                            task.get("description", ""),
                            success=False,
                            summary=f"Rejected after {MAX_RETRIES} attempts: {previous_feedback[:100]}",
                        )
                        if memory:
                            memory.save_task_result(
                                task_description=task.get("description", ""),
                                success=False,
                                summary=f"Rejected after {MAX_RETRIES} attempts: {previous_feedback}",
                            )
                        if repository and i in task_ids:
                            try:
                                repository.update_task_status(
                                    task_ids[i],
                                    TaskStatus.FAILED,
                                    reason=f"Rejected after {MAX_RETRIES} attempts",
                                )
                            except Exception:
                                pass

                context.transition_to(SessionState.EXECUTING)

            except ClaudeError as e:
                console.print(f"[red]Claude error on task {i}: {e}[/red]")
                failed += 1
                context.add_error(str(e))
                add_task(
                    task.get("description", ""),
                    success=False,
                    summary=f"Error: {str(e)[:200]}",
                )
                if repository and i in task_ids:
                    try:
                        repository.update_task_status(
                            task_ids[i], TaskStatus.FAILED, reason=f"Claude error: {str(e)[:100]}"
                        )
                    except Exception:
                        pass
                break

    # Final summary
    context.transition_to(SessionState.IDLE)
    glm_stats = glm_client.get_usage_stats()
    glm_tokens = glm_stats.get("total_tokens", 0)
    glm_cost = glm_tokens * 0.0000006

    console.print()
    console.print(
        Panel(
            f"[bold]Session Complete[/bold]\n\n"
            f"  Completed: [green]{completed}[/green]\n"
            f"  Failed: [red]{failed}[/red]\n"
            f"  Total: {len(tasks)}\n"
            f"  GLM tokens: {glm_tokens:,} (~${glm_cost:.4f})",
            border_style="blue",
        )
    )

    # Record in context and memory
    summary = f"Executed {completed}/{len(tasks)} tasks successfully"
    context.add_glm_message("assistant", summary)

    if memory:
        memory.save(
            key=f"request-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            value=f"Request: {user_request}\nResult: {summary}",
            category="progress",
            priority="normal",
        )

    # Update project documentation if files were changed
    if completed > 0 and all_file_changes:
        try:
            updater = ProjectUpdater(project.path, glm_client)
            combined_summary = " ".join(all_summaries) if all_summaries else user_request

            with console.status("[bold blue]Updating project docs...[/bold blue]"):
                update_results = await updater.update_after_task(
                    task_description=user_request,
                    files_changed=list(set(all_file_changes)),
                    claude_summary=combined_summary,
                    update_starmap=True,
                    update_changelog=True,
                )

            if update_results.get("starmap"):
                console.print("  [dim]Updated STARMAP.md[/dim]")
            if update_results.get("changelog"):
                console.print("  [dim]Updated CHANGELOG.md[/dim]")
        except Exception as e:
            console.print(f"  [dim]Warning: Project update failed: {e}[/dim]")


async def process_user_request(
    glm_client: GLMClient,
    context: SessionContext,
    project: Project,
    user_request: str,
    memory: VibeMemory | None = None,
    repository: VibeRepository | None = None,
    debug_session: DebugSession | None = None,
) -> None:
    """
    Process a user request through GLM and execute with Claude.

    Args:
        glm_client: GLM client instance
        context: Session context
        project: Current project
        user_request: User's request text
        memory: Optional memory client
        repository: Optional persistence repository
        debug_session: Optional active debug session
    """
    # Log user request
    session_logger.info(
        SessionLogEntry(
            timestamp=now_iso(),
            session_id=get_session_id(),
            event_type="request",
            project_name=context.project_name,
            user_request=user_request[:500],
        ).to_json()
    )

    # Check if this is a debug request
    DEBUG_KEYWORDS = ["debug", "broken", "not working", "bug", "crash", "why is", "what's wrong"]
    first_line = user_request.split("\n")[0].lower()
    is_debug = any(keyword in first_line for keyword in DEBUG_KEYWORDS)

    if is_debug:
        console.print("[dim]Debug request detected - using debug workflow[/dim]")
        add_request(user_request)

        executor = ClaudeExecutor(
            project_path=project.path,
            timeout_tier="research",
        )

        await execute_debug_workflow(
            glm_client=glm_client,
            executor=executor,
            project=project,
            problem=user_request,
            memory=memory,
        )
        return

    # Load project context
    project_context = load_project_context(project)
    add_request(user_request)
    context.add_glm_message("user", user_request)

    # Check if clarification is needed
    console.print()
    with console.status("[bold blue]GLM analyzing request...[/bold blue]"):
        clarification = await glm_client.ask_clarification(
            user_request, project_context, clarification_count=context.clarification_count
        )

    if clarification:
        context.clarification_count += 1
        console.print(
            Panel(
                clarification,
                title="[bold yellow]GLM needs clarification[/bold yellow]",
                border_style="yellow",
            )
        )
        context.add_glm_message("assistant", clarification)
        return

    # Reset clarification count when proceeding
    context.clarification_count = 0

    # Decompose into tasks
    console.print()
    with console.status("[bold blue]GLM decomposing into tasks...[/bold blue]"):
        try:
            tasks = await glm_client.decompose_task(user_request, project_context)
        except Exception as e:
            console.print(f"[red]Error decomposing task: {e}[/red]")
            return

    # Persist GLM decomposition
    if tasks:
        decomposition_content = json_module.dumps(
            [
                {
                    "description": t.get("description", ""),
                    "files": t.get("files", []),
                    "constraints": t.get("constraints", []),
                }
                for t in tasks
            ]
        )
        persist_message(
            repository,
            context.repo_session_id,
            MessageRole.GLM,
            decomposition_content,
            MessageType.DECOMPOSITION,
        )

    # Show task plan
    console.print(
        Panel.fit(
            "[bold]Task Plan[/bold]",
            border_style="blue",
        )
    )

    for i, task in enumerate(tasks, 1):
        console.print(
            f"\n  [bold cyan]Task {i}:[/bold cyan] {task.get('description', 'No description')}"
        )
        if task.get("files"):
            console.print(f"    [dim]Files: {', '.join(task['files'])}[/dim]")
        if task.get("constraints"):
            console.print(f"    [dim]Constraints: {', '.join(task['constraints'])}[/dim]")

    console.print()

    # Ask for confirmation
    try:
        console.print("[bold]Execute these tasks?[/bold] [dim][y/n] (y):[/dim] ", end="")
        sys.stdout.flush()
        confirm = input().strip().lower() or "y"
    except (EOFError, KeyboardInterrupt):
        confirm = "n"

    if confirm != "y":
        console.print("[yellow]Cancelled.[/yellow]")
        return

    # Create checkpoint before execution
    if memory:
        try:
            checkpoint_name = f"pre-execution-{datetime.now().strftime('%H%M%S')}"
            memory.create_checkpoint_with_git(
                name=checkpoint_name,
                description=f"Before executing: {user_request[:50]}",
                project_path=project.path,
            )
            console.print(f"  [dim]Checkpoint created: {checkpoint_name}[/dim]")
        except Exception as e:
            console.print(f"  [dim]Warning: Could not create checkpoint: {e}[/dim]")

    # Load global conventions
    global_conventions = []
    if memory:
        try:
            global_conventions = memory.load_conventions()
            if global_conventions:
                console.print(f"  [dim]Loaded {len(global_conventions)} global conventions[/dim]")
        except Exception:
            pass

    # Initialize Claude executor
    try:
        executor = ClaudeExecutor(
            project_path=project.path,
            timeout_tier="code",
            global_conventions=global_conventions,
        )
    except ClaudeError as e:
        console.print(f"[red]Claude error: {e}[/red]")
        return

    # Execute tasks
    await execute_tasks(
        glm_client=glm_client,
        executor=executor,
        project=project,
        tasks=tasks,
        memory=memory,
        context=context,
        user_request=user_request,
        repository=repository,
        debug_session=debug_session,
    )


def conversation_loop(
    context: SessionContext,
    config: VibeConfig,
    project: Project,
    glm_client: GLMClient,
    memory: VibeMemory | None = None,
    repository: VibeRepository | None = None,
    perplexity: PerplexityClient | None = None,
    github: GitHubOps | None = None,
) -> None:
    """
    Main conversation loop with GLM.

    This is where the user interacts with GLM, which delegates to Claude.
    Features: command history (up/down arrows), tab completion for /commands.
    """
    console.print("[bold]What do you want to work on?[/bold]")
    console.print("[dim]Type your request, or /help for commands, /quit to exit[/dim]")
    console.print("[dim]Use ↑/↓ arrows for history, Tab for command completion[/dim]")
    console.print()

    # Create prompt session with history and completion
    try:
        from vibe.cli.prompt import create_prompt_session, prompt_input

        prompt_session = create_prompt_session(project_path=project.path)
        use_enhanced_prompt = True
    except ImportError:
        prompt_session = None
        use_enhanced_prompt = False
        console.print("[dim]Note: Install prompt_toolkit for enhanced input features[/dim]")

    # Track debug session
    debug_session: DebugSession | None = None

    def read_multiline_input() -> str:
        """Read input that may span multiple lines."""
        if use_enhanced_prompt and prompt_session:
            try:
                return prompt_input(prompt_session, "> ", multiline=True)
            except (KeyboardInterrupt, EOFError):
                return ""

        # Fallback to basic input
        lines: list[str] = []
        console.print("[bold cyan]>[/bold cyan] ", end="")
        sys.stdout.flush()

        empty_count = 0

        try:
            while True:
                line = input()

                if not lines and line.startswith("/"):
                    return line

                if not line.strip():
                    empty_count += 1
                    if empty_count >= 2 and lines:
                        break
                    if empty_count >= 1 and lines:
                        lines.append("")
                        import select

                        if hasattr(select, "select"):
                            readable, _, _ = select.select([sys.stdin], [], [], 0.15)
                            if not readable:
                                if lines and lines[-1] == "":
                                    lines.pop()
                                break
                        else:
                            lines.pop()
                            break
                    continue
                else:
                    empty_count = 0

                lines.append(line)

        except EOFError:
            pass

        return "\n".join(lines)

    while True:
        try:
            user_input = read_multiline_input()

            if not user_input.strip():
                continue

            # Handle exit commands
            if user_input.lower().strip() in ("exit", "quit", "q"):
                if memory and memory.session_id:
                    memory.end_session("Session ended by user")
                if repository and context.repo_session_id:
                    try:
                        stats = context.get_stats()
                        summary = f"Completed {stats['completed_tasks']} tasks, {stats['error_count']} errors"
                        repository.end_session(context.repo_session_id, summary=summary)
                    except Exception:
                        pass
                console.print("[yellow]Goodbye![/yellow]")
                break

            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower().strip()

                if cmd in ("/quit", "/exit", "/q"):
                    if memory and memory.session_id:
                        memory.end_session("Session ended by user")
                    if repository and context.repo_session_id:
                        try:
                            stats = context.get_stats()
                            summary = f"Completed {stats['completed_tasks']} tasks, {stats['error_count']} errors"
                            repository.end_session(context.repo_session_id, summary=summary)
                        except Exception:
                            pass
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                elif cmd == "/help":
                    commands.handle_help()
                elif cmd == "/status":
                    commands.handle_status(context)
                elif cmd == "/usage":
                    commands.handle_usage(glm_client)
                elif cmd == "/memory":
                    commands.handle_memory(memory)
                elif cmd == "/history":
                    commands.handle_history()
                elif cmd == "/redo" or cmd.startswith("/redo "):
                    commands.handle_redo(
                        project,
                        glm_client,
                        context,
                        memory,
                        lambda **kw: asyncio.run(execute_tasks(**kw, repository=repository)),
                    )
                elif cmd.startswith("/convention"):
                    commands.handle_convention(user_input, memory)
                elif cmd.startswith("/research"):
                    commands.handle_research(user_input, project, perplexity)
                elif cmd == "/github":
                    commands.handle_github(github)
                elif cmd == "/issues":
                    commands.handle_issues(github)
                elif cmd == "/prs":
                    commands.handle_prs(github)
                elif cmd.startswith("/debug"):
                    debug_session = commands.handle_debug(
                        user_input, project, memory, debug_session
                    )
                elif cmd.startswith("/rollback"):
                    commands.handle_rollback(user_input, debug_session)
                else:
                    console.print(f"[red]Unknown command: {user_input}[/red]")
                continue

            # Input size validation
            MAX_INPUT_SIZE = 50000
            if len(user_input) > MAX_INPUT_SIZE:
                console.print(
                    f"[yellow]Warning: Input is very large ({len(user_input):,} chars). Truncating.[/yellow]"
                )
                user_input = user_input[:MAX_INPUT_SIZE] + "\n\n[... truncated ...]"

            # Persist user message
            persist_message(
                repository, context.repo_session_id, MessageRole.USER, user_input, MessageType.CHAT
            )

            # Update heartbeat
            if repository and context.repo_session_id:
                try:
                    repository.update_heartbeat(context.repo_session_id)
                except Exception:
                    pass

            # Process request through GLM
            asyncio.run(
                process_user_request(
                    glm_client, context, project, user_input, memory, repository, debug_session
                )
            )

        except KeyboardInterrupt:
            console.print("\n[yellow]Use /quit to exit[/yellow]")
        except EOFError:
            break

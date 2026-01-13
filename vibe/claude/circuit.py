"""
Circuit Breaker for Claude CLI

Prevents repeated failures from overwhelming the system.
Based on Athena's circuit breaker pattern.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, TypeVar

from vibe.exceptions import ClaudeCircuitOpenError

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = auto()  # Normal operation
    OPEN = auto()  # Failing, rejecting calls
    HALF_OPEN = auto()  # Testing if recovered


@dataclass
class CircuitStats:
    """Statistics for circuit breaker."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None


class CircuitBreaker:
    """
    Circuit breaker for Claude CLI calls.

    Opens after consecutive failures to prevent overwhelming a failing service.
    Resets after a cooldown period.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        reset_timeout: float = 60.0,
        half_open_max_calls: int = 1,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit
            reset_timeout: Seconds before trying again after opening
            half_open_max_calls: Calls allowed in half-open state
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
        self._stats = CircuitStats()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        self._check_state_transition()
        return self._state

    @property
    def stats(self) -> CircuitStats:
        """Get circuit statistics."""
        return self._stats

    def _check_state_transition(self) -> None:
        """Check if circuit should transition states."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time is not None:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.reset_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0

    def can_execute(self) -> bool:
        """Check if a call can be made."""
        self._check_state_transition()

        if self._state == CircuitState.CLOSED:
            return True
        elif self._state == CircuitState.HALF_OPEN:
            return self._half_open_calls < self.half_open_max_calls
        else:  # OPEN
            return False

    def record_success(self) -> None:
        """Record a successful call."""
        self._stats.total_calls += 1
        self._stats.successful_calls += 1
        self._stats.last_success_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            # Success in half-open state closes the circuit
            self._state = CircuitState.CLOSED
            self._failure_count = 0
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self._stats.total_calls += 1
        self._stats.failed_calls += 1
        self._stats.last_failure_time = time.time()
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            # Failure in half-open state opens the circuit
            self._state = CircuitState.OPEN
        elif self._state == CircuitState.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN

    def record_rejection(self) -> None:
        """Record a rejected call (circuit open)."""
        self._stats.rejected_calls += 1

    async def execute(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute a function through the circuit breaker.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            ClaudeCircuitOpenError: If circuit is open
        """
        if not self.can_execute():
            self.record_rejection()
            raise ClaudeCircuitOpenError(
                "Circuit breaker is open - too many failures",
                failures=self._failure_count,
                reset_time=self.reset_timeout,
            )

        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1

        try:
            result = await func(*args, **kwargs)
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise

    def reset(self) -> None:
        """Reset circuit to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0

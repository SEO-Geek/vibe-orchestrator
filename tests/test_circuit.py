"""Tests for circuit breaker module."""

import asyncio
import time

import pytest

from vibe.claude.circuit import CircuitBreaker, CircuitState, CircuitStats
from vibe.exceptions import ClaudeCircuitOpenError


class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_all_states_exist(self):
        """Verify all expected states exist."""
        assert CircuitState.CLOSED
        assert CircuitState.OPEN
        assert CircuitState.HALF_OPEN


class TestCircuitStats:
    """Tests for CircuitStats dataclass."""

    def test_stats_defaults(self):
        """Test default stats values."""
        stats = CircuitStats()
        assert stats.total_calls == 0
        assert stats.successful_calls == 0
        assert stats.failed_calls == 0
        assert stats.rejected_calls == 0
        assert stats.last_failure_time is None
        assert stats.last_success_time is None


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_initial_state_closed(self):
        """Circuit starts in CLOSED state."""
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED

    def test_can_execute_when_closed(self):
        """Can execute when circuit is CLOSED."""
        cb = CircuitBreaker()
        assert cb.can_execute()

    def test_opens_after_failures(self):
        """Circuit opens after failure_threshold failures."""
        cb = CircuitBreaker(failure_threshold=3)

        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_cannot_execute_when_open(self):
        """Cannot execute when circuit is OPEN."""
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()

        assert cb.state == CircuitState.OPEN
        assert not cb.can_execute()

    def test_success_resets_failure_count(self):
        """Success in CLOSED state resets failure count."""
        cb = CircuitBreaker(failure_threshold=3)

        cb.record_failure()
        cb.record_failure()
        cb.record_success()

        # After success, need 3 more failures to open
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_transitions_to_half_open_after_timeout(self):
        """Circuit transitions to HALF_OPEN after reset_timeout."""
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)

        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_allows_limited_calls(self):
        """HALF_OPEN state allows half_open_max_calls."""
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.01, half_open_max_calls=2)

        cb.record_failure()
        cb.record_failure()
        time.sleep(0.02)

        assert cb.state == CircuitState.HALF_OPEN
        assert cb.can_execute()

    def test_success_in_half_open_closes_circuit(self):
        """Success in HALF_OPEN closes the circuit."""
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.01)

        cb.record_failure()
        cb.record_failure()
        time.sleep(0.02)

        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_failure_in_half_open_opens_circuit(self):
        """Failure in HALF_OPEN opens the circuit again."""
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.01)

        cb.record_failure()
        cb.record_failure()
        time.sleep(0.02)

        assert cb.state == CircuitState.HALF_OPEN

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_rejection_increments_stats(self):
        """Rejected calls increment rejected_calls stat."""
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()

        cb.record_rejection()
        cb.record_rejection()

        assert cb.stats.rejected_calls == 2

    def test_stats_tracking(self):
        """Stats are properly tracked."""
        cb = CircuitBreaker()

        cb.record_success()
        cb.record_success()
        cb.record_failure()

        stats = cb.stats
        assert stats.total_calls == 3
        assert stats.successful_calls == 2
        assert stats.failed_calls == 1
        assert stats.last_success_time is not None
        assert stats.last_failure_time is not None

    def test_reset(self):
        """Reset returns circuit to CLOSED state."""
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()

        assert cb.state == CircuitState.OPEN

        cb.reset()

        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute()


class TestCircuitBreakerExecute:
    """Tests for CircuitBreaker.execute() method."""

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Successful execution records success."""
        cb = CircuitBreaker()

        async def success_func():
            return "result"

        result = await cb.execute(success_func)

        assert result == "result"
        assert cb.stats.successful_calls == 1

    @pytest.mark.asyncio
    async def test_execute_failure(self):
        """Failed execution records failure and re-raises."""
        cb = CircuitBreaker()

        async def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            await cb.execute(failing_func)

        assert cb.stats.failed_calls == 1

    @pytest.mark.asyncio
    async def test_execute_when_open_raises(self):
        """Execute when OPEN raises ClaudeCircuitOpenError."""
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()

        async def some_func():
            return "result"

        with pytest.raises(ClaudeCircuitOpenError) as exc_info:
            await cb.execute(some_func)

        assert exc_info.value.failures == 2
        assert exc_info.value.reset_time == 60.0

    @pytest.mark.asyncio
    async def test_execute_with_args(self):
        """Execute passes args and kwargs correctly."""
        cb = CircuitBreaker()

        async def add_func(a, b, c=0):
            return a + b + c

        result = await cb.execute(add_func, 1, 2, c=3)

        assert result == 6

    @pytest.mark.asyncio
    async def test_half_open_increments_call_count(self):
        """Execute in HALF_OPEN increments half_open_calls."""
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.01, half_open_max_calls=1)

        cb.record_failure()
        cb.record_failure()
        time.sleep(0.02)

        assert cb.state == CircuitState.HALF_OPEN

        async def success_func():
            return "result"

        await cb.execute(success_func)

        # After success, circuit should be closed
        assert cb.state == CircuitState.CLOSED

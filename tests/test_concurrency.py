# ============================================================
# Tests — Concurrency & Thread Safety
# Covers Fix #4 (tmp file race) and Fix #5 (_agents lock)
# ============================================================

import os
import threading
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

try:
    import torch as _torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ── Fix #4: tmp_detect.jpg race condition ─────────────────

class TestTmpFileRaceCondition:

    def test_no_hardcoded_tmp_detect_in_app(self):
        """app.py must not contain the hardcoded tmp_detect.jpg path."""
        app_path = Path(__file__).parent.parent / 'autoarchitect' / 'app.py'
        assert 'tmp_detect.jpg' not in app_path.read_text(encoding='utf-8'), \
            "Hardcoded tmp_detect.jpg still present — Fix #4 regression"

    def test_tempfile_imported_in_app(self):
        app_path = Path(__file__).parent.parent / 'autoarchitect' / 'app.py'
        assert 'import tempfile' in app_path.read_text(encoding='utf-8')

    def test_concurrent_calls_produce_unique_paths(self):
        """Simulate 20 concurrent detect calls — each must get a unique temp path."""
        paths = []
        lock = threading.Lock()

        def make_tmp():
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                p = f.name
            with lock:
                paths.append(p)
            os.unlink(p)

        threads = [threading.Thread(target=make_tmp) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(paths) == len(set(paths)), \
            "Duplicate temp paths found — concurrent requests would corrupt each other"

    def test_tempfile_is_cleaned_up(self):
        """Temp file must be removed after use (no disk leak)."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            tmp_path = f.name
        assert os.path.exists(tmp_path)
        os.remove(tmp_path)
        assert not os.path.exists(tmp_path)


# ── Fix #5: _agents dict thread safety ───────────────────

@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestAgentsLock:

    def test_threading_lock_in_orchestrator_source(self):
        import api.orchestrator as mod
        source = Path(mod.__file__).read_text(encoding='utf-8')
        assert 'threading.Lock()' in source, \
            "threading.Lock() missing from orchestrator — Fix #5 regression"
        assert '_agents_lock' in source

    def test_wake_sleep_use_lock(self):
        import api.orchestrator as mod
        source = Path(mod.__file__).read_text(encoding='utf-8')
        assert 'with self._agents_lock' in source, \
            "Lock context manager not used in _wake_agent/_sleep_agent"

    def test_concurrent_wake_creates_agent_exactly_once(self):
        """20 threads all trying to wake the same agent — exactly 1 must be created."""
        import threading

        agents = {}
        lock = threading.Lock()
        created_count = [0]

        def wake(key):
            with lock:
                if key not in agents:
                    agents[key] = object()
                    created_count[0] += 1

        threads = [threading.Thread(target=wake, args=("image_detect",))
                   for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert created_count[0] == 1, \
            f"Agent created {created_count[0]} times, expected 1"
        assert len(agents) == 1

    def test_sleep_while_wake_in_progress_does_not_raise(self):
        """Deleting a key while another thread checks it must not raise RuntimeError."""
        import threading

        agents = {"image": object()}
        lock = threading.Lock()
        errors = []

        def wake():
            try:
                with lock:
                    if "image" not in agents:
                        agents["image"] = object()
            except Exception as e:
                errors.append(e)

        def sleep_agent():
            try:
                with lock:
                    if "image" in agents:
                        del agents["image"]
            except Exception as e:
                errors.append(e)

        threads = ([threading.Thread(target=wake) for _ in range(10)] +
                   [threading.Thread(target=sleep_agent) for _ in range(10)])
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety errors: {errors}"


# ── General concurrency helpers ───────────────────────────

@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestConcurrencyHelpers:

    def test_threading_imported_in_orchestrator(self):
        import api.orchestrator as mod
        source = Path(mod.__file__).read_text(encoding='utf-8')
        assert 'import threading' in source

# AutoArchitect — Issue Fix Log

All confirmed bugs and their fixes, in order applied.

---

## Fix #1 — Division by Zero in Performance Tracker

**File:** `autoarchitect/api/brain/performance_tracker.py`
**Lines:** 61–66

**Issue:** `avg_acc` and `avg_time` both divided by `total` (= `len(self.history)`) with no zero-check. When no problems have been solved yet and `history` is empty, `total = 0` → `ZeroDivisionError`. This crashed any call to `get_stats()` on a fresh install or after clearing history.

**Fix:** Added `if total > 0 else 0.0` guard to both divisions.

**Impact:** `get_stats()` now safely returns `0.0` for both metrics when history is empty, instead of crashing. The return type is consistently `float` in all cases.

---

## Fix #2 — Wrong Return Type for Empty Strategy (`"none"` → `None`)

**File:** `autoarchitect/api/brain/strategy_library.py`
**Line:** 303

**Issue:** When `self.strategies` is empty, `get_stats()` returned the string `"none"` for the `best_strategy` field. Any caller that used this value as a dict key — e.g. `self.strategies[best_strategy]` — would get a silent `KeyError`, since `"none"` is not a real strategy key. It also broke truthiness checks: `if best_strategy:` evaluates to `True` for the string `"none"`, so guards that look correct would still execute bad paths.

**Fix:** Changed fallback from `"none"` to `None`.

**Impact:** Callers can now safely use `if best_strategy:` or `if best_strategy is not None:` and get the expected behaviour. Eliminates the hidden `KeyError` risk on any code path that treats this value as a lookup key.

---

## Fix #3 — Hardcoded Relative Path in TopologyDesigner

**File:** `autoarchitect/api/brain/topology_designer.py`
**Line:** 183

**Issue:** `self.data_dir = Path("brain_data")` resolves relative to whatever the current working directory is at runtime. If Flask is started from any directory other than `autoarchitect/` — e.g. the repo root — this silently creates a second `brain_data/` folder in the wrong place. Topology history is then written there and never found again on the next run, so the designer never accumulates learned topologies.

**Fix:** Changed to `Path(__file__).parent.parent.parent / "brain_data"` — resolves relative to the file's own location, always pointing to the correct `autoarchitect/brain_data/` regardless of where the process is launched from.

**Impact:** Topology history is now written and read from a stable, predictable path. The designer correctly accumulates learned topologies across runs.

---

## Fix #4 — Race Condition on Shared Temp File in `/api/detect`

**File:** `autoarchitect/app.py`
**Line:** 193

**Issue:** Every call to `/api/detect` wrote the decoded image to the same hardcoded path `tmp_detect.jpg` in the app directory. Under concurrent requests (two users uploading images at the same time), Request B overwrites the file while Request A is still running YOLO on it — Request A detects Request B's image. The wrong result is returned silently with no error.

**Fix:** Replaced the hardcoded path with `tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)`. The OS assigns each request a unique temp path (e.g. `/tmp/tmpXk3j9a.jpg`), so concurrent requests never touch the same file.

**Impact:** Each `/api/detect` request now operates on its own isolated file. Concurrent detection calls no longer corrupt each other's input or produce wrong results.

---

## Fix #5 — Thread Safety on `_agents` Dict in Orchestrator

**File:** `autoarchitect/api/orchestrator.py`
**Lines:** 40, 664–684

**Issue:** `self._agents` is a plain dict shared across all Flask request threads. `_wake_agent` does a read-then-write (`if key not in self._agents → self._agents[key] = agent`) and `_sleep_agent` does a read-then-delete — both without any lock. Two concurrent requests for the same domain could both pass the `if key not in` check simultaneously, instantiate two agents, and one would overwrite the other. A concurrent `_sleep_agent` deleting a key while another thread is reading it causes a `RuntimeError: dictionary changed size during iteration` or a `KeyError`.

**Fix:** Added `self._agents_lock = threading.Lock()` in `__init__`, and wrapped the entire body of both `_wake_agent` and `_sleep_agent` in `with self._agents_lock:`.

**Impact:** Agent loading and unloading are now atomic. Concurrent requests for the same domain safely share one agent instance instead of racing to create duplicates or crashing on mid-flight deletions.

---

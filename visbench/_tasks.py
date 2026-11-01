"""Single source of truth for the set of eval tasks visbench understands.

Lives here (not in the dataset registry or the orchestrator) so both the
registry's `register()` validation and the orchestrator's `_TASK_DISPATCH`
can import it without forming an import cycle through the dataset
package's import-time registration loop.

Adding a new task means:
  1. Add the name to TASKS below.
  2. Add a dispatch entry in `visbench/orchestrators/run.py::_TASK_DISPATCH`.
A tiny `_check_dispatch_matches_tasks()` runs at orchestrator import time and
warns if the two drift apart.
"""

from __future__ import annotations

TASKS: frozenset[str] = frozenset({
    "pose",
    "homography",
    "homography_points",
    "correspondence",
    "descriptor",
    "fundamental",
    "qualitative",
})

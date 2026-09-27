"""Replay lanes against a Hermes fleet's own Kanban history: which lane each task would get,
what it actually ran on, what it cost in tokens, and whether it finished.

    jev lane replay-build --kanban-db <root>/kanban.db --hermes-root <root> --out rows.jsonl
    jev batch --policy lane --in rows.jsonl --out decisions.jsonl --yes
    jev lane replay-report --rows decisions.jsonl

The build step opens every database read-only (``immutable=1``: a live fleet keeps writing to
them) and sends nothing. A row is one task: its title and body are the ``state`` Jev reads;
the runs it took, the models and reasoning effort they used, their tokens and the final outcome
are labels, never sent. Tasks on a profile listed in ``private_profiles`` are left out.

The report answers the two questions that decide promotion, per lane:

* **tokens**: what the tasks in each lane cost as they ran, and an estimate of what they would
  cost at the lane's own effort, from the fleet's own runs of the same lane at each effort;
* **success parity**: the finish rate of tasks in each lane that happened to run at low effort
  versus higher effort. History is a natural experiment here, not a controlled one: effort was
  set per profile, so the report shows sample sizes and 95% intervals and says when they are too
  small to judge.
"""
from __future__ import annotations

import json
import math
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from . import lanes

SUCCESS = {"completed", "success", "review_requested"}
FAILED = {"crashed", "timed_out", "gave_up", "spawn_failed", "reclaimed", "infrastructure_interrupted",
          "infrastructure_transient", "stale"}
EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")


def _ro(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{path}?immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _effort(model_config: Optional[str]) -> Optional[str]:
    try:
        data = json.loads(model_config or "{}")
    except ValueError:
        return None
    config = data.get("reasoning_config") if isinstance(data, dict) else None
    if isinstance(config, dict):
        if config.get("enabled") is False:
            return "none"
        return str(config.get("effort") or "") or None
    return None


def _sessions(root: Path, profile: str, ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    db = root / "profiles" / profile / "state.db"
    if not db.is_file() or not ids:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    con = _ro(db)
    try:
        for start in range(0, len(ids), 500):
            chunk = list(ids[start:start + 500])
            marks = ",".join("?" * len(chunk))
            for row in con.execute(
                    f"SELECT id, model, model_config, input_tokens, output_tokens, cache_read_tokens, "
                    f"reasoning_tokens, estimated_cost_usd, api_call_count FROM sessions WHERE id IN ({marks})", chunk):
                out[row["id"]] = {
                    "model": row["model"], "effort": _effort(row["model_config"]),
                    "input": int(row["input_tokens"] or 0), "output": int(row["output_tokens"] or 0),
                    "cache_read": int(row["cache_read_tokens"] or 0), "reasoning": int(row["reasoning_tokens"] or 0),
                    "cost": float(row["estimated_cost_usd"] or 0.0), "calls": int(row["api_call_count"] or 0),
                }
    finally:
        con.close()
    return out


def _private_profiles(root: Path) -> List[str]:
    try:
        data = json.loads((root / "jev" / "routing.json").read_text(encoding="utf-8"))
        return [str(p) for p in data.get("private_profiles") or ()]
    except (OSError, ValueError, AttributeError):
        return []


def outcome_class(outcomes: Sequence[Optional[str]], status: Optional[str]) -> str:
    if status in ("done",) or any(o in SUCCESS for o in outcomes):
        return "success"
    last = next((o for o in reversed(outcomes) if o), None)
    if last == "blocked" or status == "blocked":
        return "blocked"
    if last in FAILED or status in ("failed",):
        return "failed"
    return "other"


def build_rows(kanban_db: Any, hermes_root: Any, *, since: Optional[float] = None,
               limit: int = 0, body_chars: int = 2_500) -> List[Dict[str, Any]]:
    """One row per task that has at least one run with a known session (so tokens are known)."""
    root = Path(hermes_root).expanduser()
    private = set(_private_profiles(root))
    con = _ro(Path(kanban_db).expanduser())
    try:
        tasks = {row["id"]: dict(row) for row in con.execute(
            "SELECT id, title, body, status, assignee, created_at, model_override, reasoning_effort FROM tasks")}
        runs = [dict(row) for row in con.execute(
            "SELECT id, task_id, profile, outcome, status, metadata, started_at, ended_at FROM task_runs ORDER BY id")]
    finally:
        con.close()
    by_profile: Dict[str, List[str]] = defaultdict(list)
    for run in runs:
        try:
            meta = json.loads(run.get("metadata") or "{}")
        except ValueError:
            meta = {}
        run["session"] = meta.get("worker_session_id") if isinstance(meta, dict) else None
        if run["session"] and run.get("profile"):
            by_profile[run["profile"]].append(run["session"])
    sessions: Dict[str, Dict[str, Any]] = {}
    for profile, ids in by_profile.items():
        sessions.update(_sessions(root, profile, ids))
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[run["task_id"]].append(run)
    rows: List[Dict[str, Any]] = []
    for task_id, task_runs in grouped.items():
        task = tasks.get(task_id)
        if not task or (since and (task.get("created_at") or 0) < since):
            continue
        if (task.get("assignee") or "") in private or any((r.get("profile") or "") in private for r in task_runs):
            continue
        joined = [(r, sessions.get(r["session"])) for r in task_runs if r.get("session")]
        joined = [(r, s) for r, s in joined if s]
        if not joined:
            continue
        tokens = sum(s["input"] + s["output"] for _, s in joined)
        text = (task.get("title") or "").strip()
        body = (task.get("body") or "").strip()
        if body:
            text += "\n\n" + body[:body_chars]
        efforts = [s["effort"] for _, s in joined]
        rows.append({
            "id": task_id, "state": {"task": text},
            "had_model_override": bool(task.get("model_override")),
            "profile": task.get("assignee"), "runs": len(task_runs), "runs_with_usage": len(joined),
            "models": sorted({s["model"] for _, s in joined if s["model"]}),
            "effort": efforts[-1], "efforts": efforts,
            "tokens": tokens, "input_tokens": sum(s["input"] for _, s in joined),
            "output_tokens": sum(s["output"] for _, s in joined),
            "cache_read_tokens": sum(s["cache_read"] for _, s in joined),
            "reasoning_tokens": sum(s["reasoning"] for _, s in joined),
            "cost_usd": round(sum(s["cost"] for _, s in joined), 6),
            "outcome": outcome_class([r.get("outcome") for r in task_runs], task.get("status")),
            "created_at": task.get("created_at"),
        })
    rows.sort(key=lambda row: row.get("created_at") or 0)
    return rows[-limit:] if limit else rows


def write_rows(rows: Iterable[Mapping[str, Any]], out: Any) -> int:
    count = 0
    with Path(out).expanduser().open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")
            count += 1
    return count


# ── report ──────────────────────────────────────────────────────────────────

def wilson(successes: int, total: int, z: float = 1.96) -> Optional[List[float]]:
    if not total:
        return None
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def _rate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    judged = [r for r in rows if r.get("outcome") in ("success", "failed")]
    wins = sum(1 for r in judged if r["outcome"] == "success")
    return {"n": len(rows), "judged": len(judged), "success_rate": round(wins / len(judged), 3) if judged else None,
            "ci95": wilson(wins, len(judged))}


def _median(values: Sequence[float]) -> Optional[float]:
    return float(statistics.median(values)) if values else None


def report(rows: Sequence[Mapping[str, Any]], *, host: str = "hermes", min_cell: int = 20,
           top: str = "high") -> Dict[str, Any]:
    """Tokens and success by the lane each task would get, against what actually ran.

    ``top`` is the effort an always-top-model policy would run everything at. Projected tokens
    for a lane at effort ``e`` = the task's actual tokens x (median tokens/task at ``e``) /
    (median tokens/task at the effort it ran at), both medians taken within that lane. A cell
    with fewer than ``min_cell`` tasks is not used; the projection then keeps the actual tokens.
    """
    mapped = lanes.targets(host)
    by_lane: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        decision = row.get("decision") or {}
        by_lane[str(decision.get("action") or "unknown")].append(row)
    out: Dict[str, Any] = {"tasks": len(rows), "host": host, "lane_map": mapped, "lanes": {},
                           "sources": _count(r.get("decision", {}).get("source") for r in rows)}
    actual_total = projected_total = top_total = 0.0
    for lane, members in sorted(by_lane.items(), key=lambda kv: (lanes.LANES + ("keep_current", "unknown")).index(kv[0])
                                if kv[0] in lanes.LANES + ("keep_current", "unknown") else 99):
        medians = {}
        by_effort: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in members:
            by_effort[str(row.get("effort") or "unknown")].append(row)
        for effort, cell in by_effort.items():
            if len(cell) >= min_cell:
                medians[effort] = _median([float(r.get("tokens") or 0) for r in cell])
        target_effort = (mapped.get(lane) or {}).get("effort")
        actual = sum(float(r.get("tokens") or 0) for r in members)

        def projected_at(effort: Optional[str]) -> float:
            total = 0.0
            for row in members:
                ran = str(row.get("effort") or "unknown")
                tokens = float(row.get("tokens") or 0)
                if effort and effort in medians and ran in medians and medians[ran]:
                    total += tokens * medians[effort] / medians[ran]
                else:
                    total += tokens
            return total

        lane_projected = projected_at(target_effort)
        lane_top = projected_at(top)
        actual_total += actual
        projected_total += lane_projected
        top_total += lane_top
        out["lanes"][lane] = {
            "tasks": len(members), "share": round(len(members) / len(rows), 3) if rows else None,
            "actual_tokens": int(actual), "projected_tokens_at_lane_effort": int(lane_projected),
            "projected_tokens_always_top": int(lane_top), "target": mapped.get(lane),
            "median_tokens_per_task_by_effort": {e: int(m) for e, m in sorted(medians.items())},
            "success_by_effort": {effort: _rate(cell) for effort, cell in sorted(by_effort.items())},
            "success_all": _rate(members),
            "models": _count(m for r in members for m in (r.get("models") or [])),
        }
    out["totals"] = {
        "actual_tokens": int(actual_total), "projected_tokens_lanes": int(projected_total),
        "projected_tokens_always_top": int(top_total),
        "saving_vs_actual": round(1 - projected_total / actual_total, 3) if actual_total else None,
        "saving_vs_always_top": round(1 - projected_total / top_total, 3) if top_total else None,
    }
    return out


def _count(values: Iterable[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for value in values:
        counts[str(value)] += 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def read_jsonl(path: Any) -> List[Dict[str, Any]]:
    rows = []
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows

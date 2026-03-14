"""
Telemetry collector for Helion kernel evolution runs.

Tracks: generation timing, success/failure rates, score progression,
compilation stats, substrate health. Writes periodic summaries.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class GenerationStats:
    generation: int
    timestamp: float
    wall_time_s: float
    candidates_attempted: int = 0
    candidates_valid: int = 0
    candidates_failed_compile: int = 0
    candidates_failed_correctness: int = 0
    candidates_failed_diff: int = 0
    candidates_improved: int = 0
    best_score: float = -1.0
    best_runtime_us: float = float("inf")
    mean_eval_time_s: float = 0.0
    population_size: int = 0


@dataclass
class RunTelemetry:
    problem_name: str
    substrate_type: str
    start_time: float = field(default_factory=time.time)
    generations: list[GenerationStats] = field(default_factory=list)
    total_candidates: int = 0
    total_valid: int = 0
    total_failed: int = 0
    best_ever_score: float = -1.0
    best_ever_runtime_us: float = float("inf")
    best_ever_generation: int = 0
    best_ever_candidate_id: str = ""
    score_history: list[float] = field(default_factory=list)
    runtime_history_us: list[float] = field(default_factory=list)
    stagnation_counter: int = 0

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.start_time

    @property
    def success_rate(self) -> float:
        if self.total_candidates == 0:
            return 0.0
        return self.total_valid / self.total_candidates

    @property
    def improvement_rate(self) -> float:
        improved = sum(g.candidates_improved for g in self.generations)
        if self.total_valid == 0:
            return 0.0
        return improved / self.total_valid


class TelemetryCollector:
    """Collects and persists telemetry for a single evolution run."""

    def __init__(self, problem_name: str, substrate_type: str, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.data = RunTelemetry(
            problem_name=problem_name, substrate_type=substrate_type
        )
        self._events_path = out_dir / "telemetry_events.jsonl"
        self._current_gen_start: float = 0.0
        self._current_gen_stats: GenerationStats | None = None

    def begin_generation(self, generation: int) -> None:
        self._current_gen_start = time.time()
        self._current_gen_stats = GenerationStats(
            generation=generation,
            timestamp=time.time(),
            wall_time_s=0.0,
        )

    def record_candidate(
        self,
        valid: bool,
        improved: bool,
        score: float,
        runtime_us: float,
        eval_time_s: float,
        failure_type: str | None = None,
        candidate_id: str = "",
    ) -> None:
        g = self._current_gen_stats
        if g is None:
            return

        g.candidates_attempted += 1
        self.data.total_candidates += 1

        if valid:
            g.candidates_valid += 1
            self.data.total_valid += 1
            if improved:
                g.candidates_improved += 1
            if score > g.best_score:
                g.best_score = score
                g.best_runtime_us = runtime_us
        else:
            self.data.total_failed += 1
            if failure_type == "compile":
                g.candidates_failed_compile += 1
            elif failure_type == "correctness":
                g.candidates_failed_correctness += 1
            elif failure_type == "diff":
                g.candidates_failed_diff += 1

        # Track global best
        if valid and score > self.data.best_ever_score:
            self.data.best_ever_score = score
            self.data.best_ever_runtime_us = runtime_us
            self.data.best_ever_generation = g.generation
            self.data.best_ever_candidate_id = candidate_id
            self.data.stagnation_counter = 0
        elif valid:
            self.data.stagnation_counter += 1

        # Append event
        event = {
            "type": "candidate_eval",
            "generation": g.generation,
            "ts": time.time(),
            "valid": valid,
            "improved": improved,
            "score": score,
            "runtime_us": runtime_us,
            "eval_time_s": eval_time_s,
            "failure_type": failure_type,
            "candidate_id": candidate_id,
        }
        self._append_event(event)

    def end_generation(self, population_size: int) -> GenerationStats:
        g = self._current_gen_stats
        if g is None:
            raise RuntimeError("end_generation called without begin_generation")

        g.wall_time_s = time.time() - self._current_gen_start
        g.population_size = population_size
        if g.candidates_attempted > 0:
            # mean eval time only from valid candidates
            pass

        self.data.generations.append(g)
        self.data.score_history.append(g.best_score)
        self.data.runtime_history_us.append(g.best_runtime_us)

        self._append_event({
            "type": "generation_end",
            "generation": g.generation,
            "ts": time.time(),
            "wall_time_s": g.wall_time_s,
            "candidates_attempted": g.candidates_attempted,
            "candidates_valid": g.candidates_valid,
            "candidates_improved": g.candidates_improved,
            "best_score": g.best_score,
            "best_runtime_us": g.best_runtime_us,
            "population_size": population_size,
            "stagnation_counter": self.data.stagnation_counter,
        })

        self._current_gen_stats = None
        return g

    def write_summary(self) -> Path:
        summary = {
            "problem_name": self.data.problem_name,
            "substrate_type": self.data.substrate_type,
            "elapsed_s": self.data.elapsed_s,
            "total_generations": len(self.data.generations),
            "total_candidates": self.data.total_candidates,
            "total_valid": self.data.total_valid,
            "total_failed": self.data.total_failed,
            "success_rate": self.data.success_rate,
            "improvement_rate": self.data.improvement_rate,
            "best_ever_score": self.data.best_ever_score,
            "best_ever_runtime_us": self.data.best_ever_runtime_us,
            "best_ever_generation": self.data.best_ever_generation,
            "best_ever_candidate_id": self.data.best_ever_candidate_id,
            "stagnation_counter": self.data.stagnation_counter,
            "score_history": self.data.score_history,
            "runtime_history_us": self.data.runtime_history_us,
            "per_generation": [asdict(g) for g in self.data.generations],
        }
        path = self.out_dir / "telemetry_summary.json"
        path.write_text(json.dumps(summary, indent=2))
        return path

    def print_status(self, generation: int) -> None:
        d = self.data
        gen = self.data.generations[-1] if self.data.generations else None
        elapsed = d.elapsed_s

        print(f"\n{'='*60}")
        print(f"  Gen {generation} | {elapsed:.0f}s elapsed | {d.total_candidates} candidates total")
        print(f"  Success rate: {d.success_rate:.0%} ({d.total_valid}/{d.total_candidates})")
        if gen:
            print(f"  This gen: {gen.candidates_valid}/{gen.candidates_attempted} valid, "
                  f"{gen.candidates_improved} improved, {gen.wall_time_s:.1f}s")
        if d.best_ever_runtime_us < float("inf"):
            print(f"  Best runtime: {d.best_ever_runtime_us:.2f} us (gen {d.best_ever_generation})")
        if d.stagnation_counter > 0:
            print(f"  Stagnation: {d.stagnation_counter} evals without improvement")
        print(f"{'='*60}")

    def _append_event(self, event: dict) -> None:
        with self._events_path.open("a") as f:
            f.write(json.dumps(event) + "\n")

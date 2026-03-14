from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from substrate import EvalResult
from substrate import LocalSubstrate
from substrate import SSHSubstrate
from substrate import make_substrate


@dataclass(frozen=True)
class VariantSpec:
    label: str
    source_path: Path


@dataclass
class VariantResult:
    label: str
    source_path: str
    problem_name: str
    valid: bool
    passed_correctness: bool
    aggregate_score: float
    mean_runtime_us: float | None
    min_runtime_us: float | None
    compile_time_s: float
    test_time_s: float
    benchmark_time_s: float
    failure_reasons: list[str]
    raw_stdout_tail: str
    raw_stderr_tail: str
    improves_baseline: bool = False


def _problem_dir_name(problem: str) -> str:
    return problem if problem.endswith("_py") else f"{problem}_py"


def _sanitize_label(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
    return sanitized or "variant"


def _load_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text())


def _load_variants(problem_dir: Path, variant_dir: Path, include_baseline: bool) -> list[VariantSpec]:
    specs: list[VariantSpec] = []
    if include_baseline:
        specs.append(VariantSpec(label="baseline_seed", source_path=problem_dir / "seed.py"))

    for path in sorted(variant_dir.glob("*.py")):
        specs.append(VariantSpec(label=path.stem, source_path=path))

    if not specs:
        raise RuntimeError(f"No variants found in {variant_dir}")
    return specs


def _prepare_problem_copy(config: dict, base_problem_name: str, new_problem_name: str) -> None:
    substrate = make_substrate(config)
    if isinstance(substrate, SSHSubstrate):
        cmd = (
            "set -euo pipefail && "
            f"cd {substrate.work_dir} && "
            f"test -d {base_problem_name} && "
            f"cp -R {base_problem_name} {new_problem_name}"
        )
        result = substrate._ssh_cmd(cmd, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(
                "Failed to copy remote problem directory: "
                f"stdout={result.stdout[-400:]} stderr={result.stderr[-400:]}"
            )
        return

    if isinstance(substrate, LocalSubstrate):
        src = Path(substrate.work_dir) / base_problem_name
        dst = Path(substrate.work_dir) / new_problem_name
        shutil.copytree(src, dst)
        return

    raise TypeError(f"Unsupported substrate type: {type(substrate)!r}")


def _problem_name_for_variant(base_problem_name: str, run_tag: str, spec: VariantSpec) -> str:
    label_suffix = _sanitize_label(spec.label)
    return f"{base_problem_name}_{run_tag}_{label_suffix}"


def _result_from_eval(
    spec: VariantSpec,
    problem_name: str,
    result: EvalResult,
) -> VariantResult:
    mean_runtime_us = None
    min_runtime_us = None
    if result.benchmark_times_us:
        mean_runtime_us = result.mean_runtime_us
        min_runtime_us = result.min_runtime_us

    aggregate_score = -1.0
    if result.valid and mean_runtime_us is not None and math.isfinite(mean_runtime_us):
        aggregate_score = -mean_runtime_us

    return VariantResult(
        label=spec.label,
        source_path=str(spec.source_path),
        problem_name=problem_name,
        valid=result.valid,
        passed_correctness=result.passed_correctness,
        aggregate_score=aggregate_score,
        mean_runtime_us=mean_runtime_us,
        min_runtime_us=min_runtime_us,
        compile_time_s=result.compile_time_s,
        test_time_s=result.test_time_s,
        benchmark_time_s=result.benchmark_time_s,
        failure_reasons=result.failure_reasons,
        raw_stdout_tail=result.raw_stdout[-2000:],
        raw_stderr_tail=result.raw_stderr[-1000:],
    )


def _evaluate_variant_mode(
    config: dict,
    spec: VariantSpec,
    problem_name: str,
    mode: str,
) -> VariantResult:
    substrate = make_substrate(config)
    source = spec.source_path.read_text()
    result = substrate.eval_kernel(source, problem_name, mode=mode)
    return _result_from_eval(spec, problem_name, result)


def _failed_result(
    spec: VariantSpec,
    problem_name: str,
    failure_reason: str,
) -> VariantResult:
    return VariantResult(
        label=spec.label,
        source_path=str(spec.source_path),
        problem_name=problem_name,
        valid=False,
        passed_correctness=False,
        aggregate_score=-1.0,
        mean_runtime_us=None,
        min_runtime_us=None,
        compile_time_s=0.0,
        test_time_s=0.0,
        benchmark_time_s=0.0,
        failure_reasons=[failure_reason],
        raw_stdout_tail="",
        raw_stderr_tail="",
    )


def _merge_test_and_benchmark(
    test_result: VariantResult,
    benchmark_result: VariantResult,
) -> VariantResult:
    combined_failures = test_result.failure_reasons + benchmark_result.failure_reasons
    return VariantResult(
        label=test_result.label,
        source_path=test_result.source_path,
        problem_name=test_result.problem_name,
        valid=test_result.valid and benchmark_result.valid,
        passed_correctness=test_result.passed_correctness,
        aggregate_score=benchmark_result.aggregate_score
        if test_result.valid and benchmark_result.valid
        else -1.0,
        mean_runtime_us=benchmark_result.mean_runtime_us,
        min_runtime_us=benchmark_result.min_runtime_us,
        compile_time_s=test_result.compile_time_s + benchmark_result.compile_time_s,
        test_time_s=test_result.test_time_s,
        benchmark_time_s=benchmark_result.benchmark_time_s,
        failure_reasons=combined_failures,
        raw_stdout_tail=(test_result.raw_stdout_tail + "\n" + benchmark_result.raw_stdout_tail)[-2000:],
        raw_stderr_tail=(test_result.raw_stderr_tail + "\n" + benchmark_result.raw_stderr_tail)[-1000:],
    )


def _parallel_eval_phase(
    config: dict,
    prepared_variants: list[tuple[VariantSpec, str]],
    mode: str,
    parallelism: int,
) -> list[VariantResult]:
    results: list[VariantResult] = []
    with ThreadPoolExecutor(max_workers=max(1, parallelism)) as executor:
        future_map = {
            executor.submit(
                _evaluate_variant_mode,
                config,
                spec,
                problem_name,
                mode,
            ): (spec, problem_name)
            for spec, problem_name in prepared_variants
        }
        for future in as_completed(future_map):
            spec, problem_name = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = _failed_result(spec, problem_name, str(exc))
            results.append(result)
            print(json.dumps(asdict(result), indent=2))
    return results


def _mark_baseline_improvements(results: list[VariantResult]) -> None:
    baseline = next((r for r in results if r.label == "baseline_seed"), None)
    if baseline is None or not baseline.valid or baseline.mean_runtime_us is None:
        return

    for result in results:
        if result.label == "baseline_seed":
            continue
        if result.valid and result.mean_runtime_us is not None:
            result.improves_baseline = result.mean_runtime_us < baseline.mean_runtime_us


def _summary_payload(results: list[VariantResult], run_dir: Path, run_tag: str) -> dict:
    valid_results = [r for r in results if r.valid and r.mean_runtime_us is not None]
    best = min(valid_results, key=lambda r: r.mean_runtime_us) if valid_results else None
    return {
        "run_tag": run_tag,
        "run_dir": str(run_dir),
        "total_variants": len(results),
        "valid_variants": len(valid_results),
        "best_label": best.label if best is not None else None,
        "best_mean_runtime_us": best.mean_runtime_us if best is not None else None,
        "results": [asdict(result) for result in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run manual kernel mutations with isolated evaluation sandboxes")
    parser.add_argument("--problem", required=True, help="Problem name, e.g. fp8_quant")
    parser.add_argument("--variant-dir", type=Path, required=True, help="Directory containing manual mutation .py files")
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "substrate.json", help="Substrate config JSON")
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--mode", choices=["test", "benchmark", "both"], default="both")
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    problem_dir = SCRIPT_DIR / args.problem
    if not problem_dir.exists():
        raise FileNotFoundError(f"Problem directory not found: {problem_dir}")

    config = _load_config(args.config)
    run_tag = args.run_name or str(int(time.time()))
    run_dir = SCRIPT_DIR / "manual_runs" / f"{args.problem}_{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    base_problem_name = _problem_dir_name(args.problem)
    variants = _load_variants(problem_dir, args.variant_dir, args.include_baseline)

    connection_probe = make_substrate(config)
    if not connection_probe.check_connection():
        raise RuntimeError("Substrate connection check failed.")

    prepared_variants: list[tuple[VariantSpec, str]] = []
    results_by_label: dict[str, VariantResult] = {}

    for spec in variants:
        problem_name = _problem_name_for_variant(base_problem_name, run_tag, spec)
        try:
            _prepare_problem_copy(config, base_problem_name, problem_name)
            prepared_variants.append((spec, problem_name))
        except Exception as exc:
            results_by_label[spec.label] = _failed_result(spec, problem_name, f"Preparation failed: {exc}")

    if args.mode == "both":
        test_results = _parallel_eval_phase(
            config=config,
            prepared_variants=prepared_variants,
            mode="test",
            parallelism=args.parallelism,
        )
        for result in test_results:
            results_by_label[result.label] = result

        benchmark_inputs = [
            (spec, problem_name)
            for spec, problem_name in prepared_variants
            if results_by_label.get(spec.label) is not None and results_by_label[spec.label].valid
        ]
        for spec, problem_name in benchmark_inputs:
            try:
                benchmark_result = _evaluate_variant_mode(
                    config=config,
                    spec=spec,
                    problem_name=problem_name,
                    mode="benchmark",
                )
                result = _merge_test_and_benchmark(results_by_label[spec.label], benchmark_result)
            except Exception as exc:
                result = _failed_result(spec, problem_name, str(exc))
            results_by_label[spec.label] = result
            print(json.dumps(asdict(result), indent=2))
    else:
        phase_results = _parallel_eval_phase(
            config=config,
            prepared_variants=prepared_variants,
            mode=args.mode,
            parallelism=args.parallelism,
        )
        for result in phase_results:
            results_by_label[result.label] = result

    results = sorted(results_by_label.values(), key=lambda item: item.label)

    _mark_baseline_improvements(results)
    payload = _summary_payload(results, run_dir, run_tag)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"Saved summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Run AlphaEvolve evolution on a Helion kernel target.

This is the top-level entry point. It:
1. Loads the target (task.json, seed, context)
2. Connects substrate and verifies it works
3. Runs AlphaEvolve's evolution controller
4. Collects telemetry throughout
5. Extracts the best kernel at the end

Usage:
    # Dry run (mock mutations, no LLM calls, no GPU)
    python helion_targets/run.py --problem fp8_quant --mode mock --generations 3

    # Real run with OpenAI on remote B200
    python helion_targets/run.py --problem fp8_quant --mode openai --generations 20

    # Real run with Gemini
    python helion_targets/run.py --problem fp8_quant --mode gemini --generations 20

    # Test substrate connectivity
    python helion_targets/run.py --problem fp8_quant --check-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ALPHAEVOLVE_DIR = REPO_ROOT / "alphaevolve"

# Add alphaevolve to path so we can import its modules
sys.path.insert(0, str(ALPHAEVOLVE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from substrate import make_substrate, EvalResult
from telemetry import TelemetryCollector


def check_substrate(config_path: Path, problem_name: str) -> bool:
    """Verify substrate can connect and run a basic eval."""
    config = json.loads(config_path.read_text())
    substrate = make_substrate(config)

    print(f"Checking substrate connection ({config.get('substrate', 'ssh')})...")
    if not substrate.check_connection():
        print("FAIL: Cannot connect to substrate.")
        return False
    print("OK: Connected.")

    # Quick test with the seed kernel
    target_dir = SCRIPT_DIR / problem_name
    seed_path = target_dir / "seed.py"
    if not seed_path.exists():
        print(f"FAIL: seed.py not found in {target_dir}")
        return False

    problem_dir_name = problem_name + "_py" if not problem_name.endswith("_py") else problem_name
    print(f"Running correctness test with seed kernel on {problem_dir_name}...")
    result = substrate.eval_kernel(seed_path.read_text(), problem_dir_name, mode="test")

    if result.passed_correctness:
        print(f"OK: Seed passes correctness ({result.test_time_s:.1f}s)")
    else:
        print(f"FAIL: Seed failed correctness: {result.failure_reasons}")
        return False

    return True


def run_evolution(args: argparse.Namespace) -> int:
    problem_name = args.problem
    target_dir = SCRIPT_DIR / problem_name
    problem_dir_name = problem_name + "_py" if not problem_name.endswith("_py") else problem_name

    # Load substrate
    config_path = args.substrate_config or (SCRIPT_DIR / "substrate.json")
    if not config_path.exists():
        print(f"Error: substrate config not found: {config_path}")
        return 1
    substrate_config = json.loads(config_path.read_text())
    substrate = make_substrate(substrate_config)

    # Verify task.json exists
    task_json_path = target_dir / "task.json"
    if not task_json_path.exists():
        print(f"Error: task.json not found in {target_dir}")
        return 1

    # Setup run directory
    run_name = args.run_name or f"{problem_name}_{args.mode}_{int(time.time())}"
    run_dir = SCRIPT_DIR / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Init telemetry
    telem = TelemetryCollector(
        problem_name=problem_name,
        substrate_type=substrate_config.get("substrate", "ssh"),
        out_dir=run_dir,
    )

    print(f"\n{'='*60}")
    print(f"  AlphaEvolve x Helion — {problem_name}")
    print(f"  Mode: {args.mode} | Generations: {args.generations}")
    print(f"  Parallel: {args.parallel_candidates} | Substrate: {substrate_config.get('substrate', 'ssh')}")
    print(f"  Run dir: {run_dir}")
    print(f"{'='*60}\n")

    # Build AlphaEvolve config
    from mvp.config import RunConfig

    if args.model:
        model_name = args.model
    elif args.mode == "openai":
        model_name = "gpt-5.4"
    elif args.mode == "gemini":
        model_name = "gemini-3-flash-latest"
    else:
        model_name = "mock"

    alphaevolve_config = RunConfig(
        mode=args.mode,
        model_name=model_name,
        openai_fast_model_name=args.fast_model,
        openai_slow_every=args.slow_every,
        openai_max_output_tokens=100000,  # Kernels need many tokens, xhigh reasoning produces more
        openai_request_timeout_s=1000.0,  # xhigh reasoning on gpt-5.4 can take 3-5min
        parallel_candidates=args.parallel_candidates,
        llm_concurrency=args.llm_concurrency,
        generations=args.generations,
        inspirations_k=args.inspirations,
        survivor_top_k=args.top_k,
        diversity_slots=args.diversity_slots,
        seed=args.seed,
        run_name=run_name,
    )

    # Wire up AlphaEvolve's event callback to our telemetry
    current_gen = [0]
    best_score_so_far = [-1.0]

    def event_callback(event: dict) -> None:
        etype = event.get("event_type", "")

        if etype == "generation_start":
            gen = int(event.get("generation", 0))
            current_gen[0] = gen
            telem.begin_generation(gen)

        elif etype == "slot_evaluated":
            valid = bool(event.get("valid", False))
            score = float(event.get("aggregate_score", -1.0))
            candidate_id = str(event.get("candidate_id", ""))

            # Determine failure type
            failure_type = None
            mutation_model = event.get("mutation_model", "")
            if not valid:
                if mutation_model == "failed_diff":
                    failure_type = "diff"
                elif mutation_model == "controller_error":
                    failure_type = "compile"
                else:
                    failure_type = "correctness"

            improved = valid and score > best_score_so_far[0]
            if improved:
                best_score_so_far[0] = score

            # We don't have runtime_us from the AlphaEvolve event directly,
            # but score = -runtime_us, so:
            runtime_us = -score if valid and score < 0 else float("inf")

            telem.record_candidate(
                valid=valid,
                improved=improved,
                score=score,
                runtime_us=runtime_us,
                eval_time_s=0.0,  # Not available from event
                failure_type=failure_type,
                candidate_id=candidate_id,
            )

        elif etype == "generation_end":
            pop_size = int(event.get("active_population", 0))
            gen_stats = telem.end_generation(pop_size)
            telem.print_status(current_gen[0])

    # Run AlphaEvolve with task_dir pointing to our target
    from mvp.controller import EvolutionController

    # If in mock mode but no mock diffs exist for this target,
    # the mock mutator will fail. That's expected — use a real LLM mode.
    tui = None
    tui_callback = event_callback
    if args.tui:
        try:
            from mvp.tui import EvolutionLiveTUI
            tui = EvolutionLiveTUI(
                mode=args.mode,
                model_name=model_name,
                total_generations=args.generations,
                parallel_candidates=args.parallel_candidates,
            )
            original_callback = event_callback
            def combined_callback(event):
                original_callback(event)
                tui.handle_event(event)
            tui_callback = combined_callback
            tui.start()
        except Exception as e:
            print(f"TUI unavailable: {e}, continuing without it")

    try:
        controller = EvolutionController(
            base_dir=ALPHAEVOLVE_DIR,
            config=alphaevolve_config,
            event_callback=tui_callback,
            task_dir=target_dir,
        )
        summary = controller.run()
    except Exception as e:
        print(f"\nEvolution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if tui is not None:
            tui.stop()

    # Write telemetry
    telem_path = telem.write_summary()

    # Extract best kernel
    best_source = None
    alphaevolve_run_dir = Path(summary.get("run_dir", ""))
    best_program_path = alphaevolve_run_dir / "best_program.py"
    if best_program_path.exists():
        best_source = best_program_path.read_text()
        # Also save in our run dir
        (run_dir / "best_kernel.py").write_text(best_source)

    # Final report
    print(f"\n{'='*60}")
    print(f"  EVOLUTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Problem:        {problem_name}")
    print(f"  Generations:    {args.generations}")
    print(f"  Total time:     {telem.data.elapsed_s:.0f}s")
    print(f"  Candidates:     {telem.data.total_candidates} ({telem.data.total_valid} valid)")
    print(f"  Success rate:   {telem.data.success_rate:.0%}")
    if telem.data.best_ever_runtime_us < float("inf"):
        print(f"  Best runtime:   {telem.data.best_ever_runtime_us:.2f} us")
        print(f"  Best gen:       {telem.data.best_ever_generation}")
    print(f"  Telemetry:      {telem_path}")
    if best_source:
        print(f"  Best kernel:    {run_dir / 'best_kernel.py'}")
    print(f"  AlphaEvolve:    {alphaevolve_run_dir}")
    print(f"{'='*60}\n")

    # Save combined summary
    combined = {
        "problem": problem_name,
        "mode": args.mode,
        "generations": args.generations,
        "alphaevolve_summary": summary,
        "telemetry_summary_path": str(telem_path),
        "best_kernel_path": str(run_dir / "best_kernel.py") if best_source else None,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(combined, indent=2))

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AlphaEvolve x Helion — Evolve GPU kernels"
    )
    parser.add_argument("--problem", required=True, help="Problem name (e.g. fp8_quant)")
    parser.add_argument("--mode", choices=["mock", "gemini", "openai"], default="mock")
    parser.add_argument("--model", default=None, help="LLM model override")
    parser.add_argument("--fast-model", default="gpt-5.4")
    parser.add_argument("--slow-every", type=int, default=4)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--parallel-candidates", type=int, default=3)
    parser.add_argument("--llm-concurrency", type=int, default=3)
    parser.add_argument("--inspirations", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--diversity-slots", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--substrate-config", type=Path, default=None)
    parser.add_argument("--tui", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Only check substrate, don't evolve")
    args = parser.parse_args()

    config_path = args.substrate_config or (SCRIPT_DIR / "substrate.json")

    if args.check_only:
        ok = check_substrate(config_path, args.problem)
        return 0 if ok else 1

    return run_evolution(args)


if __name__ == "__main__":
    sys.exit(main())

"""
Bridge between AlphaEvolve's generic evaluator and the substrate.

This script is called by AlphaEvolve via task.json's eval_command.
It reads a candidate file, sends it to the substrate, and prints
the JSON result to stdout.

Usage:
    python eval_kernel.py <candidate_file> <problem_name> [--config substrate.json]

Env vars (override config):
    SUBSTRATE_HOST   — SSH host
    SUBSTRATE_USER   — SSH user (default: ubuntu)
    SUBSTRATE_TYPE   — "ssh" or "local"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow importing from this directory
sys.path.insert(0, str(Path(__file__).parent))

from substrate import EvalResult, SSHSubstrate, LocalSubstrate, make_substrate


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a Helion kernel candidate")
    parser.add_argument("candidate_file", help="Path to candidate submission.py")
    parser.add_argument("problem_name", help="Problem directory name (e.g. fp8_quant_py)")
    parser.add_argument("--config", default=None, help="Path to substrate config JSON")
    parser.add_argument("--mode", default="both", choices=["test", "benchmark", "both"])
    args = parser.parse_args()

    # Read candidate source
    candidate_path = Path(args.candidate_file)
    if not candidate_path.exists():
        print(json.dumps({
            "valid": False,
            "aggregate_score": -1.0,
            "failure_reasons": [f"Candidate file not found: {args.candidate_file}"],
        }))
        return 0

    candidate_source = candidate_path.read_text()

    # Build substrate from config or env
    if args.config and Path(args.config).exists():
        config = json.loads(Path(args.config).read_text())
    else:
        config = {}

    # Env overrides
    if os.environ.get("SUBSTRATE_TYPE"):
        config["substrate"] = os.environ["SUBSTRATE_TYPE"]
    if os.environ.get("SUBSTRATE_HOST"):
        config["host"] = os.environ["SUBSTRATE_HOST"]
    if os.environ.get("SUBSTRATE_USER"):
        config["user"] = os.environ["SUBSTRATE_USER"]

    # Default to SSH if host is set, otherwise local
    if "substrate" not in config:
        config["substrate"] = "ssh" if "host" in config else "local"

    substrate = make_substrate(config)

    # Evaluate
    result = substrate.eval_kernel(candidate_source, args.problem_name, mode=args.mode)

    # Output JSON for AlphaEvolve
    print(json.dumps(result.to_evaluator_json()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

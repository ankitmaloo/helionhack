"""
Substrate abstraction — how and where kernels get evaluated.

Each substrate implements: upload candidate, run test, run benchmark, parse results.
Swap substrates without changing anything else.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalResult:
    valid: bool
    passed_correctness: bool
    benchmark_times_us: list[float] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)
    compile_time_s: float = 0.0
    test_time_s: float = 0.0
    benchmark_time_s: float = 0.0
    raw_stdout: str = ""
    raw_stderr: str = ""

    @property
    def mean_runtime_us(self) -> float:
        if not self.benchmark_times_us:
            return float("inf")
        return sum(self.benchmark_times_us) / len(self.benchmark_times_us)

    @property
    def min_runtime_us(self) -> float:
        if not self.benchmark_times_us:
            return float("inf")
        return min(self.benchmark_times_us)

    def to_evaluator_json(self) -> dict:
        """Format for AlphaEvolve's generic evaluator contract."""
        if not self.valid:
            return {
                "valid": False,
                "aggregate_score": -1.0,
                "metrics": {},
                "failure_reasons": self.failure_reasons,
            }
        # AlphaEvolve maximizes. Runtime should be minimized.
        # Use negative runtime as score so lower runtime = higher score.
        mean_us = self.mean_runtime_us
        score = -mean_us if mean_us < float("inf") else -1.0
        return {
            "valid": True,
            "aggregate_score": score,
            "metrics": {
                "mean_runtime_us": mean_us,
                "min_runtime_us": self.min_runtime_us,
                "correctness": 1.0 if self.passed_correctness else 0.0,
                "compile_time_s": self.compile_time_s,
                "test_time_s": self.test_time_s,
                "benchmark_time_s": self.benchmark_time_s,
                "num_benchmark_shapes": len(self.benchmark_times_us),
            },
            "failure_reasons": [],
        }


class Substrate(ABC):
    """Base class. Implement for your target hardware."""

    @abstractmethod
    def eval_kernel(
        self, candidate_source: str, problem_name: str, mode: str = "both"
    ) -> EvalResult:
        """
        Evaluate a candidate kernel.

        Args:
            candidate_source: Full Python source of submission.py
            problem_name: e.g. "fp8_quant_py"
            mode: "test", "benchmark", or "both"

        Returns:
            EvalResult with correctness and timing data.
        """
        ...

    @abstractmethod
    def check_connection(self) -> bool:
        ...


class SSHSubstrate(Substrate):
    """Evaluate kernels on a remote GPU machine over SSH."""

    def __init__(
        self,
        host: str,
        user: str = "ubuntu",
        venv_activate: str = "source ~/helion_env/bin/activate",
        work_dir: str = "~/reference-kernels/problems/helion",
        ssh_opts: str = "-o StrictHostKeyChecking=no -o ConnectTimeout=15",
        timeout_s: int = 300,
    ):
        self.host = host
        self.user = user
        self.venv_activate = venv_activate
        self.work_dir = work_dir
        self.ssh_opts = ssh_opts
        self.timeout_s = timeout_s

    def _ssh_cmd(self, remote_cmd: str, timeout: int | None = None) -> subprocess.CompletedProcess:
        t = timeout or self.timeout_s
        cmd = f"ssh {self.ssh_opts} {self.user}@{self.host} 'bash -c \"{remote_cmd}\"'"
        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=t
        )

    def _scp_to(self, local_path: str, remote_path: str) -> None:
        cmd = f"scp {self.ssh_opts} {local_path} {self.user}@{self.host}:{remote_path}"
        subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)

    def check_connection(self) -> bool:
        try:
            r = self._ssh_cmd("echo ok", timeout=10)
            return "ok" in r.stdout
        except Exception:
            return False

    def eval_kernel(
        self, candidate_source: str, problem_name: str, mode: str = "both"
    ) -> EvalResult:
        result = EvalResult(valid=False, passed_correctness=False)

        # Upload candidate
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="candidate_"
        ) as f:
            f.write(candidate_source)
            local_tmp = f.name

        remote_submission = f"{self.work_dir}/{problem_name}/submission.py"
        try:
            self._scp_to(local_tmp, remote_submission)
        except Exception as e:
            result.failure_reasons.append(f"SCP upload failed: {e}")
            return result
        finally:
            os.unlink(local_tmp)

        # Run correctness test
        if mode in ("test", "both"):
            t0 = time.time()
            try:
                r = self._ssh_cmd(
                    f"{self.venv_activate} && cd {self.work_dir} && python eval.py test {problem_name}/",
                    timeout=self.timeout_s,
                )
                result.test_time_s = time.time() - t0
                result.raw_stdout += r.stdout
                result.raw_stderr += r.stderr

                if "All tests passed" in r.stdout:
                    result.passed_correctness = True
                else:
                    # Extract failure info
                    for line in r.stdout.splitlines():
                        if "FAIL" in line:
                            result.failure_reasons.append(line.strip())
                    if r.returncode != 0 and not result.failure_reasons:
                        result.failure_reasons.append(
                            f"Test exited with code {r.returncode}"
                        )
                    if r.stderr.strip():
                        # Grab last few lines of stderr for compile errors etc.
                        stderr_tail = "\n".join(r.stderr.strip().splitlines()[-5:])
                        result.failure_reasons.append(f"stderr: {stderr_tail}")
                    return result

            except subprocess.TimeoutExpired:
                result.failure_reasons.append(
                    f"Test timed out after {self.timeout_s}s"
                )
                return result
            except Exception as e:
                result.failure_reasons.append(f"Test failed: {e}")
                return result

        # Run benchmark
        if mode in ("benchmark", "both"):
            t0 = time.time()
            try:
                r = self._ssh_cmd(
                    f"{self.venv_activate} && cd {self.work_dir} && python eval.py benchmark {problem_name}/",
                    timeout=self.timeout_s,
                )
                result.benchmark_time_s = time.time() - t0
                result.raw_stdout += r.stdout
                result.raw_stderr += r.stderr

                # Parse benchmark lines:
                #   Benchmark 0: 0.0073 ms (min=0.0073, max=0.0073)  {...}
                for line in r.stdout.splitlines():
                    line = line.strip()
                    if line.startswith("Benchmark") and "FAIL" not in line and ":" in line:
                        try:
                            # "Benchmark N: X.XXXX ms ..."
                            parts = line.split(":")
                            ms_part = parts[1].strip().split(" ")[0]
                            us = float(ms_part) * 1000.0
                            result.benchmark_times_us.append(us)
                        except (IndexError, ValueError):
                            pass
                    elif "FAIL" in line:
                        result.failure_reasons.append(line)

            except subprocess.TimeoutExpired:
                result.failure_reasons.append(
                    f"Benchmark timed out after {self.timeout_s}s"
                )
            except Exception as e:
                result.failure_reasons.append(f"Benchmark failed: {e}")

        # Valid if correctness passed (or wasn't tested) and no fatal errors
        if mode == "benchmark":
            result.valid = len(result.failure_reasons) == 0
        else:
            result.valid = result.passed_correctness and len(result.failure_reasons) == 0

        return result


class LocalSubstrate(Substrate):
    """Evaluate kernels on the local machine (must have GPU + helion)."""

    def __init__(
        self,
        work_dir: str,
        python: str = "python3",
        venv_activate: str | None = None,
        timeout_s: int = 300,
    ):
        self.work_dir = work_dir
        self.python = python
        self.venv_activate = venv_activate
        self.timeout_s = timeout_s

    def check_connection(self) -> bool:
        try:
            r = subprocess.run(
                [self.python, "-c", "import torch; print(torch.cuda.is_available())"],
                capture_output=True, text=True, timeout=10,
            )
            return "True" in r.stdout
        except Exception:
            return False

    def eval_kernel(
        self, candidate_source: str, problem_name: str, mode: str = "both"
    ) -> EvalResult:
        result = EvalResult(valid=False, passed_correctness=False)

        # Write candidate
        submission_path = Path(self.work_dir) / problem_name / "submission.py"
        submission_path.write_text(candidate_source)

        prefix = f"{self.venv_activate} && " if self.venv_activate else ""

        if mode in ("test", "both"):
            t0 = time.time()
            try:
                r = subprocess.run(
                    f"{prefix}cd {self.work_dir} && {self.python} eval.py test {problem_name}/",
                    shell=True, capture_output=True, text=True, timeout=self.timeout_s,
                )
                result.test_time_s = time.time() - t0
                result.raw_stdout += r.stdout
                result.raw_stderr += r.stderr
                if "All tests passed" in r.stdout:
                    result.passed_correctness = True
                else:
                    for line in r.stdout.splitlines():
                        if "FAIL" in line:
                            result.failure_reasons.append(line.strip())
                    return result
            except subprocess.TimeoutExpired:
                result.failure_reasons.append("Test timed out")
                return result

        if mode in ("benchmark", "both"):
            t0 = time.time()
            try:
                r = subprocess.run(
                    f"{prefix}cd {self.work_dir} && {self.python} eval.py benchmark {problem_name}/",
                    shell=True, capture_output=True, text=True, timeout=self.timeout_s,
                )
                result.benchmark_time_s = time.time() - t0
                result.raw_stdout += r.stdout
                for line in r.stdout.splitlines():
                    line = line.strip()
                    if line.startswith("Benchmark") and "FAIL" not in line and ":" in line:
                        try:
                            parts = line.split(":")
                            ms_part = parts[1].strip().split(" ")[0]
                            us = float(ms_part) * 1000.0
                            result.benchmark_times_us.append(us)
                        except (IndexError, ValueError):
                            pass
            except subprocess.TimeoutExpired:
                result.failure_reasons.append("Benchmark timed out")

        if mode == "benchmark":
            result.valid = len(result.failure_reasons) == 0
        else:
            result.valid = result.passed_correctness and len(result.failure_reasons) == 0

        return result


def make_substrate(config: dict) -> Substrate:
    """Factory from a config dict. Keeps run.py clean."""
    kind = config.get("substrate", "ssh")
    if kind == "ssh":
        return SSHSubstrate(
            host=config["host"],
            user=config.get("user", "ubuntu"),
            venv_activate=config.get("venv_activate", "source ~/helion_env/bin/activate"),
            work_dir=config.get("work_dir", "~/reference-kernels/problems/helion"),
            timeout_s=config.get("timeout_s", 300),
        )
    elif kind == "local":
        return LocalSubstrate(
            work_dir=config["work_dir"],
            python=config.get("python", "python3"),
            venv_activate=config.get("venv_activate"),
            timeout_s=config.get("timeout_s", 300),
        )
    else:
        raise ValueError(f"Unknown substrate: {kind}")

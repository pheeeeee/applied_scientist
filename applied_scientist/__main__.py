"""Entry point: python -m applied_scientist"""

from __future__ import annotations

import argparse

from applied_scientist.system import AppliedScientistSystem


def main():
    parser = argparse.ArgumentParser(description="Applied Scientist")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--task", required=True, help="Path to task adapter module")
    parser.add_argument("--n-gpus", type=int, default=None, help="Override GPU count")
    parser.add_argument("--job-runner", default=None, help="Override compute backend")
    parser.add_argument("--test-run", action="store_true",
                        help="Validate pipeline without full run")
    args = parser.parse_args()

    system = AppliedScientistSystem(
        config_path=args.config,
        task_module=args.task,
        n_gpus=args.n_gpus,
        job_runner_override=args.job_runner,
        test_run=args.test_run,
    )
    system.start()


if __name__ == "__main__":
    main()

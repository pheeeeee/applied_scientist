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
    parser.add_argument("--slack", action="store_true",
                        help="Use Slack for I/O (requires SLACK_* env vars)")
    parser.add_argument("--approve-plans", action="store_true",
                        help="Require human approval after PLAN.md before implementing")
    parser.add_argument("--approve-submit", action="store_true",
                        help="Require human approval before submitting to GPU")
    parser.add_argument("--approve-all", action="store_true",
                        help="Enable all human approval checkpoints")
    args = parser.parse_args()

    # --approve-all enables all individual flags
    approve_plans = args.approve_plans or args.approve_all
    approve_submit = args.approve_submit or args.approve_all

    system = AppliedScientistSystem(
        config_path=args.config,
        task_module=args.task,
        n_gpus=args.n_gpus,
        job_runner_override=args.job_runner,
        test_run=args.test_run,
        slack=args.slack,
        approve_plans=approve_plans,
        approve_submit=approve_submit,
    )
    system.start()


if __name__ == "__main__":
    main()

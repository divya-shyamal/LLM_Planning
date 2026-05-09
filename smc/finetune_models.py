#!/usr/bin/env python3
"""
Fine-tunes System-1 and System-2 maze planners using LoRA on Mistral-7B-Instruct-v0.2.

System-1: trained on gold plan outputs  (--system 0.0)
System-2: trained on A* verbalized search traces (--system 1.0)

Model paths produced here are the exact paths smc_experiment.py expects.
Run from repo root: python smc/finetune_models.py [--base_model ...] [--cache_dir ...]
"""

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# These paths are determined by train.py's naming convention:
# {output_dir}/{task}/{search_algo}_{method}_{decomp_style}_{level}_system_{system}_{n_train}_epoch_{epochs}_lr_{lr}_bs_{bs}
# With the training hyperparameters fixed below, these are the canonical paths.
SYS1_MODEL_PATH = os.path.join(
    REPO_ROOT,
    "models/maze/a_star_obstacles_sliding_task_system_0.0_3200_epoch_3_lr_0.0005_bs_2",
)
SYS2_MODEL_PATH = os.path.join(
    REPO_ROOT,
    "models/maze/a_star_obstacles_sliding_task_system_1.0_3200_epoch_3_lr_0.0005_bs_2",
)


def train(system_value: float, expected_path: str, label: str, base_model: str, cache_dir: str) -> None:
    if os.path.isdir(expected_path):
        print(f"[{label}] Model already exists at:\n  {expected_path}\n  Skipping training.")
        return

    print(f"\n{'=' * 60}")
    print(f"[{label}] Starting fine-tuning (system={system_value})")
    print(f"[{label}] Will save to: {expected_path}")
    print(f"{'=' * 60}\n")

    # PYTHONPATH must include repo root so that src_maze/data_utils.py can find tasks/maze.py
    # (the codebase has a hardcoded NAS path in data_utils.py; this override makes it portable)
    env = os.environ.copy()
    pythonpath = REPO_ROOT
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath

    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "src_maze", "train.py"),
        "--data_dir",    os.path.join(REPO_ROOT, "data", "maze"),
        "--output_dir",  os.path.join(REPO_ROOT, "models"),
        "--task",        "maze",
        "--level",       "task",
        "--search_algo", "a_star",
        "--method",      "obstacles",
        "--decomp_style","sliding",
        "--system",      str(system_value),
        "--num_epochs",  "3",
        "--learning_rate","0.0005",
        "--per_device_train_batch_size", "2",
        "--base_model",  base_model,
        "--cache_dir",   cache_dir,
    ]

    print(f"[{label}] Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)

    if result.returncode != 0:
        sys.exit(f"\n[{label}] Training process exited with code {result.returncode}.")

    if not os.path.isdir(expected_path):
        sys.exit(
            f"\n[{label}] Training finished (exit 0) but the expected model directory "
            f"was not created:\n  {expected_path}\n"
            f"Check that train.py's naming convention hasn't changed."
        )

    print(f"\n[{label}] Model saved to:\n  {expected_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune System-1 and System-2 maze planners.")
    parser.add_argument(
        "--base_model",
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="HuggingFace model ID or local path for the base model.",
    )
    parser.add_argument(
        "--cache_dir",
        default=os.path.join(REPO_ROOT, "cache"),
        help="Cache directory for downloaded model weights.",
    )
    parser.add_argument(
        "--sys1_only",
        action="store_true",
        help="Only fine-tune System-1 (skip System-2).",
    )
    parser.add_argument(
        "--sys2_only",
        action="store_true",
        help="Only fine-tune System-2 (skip System-1).",
    )
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(REPO_ROOT, "src_maze")):
        sys.exit(
            "ERROR: Cannot locate the repo root. "
            "Run this script from the repo root or from the smc/ subdirectory."
        )

    if not args.sys2_only:
        train(0.0, SYS1_MODEL_PATH, "System-1", args.base_model, args.cache_dir)
    if not args.sys1_only:
        train(1.0, SYS2_MODEL_PATH, "System-2", args.base_model, args.cache_dir)

    print("\nAll requested models are ready.")
    if not args.sys2_only:
        print(f"  System-1: {SYS1_MODEL_PATH}")
    if not args.sys1_only:
        print(f"  System-2: {SYS2_MODEL_PATH}")
    print("\nYou can now run: python smc/smc_experiment.py")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Minimal System-2 evaluation: load the model once, solve a maze, check the path.

Run from repo root:
    python smc/run_sys2.py [--n_test 1] [--maze_idx 0] [--max_new_tokens 2048]
"""

import argparse
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
_SRC_MAZE = os.path.join(REPO_ROOT, "src_maze")
if _SRC_MAZE not in sys.path:
    sys.path.insert(0, _SRC_MAZE)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from tasks.maze import Maze

SYS2_DEFAULT = os.path.join(
    REPO_ROOT,
    "models/maze/a_star_obstacles_sliding_task_system_1.0_3200_epoch_3_lr_0.0005_bs_2",
)

PROMPT_TEMPLATE = (
    "You are in a 2d maze of dimensions {l} and {w} and some of the cells have walls. "
    "The walls are placed in cells {walls}. "
    "Given a start and a goal state, your task is to generate the optimal plan as a "
    "sequence of actions. The optimal plan is one that has the minimum number of steps. "
    "The list of permissible actions that you can take at any given cell are {actions}. "
    "The optimal plan from {start} to {goal} is"
)

VALID_ACTIONS = {"left", "right", "up", "down"}


def build_prompt(maze: Maze, start: list) -> str:
    return PROMPT_TEMPLATE.format(
        l=maze.l, w=maze.w, walls=maze.walls,
        actions=maze.actions, start=start, goal=maze.goal,
    )


def extract_actions(output: str) -> list:
    """Extract the sequence of actions actually taken, in order."""
    matches = re.findall(r"Taking action '(.+?)' from state", output)
    return [a for a in matches if a in VALID_ACTIONS]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sys2_model", default=SYS2_DEFAULT)
    p.add_argument("--cache_dir", default=os.path.join(REPO_ROOT, "cache"))
    p.add_argument("--data_dir", default=os.path.join(REPO_ROOT, "data", "maze"))
    p.add_argument("--n_test", type=int, default=1, help="Number of mazes to solve")
    p.add_argument("--maze_idx", type=int, default=0, help="Start index into test.json")
    p.add_argument("--max_new_tokens", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.sys2_model):
        sys.exit(f"ERROR: System-2 model not found at {args.sys2_model}\n"
                 "Run  python smc/finetune_models.py  first.")

    print("Loading tokenizer and System-2 model ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.sys2_model, padding_side="left", add_eos_token=False
    )
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    base = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
    )
    base.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model = PeftModel.from_pretrained(base, args.sys2_model).to("cuda")
    model.eval()
    print("Model ready.\n")

    with open(os.path.join(args.data_dir, "test.json")) as f:
        raw = json.load(f)
    samples = raw[args.maze_idx: args.maze_idx + args.n_test]

    for i, s in enumerate(samples):
        maze = Maze(
            l=s["l"], w=s["w"], actions=s["actions"],
            start=s["start"], goal=s["goal"], walls=s["walls"],
            system1_plan=s["system1_plan"],
            system2_plan=s["system2_a_star_plan"],
            idx=s["idx"],
        )

        print(f"--- Maze {i + 1}/{len(samples)}  (dataset idx={maze.idx}) ---")
        print(f"  Grid : {maze.l}x{maze.w}")
        print(f"  Start: {maze.start}   Goal: {maze.goal}")
        print(f"  Walls: {maze.walls}")
        print(f"  Gold plan length: {s['plan_len']}")

        prompt = build_prompt(maze, maze.start)
        inputs = tokenizer([prompt], return_tensors="pt", padding=True).to("cuda")

        gen_kwargs = dict(
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if args.max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = args.max_new_tokens

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        n_generated = int((gen_ids != tokenizer.eos_token_id).sum())

        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        output = full_text[len(prompt):].strip()

        actions = extract_actions(output)

        # Re-simulate to get the path the model actually took
        state = list(maze.start)
        path = [list(state)]
        taken = []
        for action in actions:
            ns = maze.execute_action(state, action)
            if maze.is_valid_state(ns) and ns != state:
                state = list(ns)
                path.append(list(state))
                taken.append(action)
            if state == list(maze.goal):
                break

        parts = [f"start {maze.start}"] + [f"{a} [0, 0]" for a in taken]
        plan_str = " | ".join(parts)

        valid = maze.is_valid_plan(plan=plan_str, start=maze.start, goal=maze.goal)
        optimal = (
            maze.is_optimal_plan(plan=plan_str, check_validity=False,
                                  start=maze.start, goal=maze.goal)
            if valid else False
        )

        print(f"\n  Model output (first 500 chars):\n    {output[:500]}")
        print(f"\n  Extracted actions : {taken}")
        print(f"  Path taken        : {path}")
        print(f"  Plan length       : {len(taken)}  (gold: {s['plan_len']})")
        print(f"  Tokens generated  : {n_generated}")
        print(f"  Valid             : {valid}")
        print(f"  Optimal           : {optimal}")
        print()


if __name__ == "__main__":
    main()

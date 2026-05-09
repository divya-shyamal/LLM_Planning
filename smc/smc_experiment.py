#!/usr/bin/env python3
"""
SMC-based dynamic model routing for maze planning (Section 7 of project notes).

Assumes System-1 and System-2 models have already been fine-tuned by
smc/finetune_models.py and exist at their canonical paths. Exits immediately
with a clear error if either model directory is missing.

Run from repo root:
    python smc/smc_experiment.py [--args]
"""

import argparse
import copy
import json
import math
import os
import re
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Repo-root bootstrap: must happen before any local imports so that
# tasks/maze.py is findable regardless of the working directory.
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tasks.maze import Maze

# ---------------------------------------------------------------------------
# Canonical model paths (must match the names produced by finetune_models.py)
# ---------------------------------------------------------------------------
SYS1_DEFAULT = os.path.join(
    REPO_ROOT,
    "models/maze/a_star_obstacles_sliding_task_system_0.0_3200_epoch_3_lr_0.0005_bs_2",
)
SYS2_DEFAULT = os.path.join(
    REPO_ROOT,
    "models/maze/a_star_obstacles_sliding_task_system_1.0_3200_epoch_3_lr_0.0005_bs_2",
)

# ---------------------------------------------------------------------------
# Prompt template: identical to the one used during fine-tuning in data_utils.py
# so that the models see prompts in the distribution they were trained on.
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = (
    "You are in a 2d maze of dimensions {l} and {w} and some of the cells have walls. "
    "The walls are placed in cells {walls}. "
    "Given a start and a goal state, your task is to generate the optimal plan as a "
    "sequence of actions. The optimal plan is one that has the minimum number of steps. "
    "The list of permissible actions that you can take at any given cell are {actions}. "
    "The optimal plan from {start} to {goal} is"
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Particle:
    state: list               # current [row, col] in the maze
    history: List[str]        # committed action sequence so far
    model: str                # 'sys1' or 'sys2'
    log_weight: float         # log of unnormalized particle weight
    complete: bool            # True once state == goal
    states_explored: int      # cumulative LLM-call cost for this particle


# ---------------------------------------------------------------------------
# Maze helpers
# ---------------------------------------------------------------------------

def bfs_distance(maze: Maze, start: list, goal: list) -> int:
    """
    Shortest-path distance from start to goal via BFS on the maze graph.
    Returns -1 if the goal is unreachable.
    Uses the maze's own execute_action / is_valid_state so it respects walls.
    """
    if start == goal:
        return 0
    queue: deque = deque([(tuple(start), 0)])
    visited = {tuple(start)}
    while queue:
        (r, c), dist = queue.popleft()
        for action in maze.actions:
            ns = maze.execute_action([r, c], action)
            ns_t = tuple(ns)
            if ns_t not in visited and maze.is_valid_state(ns):
                if ns == goal:
                    return dist + 1
                visited.add(ns_t)
                queue.append((ns_t, dist + 1))
    return -1  # unreachable


def compute_reward(maze: Maze, state: list, model_type: str, alpha: float, C: float) -> float:
    """
    Difficulty-aware reward (Equation 3 of the proposal):
        r(x_t, theta_t) = -d*(x_t, sg) + alpha * A(theta_t, h(x_t, sg))

    Alignment term A:
        A(sys1, h) = -h          (penalise System-1 on hard sub-problems)
        A(sys2, h) = h - C       (reward System-2 for hard, penalise its cost C)

    A large negative value is returned when the goal is unreachable, which
    drives the particle's weight to zero during resampling.
    """
    d = bfs_distance(maze, state, maze.goal)
    if d < 0:
        return -1e6
    h = Maze.obstacles(state, maze.goal, maze.walls)
    alignment = -h if model_type == "sys1" else (h - C)
    return float(-d + alpha * alignment)


def apply_action(maze: Maze, state: list, action: str) -> list:
    """
    Apply one action to the maze. Per the task spec, invalid moves leave the
    agent in place (no exception is raised).
    """
    ns = maze.execute_action(state, action)
    return ns if maze.is_valid_state(ns) else state


def history_to_plan_str(start: list, history: List[str]) -> str:
    """
    Build a plan string in the format expected by Maze.is_valid_plan().
    The validator only reads the action token from each segment (not the
    state coordinate), so placeholder coordinates are safe.
    """
    parts = [f"start {start}"]
    parts += [f"{action} [0, 0]" for action in history]
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Model output parsing
# ---------------------------------------------------------------------------

VALID_ACTIONS = {"left", "right", "up", "down"}


def extract_action_sys1(output: str) -> Optional[str]:
    """
    System-1 output format (as trained in data_utils.py):
        <start system 1> start [r, c] | action [r, c] | ... <end system 1>

    Extracts the first committed action (the second pipe-separated segment).
    Returns None if the output is unparseable.
    """
    m = re.search(r"<start system 1>(.*?)(?:<end system 1>|$)", output, re.DOTALL)
    if not m:
        return None
    parts = m.group(1).strip().split(" | ")
    # parts[0] = "start [r, c]",  parts[1] = "action [r, c]", ...
    if len(parts) < 2:
        return None
    action = parts[1].split(" ")[0].strip()
    return action if action in VALID_ACTIONS else None


def extract_action_sys2(output: str) -> Optional[str]:
    """
    System-2 output format (verbalized A* trace):
        <start system 2> Moved to state [...] | ... | Taking action 'X' from state [...] | ...

    Extracts the first "Taking action" occurrence (the first committed move).
    Returns None if the trace contains no committed action.
    """
    matches = re.findall(r"Taking action '(.+?)' from state", output)
    for candidate in matches:
        if candidate in VALID_ACTIONS:
            return candidate
    return None


def count_states_sys2(output: str) -> int:
    """
    Counts 'Moved to state' tokens in a System-2 trace to estimate how many
    environment states the LLM explored (the #states-explored metric).
    Minimum 1 to account for the initial state.
    """
    return max(1, output.count("Moved to state"))


def extract_full_plan_sys1(output: str) -> List[str]:
    """Extract the complete action sequence from a System-1 output."""
    m = re.search(r"<start system 1>(.*?)(?:<end system 1>|$)", output, re.DOTALL)
    if not m:
        return []
    parts = m.group(1).strip().split(" | ")
    actions = [p.split(" ")[0].strip() for p in parts[1:]]
    return [a for a in actions if a in VALID_ACTIONS]


def extract_full_plan_sys2(output: str) -> List[str]:
    """Extract the complete committed action sequence from a System-2 trace."""
    matches = re.findall(r"Taking action '(.+?)' from state", output)
    return [a for a in matches if a in VALID_ACTIONS]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(maze: Maze, current_state: list) -> str:
    """
    Identical prompt format to the one used during fine-tuning (data_utils.py).
    current_state is the particle's present position, which may be an
    intermediate state (not the original s0).
    """
    return PROMPT_TEMPLATE.format(
        l=maze.l,
        w=maze.w,
        walls=maze.walls,
        actions=maze.actions,
        start=current_state,
        goal=maze.goal,
    )


# ---------------------------------------------------------------------------
# Batched generation
# ---------------------------------------------------------------------------

def _input_device(model) -> torch.device:
    """Return the device of the model's first parameter (handles device_map='auto')."""
    return next(model.parameters()).device


def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
) -> List[str]:
    """
    Run greedy generation for a batch of prompts and return the generated
    continuations (prompt prefix stripped).  Returns an empty list for an
    empty prompt list.
    """
    if not prompts:
        return []
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(_input_device(model))
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
        )
    decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    # Strip the prompt prefix and any trailing newline
    return [text[len(prompt):].split("\n")[0].strip() for text, prompt in zip(decoded, prompts)]


# ---------------------------------------------------------------------------
# SMC resampling
# ---------------------------------------------------------------------------

def _normalized_weights(particles: List[Particle]) -> np.ndarray:
    log_ws = np.array([p.log_weight for p in particles], dtype=np.float64)
    log_ws -= log_ws.max()              # numerical stability
    ws = np.exp(log_ws)
    return ws / ws.sum()


def effective_sample_size(particles: List[Particle]) -> float:
    ws = _normalized_weights(particles)
    return float(1.0 / np.sum(ws ** 2))


def systematic_resample(particles: List[Particle], rng: np.random.Generator) -> List[Particle]:
    """
    Systematic resampling (lower variance than multinomial).
    After resampling, all log-weights are reset to 0 (uniform).
    """
    M = len(particles)
    ws = _normalized_weights(particles)
    positions = (np.arange(M) + rng.uniform()) / M
    cumsum = np.cumsum(ws)
    indices = np.searchsorted(cumsum, positions)
    indices = np.clip(indices, 0, M - 1)
    resampled = []
    for i in indices:
        p = copy.deepcopy(particles[i])
        p.log_weight = 0.0              # reset to uniform
        resampled.append(p)
    return resampled


# ---------------------------------------------------------------------------
# Core SMC particle filter
# ---------------------------------------------------------------------------

def run_smc(
    maze: Maze,
    sys1_model,
    sys2_model,
    tokenizer,
    args,
    rng: np.random.Generator,
) -> dict:
    """
    Algorithm 2 from the proposal.

    Returns a dict with keys:
        valid, optimal, plan (list of actions), states_explored, complete_particles
    """
    M = args.M
    epsilon = args.epsilon
    alpha = args.alpha
    C = args.C
    alpha_temp = args.alpha_temp

    # ---- Initialise particles (Section 3.4, Step 0) ----
    particles: List[Particle] = []
    for _ in range(M):
        init_model = rng.choice(["sys1", "sys2"])
        particles.append(Particle(
            state=list(maze.start),
            history=[],
            model=init_model,
            log_weight=0.0,
            complete=(list(maze.start) == list(maze.goal)),
            states_explored=0,
        ))

    for _t in range(args.Tmax):
        if all(p.complete for p in particles):
            break

        # ---- Step 1: model transition ----
        for p in particles:
            if not p.complete and rng.random() < epsilon:
                p.model = "sys2" if p.model == "sys1" else "sys1"

        # ---- Steps 2–6: generate, simulate, reweight (batched per model) ----
        for model_type, model_obj in [("sys1", sys1_model), ("sys2", sys2_model)]:
            idxs = [i for i, p in enumerate(particles) if p.model == model_type and not p.complete]
            if not idxs:
                continue

            prompts = [build_prompt(maze, particles[i].state) for i in idxs]
            outputs = generate_batch(model_obj, tokenizer, prompts, args.max_new_tokens)

            for i, output in zip(idxs, outputs):
                p = particles[i]

                # Extract k=1 committed action
                action = extract_action_sys1(output) if model_type == "sys1" else extract_action_sys2(output)

                if action:
                    new_state = apply_action(maze, p.state, action)
                    if new_state != p.state:        # valid move only
                        p.history.append(action)
                        p.state = new_state

                # Track states explored (1 for sys1; count from sys2 trace)
                p.states_explored += 1 if model_type == "sys1" else count_states_sys2(output)

                if p.state == maze.goal:
                    p.complete = True

                # Reweight: accumulate log-reward
                r = compute_reward(maze, p.state, model_type, alpha, C)
                p.log_weight += r / alpha_temp

        # ---- Steps 7–8: normalise and conditionally resample ----
        ess = effective_sample_size(particles)
        if ess < M / 2:
            particles = systematic_resample(particles, rng)

    # ---- Step 20: extract best complete particle ----
    complete = [p for p in particles if p.complete]
    if complete:
        ws = _normalized_weights(complete)
        best = complete[int(np.argmax(ws))]
    else:
        # Fallback: particle geometrically closest to goal
        def fallback_key(p: Particle) -> float:
            d = bfs_distance(maze, p.state, maze.goal)
            return d if d >= 0 else 1e9
        best = min(particles, key=fallback_key)

    plan_str = history_to_plan_str(maze.start, best.history)
    valid = maze.is_valid_plan(plan=plan_str, start=maze.start, goal=maze.goal)
    optimal = (
        maze.is_optimal_plan(plan=plan_str, check_validity=False, start=maze.start, goal=maze.goal)
        if valid else False
    )

    return {
        "valid": valid,
        "optimal": optimal,
        "plan": best.history,
        "states_explored": sum(p.states_explored for p in particles),
        "complete_particles": len(complete),
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def run_single_model(
    maze: Maze,
    model,
    tokenizer,
    model_type: str,
    args,
) -> dict:
    """
    Single forward pass with one model (Single System-1 / Single System-2 baseline).
    """
    prompt = build_prompt(maze, maze.start)
    output = generate_batch(model, tokenizer, [prompt], args.max_new_tokens)[0]

    if model_type == "sys1":
        actions = extract_full_plan_sys1(output)
        states_explored = len(actions) if actions else 1
    else:
        actions = extract_full_plan_sys2(output)
        states_explored = count_states_sys2(output)

    plan_str = history_to_plan_str(maze.start, actions)
    valid = maze.is_valid_plan(plan=plan_str, start=maze.start, goal=maze.goal)
    optimal = (
        maze.is_optimal_plan(plan=plan_str, check_validity=False, start=maze.start, goal=maze.goal)
        if valid else False
    )
    return {
        "valid": valid,
        "optimal": optimal,
        "plan": actions,
        "states_explored": states_explored,
    }


def run_smc_fixed_model(
    maze: Maze,
    model,
    tokenizer,
    model_type: str,
    args,
    rng: np.random.Generator,
) -> dict:
    """
    SMC with M particles all locked to one model type (no switching).
    This is the "System-1 alone" / "System-2 alone" baseline from Section 3.7.

    With greedy decoding all particles at the same state produce the same output,
    so this degenerates to a single model run replicated M times — which is the
    point: it isolates the contribution of model-switching from particle diversity.
    """
    M = args.M
    alpha = args.alpha
    C = args.C
    alpha_temp = args.alpha_temp

    particles: List[Particle] = [
        Particle(
            state=list(maze.start),
            history=[],
            model=model_type,
            log_weight=0.0,
            complete=(list(maze.start) == list(maze.goal)),
            states_explored=0,
        )
        for _ in range(M)
    ]

    for _t in range(args.Tmax):
        if all(p.complete for p in particles):
            break

        idxs = [i for i, p in enumerate(particles) if not p.complete]
        if not idxs:
            break

        prompts = [build_prompt(maze, particles[i].state) for i in idxs]
        outputs = generate_batch(model, tokenizer, prompts, args.max_new_tokens)

        for i, output in zip(idxs, outputs):
            p = particles[i]
            action = extract_action_sys1(output) if model_type == "sys1" else extract_action_sys2(output)
            if action:
                new_state = apply_action(maze, p.state, action)
                if new_state != p.state:
                    p.history.append(action)
                    p.state = new_state
            p.states_explored += 1 if model_type == "sys1" else count_states_sys2(output)
            if p.state == maze.goal:
                p.complete = True
            r = compute_reward(maze, p.state, model_type, alpha, C)
            p.log_weight += r / alpha_temp

        ess = effective_sample_size(particles)
        if ess < M / 2:
            particles = systematic_resample(particles, rng)

    complete = [p for p in particles if p.complete]
    if complete:
        ws = _normalized_weights(complete)
        best = complete[int(np.argmax(ws))]
    else:
        def fallback_key(p: Particle) -> float:
            d = bfs_distance(maze, p.state, maze.goal)
            return d if d >= 0 else 1e9
        best = min(particles, key=fallback_key)

    plan_str = history_to_plan_str(maze.start, best.history)
    valid = maze.is_valid_plan(plan=plan_str, start=maze.start, goal=maze.goal)
    optimal = (
        maze.is_optimal_plan(plan=plan_str, check_validity=False, start=maze.start, goal=maze.goal)
        if valid else False
    )
    return {
        "valid": valid,
        "optimal": optimal,
        "plan": best.history,
        "states_explored": sum(p.states_explored for p in particles),
        "complete_particles": len(complete),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def print_metrics(results: List[dict], label: str) -> None:
    n = len(results)
    if n == 0:
        return
    validity = sum(r["valid"] for r in results) / n
    optimality = sum(r["optimal"] for r in results) / n
    avg_states = sum(r["states_explored"] for r in results) / n
    print(f"\n  {label}")
    print(f"    Plan validity:       {validity:.4f}  ({sum(r['valid'] for r in results)}/{n})")
    print(f"    Plan optimality:     {optimality:.4f}  ({sum(r['optimal'] for r in results)}/{n})")
    print(f"    Avg states explored: {avg_states:.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SMC dynamic model routing for maze planning (Section 7 of project notes)."
    )
    # Model paths
    parser.add_argument("--sys1_model", default=SYS1_DEFAULT,
                        help="Path to the fine-tuned System-1 model directory.")
    parser.add_argument("--sys2_model", default=SYS2_DEFAULT,
                        help="Path to the fine-tuned System-2 model directory.")
    parser.add_argument("--cache_dir", default=os.path.join(REPO_ROOT, "cache"),
                        help="HuggingFace cache directory.")
    # Data / output
    parser.add_argument("--data_dir", default=os.path.join(REPO_ROOT, "data", "maze"))
    parser.add_argument("--output_dir", default=os.path.join(REPO_ROOT, "output"))
    parser.add_argument("--n_test", type=int, default=-1,
                        help="Number of test samples to evaluate (-1 = all 400).")
    # SMC hyperparameters (Section 7.2)
    parser.add_argument("--M", type=int, default=10, help="Number of particles.")
    parser.add_argument("--Tmax", type=int, default=16,
                        help="Max SMC steps per problem (2x max plan length = 2*8).")
    parser.add_argument("--epsilon", type=float, default=0.15,
                        help="Model-switching probability per step.")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Alignment weight in difficulty-aware reward.")
    parser.add_argument("--C", type=float, default=2.0,
                        help="System-2 cost threshold (obstacle count).")
    parser.add_argument("--alpha_temp", type=float, default=1.0,
                        help="SMC temperature for reward-to-weight conversion.")
    parser.add_argument("--max_new_tokens", type=int, default=300,
                        help="Max tokens to generate per model call.")
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_baselines", action="store_true",
                        help="Skip baseline evaluation (faster).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Hard guard: both model directories must exist ----
    missing = []
    for label, path in [("System-1", args.sys1_model), ("System-2", args.sys2_model)]:
        if not os.path.isdir(path):
            missing.append(f"  {label}: {path}")
    if missing:
        sys.exit(
            "ERROR: The following fine-tuned model directories were not found:\n"
            + "\n".join(missing)
            + "\n\nRun  python smc/finetune_models.py  first to train the models."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- Load models ----
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.sys1_model, padding_side="left", add_eos_token=False
    )
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    print("Loading System-1 model ...")
    sys1_model = AutoModelForCausalLM.from_pretrained(
        args.sys1_model, cache_dir=args.cache_dir, device_map="auto"
    )
    sys1_model.resize_token_embeddings(len(tokenizer))
    sys1_model.eval()

    print("Loading System-2 model ...")
    sys2_model = AutoModelForCausalLM.from_pretrained(
        args.sys2_model, cache_dir=args.cache_dir, device_map="auto"
    )
    sys2_model.resize_token_embeddings(len(tokenizer))
    sys2_model.eval()

    # ---- Load test data ----
    test_path = os.path.join(args.data_dir, "test.json")
    with open(test_path) as f:
        raw = json.load(f)
    if args.n_test > 0:
        raw = raw[: args.n_test]

    mazes = [
        Maze(
            l=s["l"], w=s["w"], actions=s["actions"],
            start=s["start"], goal=s["goal"], walls=s["walls"],
            system1_plan=s["system1_plan"],
            system2_plan=s["system2_a_star_plan"],
            idx=s["idx"],
        )
        for s in raw
    ]
    print(f"Loaded {len(mazes)} test problems.\n")

    # ---- Evaluation loop ----
    smc_results: List[dict] = []
    single_sys1_results: List[dict] = []
    smc_sys1_results: List[dict] = []    # SMC locked to sys1
    smc_sys2_results: List[dict] = []    # SMC locked to sys2

    output_path = os.path.join(args.output_dir, "smc_results.json")

    for idx, maze in enumerate(mazes):
        print(f"[{idx + 1:3d}/{len(mazes)}] idx={maze.idx}  "
              f"start={maze.start}  goal={maze.goal}  "
              f"plan_len={len(maze.system1_plan.split(' | ')) - 1}")

        # -- SMC (our method) --
        r_smc = run_smc(maze, sys1_model, sys2_model, tokenizer, args, rng)
        r_smc["idx"] = maze.idx
        r_smc["gold_plan"] = maze.system1_plan
        smc_results.append(r_smc)
        print(f"         SMC        valid={r_smc['valid']}  optimal={r_smc['optimal']}  "
              f"states={r_smc['states_explored']}  complete_particles={r_smc['complete_particles']}")

        if not args.no_baselines:
            # -- Baseline 4: Single System-1 (one greedy forward pass) --
            r_s1 = run_single_model(maze, sys1_model, tokenizer, "sys1", args)
            r_s1["idx"] = maze.idx
            single_sys1_results.append(r_s1)

            # -- Baseline 1: M particles, all System-1 (no switching) --
            r_smc_s1 = run_smc_fixed_model(maze, sys1_model, tokenizer, "sys1", args, rng)
            r_smc_s1["idx"] = maze.idx
            smc_sys1_results.append(r_smc_s1)

            # -- Baseline 3: M particles, all System-2 (upper-bound on accuracy) --
            r_smc_s2 = run_smc_fixed_model(maze, sys2_model, tokenizer, "sys2", args, rng)
            r_smc_s2["idx"] = maze.idx
            smc_sys2_results.append(r_smc_s2)

            print(f"         Single-Sys1 valid={r_s1['valid']}  "
                  f"M-Sys1 valid={r_smc_s1['valid']}  "
                  f"M-Sys2 valid={r_smc_s2['valid']}")

        # -- Incremental save (safe against crashes) --
        checkpoint = {"args": vars(args), "smc": smc_results}
        if not args.no_baselines:
            checkpoint["single_sys1"]  = single_sys1_results
            checkpoint["smc_all_sys1"] = smc_sys1_results
            checkpoint["smc_all_sys2"] = smc_sys2_results
        with open(output_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

    # ---- Final metrics ----
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print_metrics(smc_results, f"SMC (M={args.M}, ε={args.epsilon}, C={args.C})  [our method]")
    if not args.no_baselines:
        print_metrics(single_sys1_results,
                      "Single System-1  (baseline 4: 1 greedy pass)")
        print_metrics(smc_sys1_results,
                      f"SMC all-System-1 (baseline 1: M={args.M} particles, no switching)")
        print_metrics(smc_sys2_results,
                      f"SMC all-System-2 (baseline 3: M={args.M} particles, upper bound)")
    print(f"\nFull results written to: {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Parse and summarize smc_results.json from smc_experiment.py.

Usage:
    python smc/parse_results.py output/smc_results.json
"""

import json
import sys
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(xs):
    return float(np.mean(xs)) if xs else float("nan")

def _median(xs):
    return float(np.median(xs)) if xs else float("nan")

def _std(xs):
    return float(np.std(xs)) if xs else float("nan")

def section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")

def row(label, val, fmt=".4f"):
    print(f"  {label:<50} {val:{fmt}}")

def row2(label, mn, med, fmt=".2f"):
    print(f"  {label:<50} mean={mn:{fmt}}  median={med:{fmt}}")

def row2s(label, mn, sd, fmt=".2f"):
    print(f"  {label:<50} mean={mn:{fmt}}  std={sd:{fmt}}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "output/smc_results.json"
    with open(path) as f:
        data = json.load(f)

    args   = data.get("args", {})
    M      = args.get("M", 10)
    smc    = data.get("smc", [])
    s1x    = data.get("system1x", [])
    n_smc  = len(smc)
    n_s1x  = len(s1x)

    print(f"\nResults file : {path}")
    print(f"SMC results  : {n_smc} mazes")
    print(f"System-1.x   : {n_s1x} mazes")
    print(f"M={M} particles  seed={args.get('seed')}  "
          f"Tmax={args.get('Tmax')}  epsilon={args.get('epsilon')}  "
          f"alpha={args.get('alpha')}  C={args.get('C')}")

    # =========================================================
    # 1. OVERALL ACCURACY
    # =========================================================
    section("1. OVERALL ACCURACY")

    smc_completion  = [r["completion_rate"] for r in smc]
    smc_validity    = [int(r["valid"]) for r in smc]
    s1x_validity    = [int(r["valid"]) for r in s1x]

    row2s("SMC avg particle completion rate (mean ± std)",
          _mean(smc_completion), _std(smc_completion))
    row("SMC validity rate (best-particle plan)",        _mean(smc_validity))
    row("System-1.x validity rate",                      _mean(s1x_validity))

    # =========================================================
    # 2. PLAN OPTIMALITY
    # =========================================================
    section("2. PLAN OPTIMALITY  (length-optimal vs BFS gold plan)")

    smc_optimal    = [int(r["optimal"]) for r in smc]
    smc_valid_only = [int(r["valid"] and not r["optimal"]) for r in smc]
    smc_invalid    = [int(not r["valid"]) for r in smc]

    s1x_optimal    = [int(r["optimal"]) for r in s1x]
    s1x_valid_only = [int(r["valid"] and not r["optimal"]) for r in s1x]
    s1x_invalid    = [int(not r["valid"]) for r in s1x]

    print("\n  SMC (best-particle plan):")
    row("    Proportion optimal (valid + length-optimal)", _mean(smc_optimal))
    row("    Proportion valid but not optimal",            _mean(smc_valid_only))
    row("    Proportion invalid",                          _mean(smc_invalid))

    print("\n  System-1.x:")
    row("    Proportion optimal (valid + length-optimal)", _mean(s1x_optimal))
    row("    Proportion valid but not optimal",            _mean(s1x_valid_only))
    row("    Proportion invalid",                          _mean(s1x_invalid))

    # =========================================================
    # 3. PLAN LENGTH
    # =========================================================
    section("3. PLAN LENGTH (steps in final chosen plan)")

    gold_lens   = [r["gold_plan_len"] for r in smc]
    smc_lens    = [r["plan_length"]   for r in smc]
    s1x_lens    = [r["plan_length"]   for r in s1x]
    smc_excess  = [r["plan_length"] - r["gold_plan_len"] for r in smc]
    s1x_excess  = [r["plan_length"] - r["gold_plan_len"] for r in s1x
                   if "gold_plan_len" in r]

    row2("Gold (BFS-optimal) plan length",          _mean(gold_lens),  _median(gold_lens),  ".1f")
    row2("SMC best-particle plan length",            _mean(smc_lens),   _median(smc_lens),   ".1f")
    row2("System-1.x plan length",                  _mean(s1x_lens),   _median(s1x_lens),   ".1f")
    row2("SMC excess steps over gold",               _mean(smc_excess), _median(smc_excess), ".1f")
    row2("System-1.x excess steps over gold",        _mean(s1x_excess), _median(s1x_excess), ".1f")

    # =========================================================
    # 4. TOKEN USAGE (per particle)
    # =========================================================
    section("4. TOKEN USAGE (per particle)")

    # SMC: flatten all individual particle token counts
    smc_pp_tok_gen = []
    smc_pp_tok_in  = []
    for r in smc:
        for pt in r.get("per_particle_tokens", []):
            smc_pp_tok_gen.append(pt["tokens_generated"])
            smc_pp_tok_in.append(pt["tokens_input"])

    s1x_tok_gen = [r["total_tokens_generated"] for r in s1x]
    s1x_tok_in  = [r["total_tokens_input"]     for r in s1x]

    print(f"\n  SMC  (distribution over all {len(smc_pp_tok_gen)} particle runs = {n_smc} mazes × {M} particles):")
    row2("    Tokens generated per particle",  _mean(smc_pp_tok_gen), _median(smc_pp_tok_gen), ".1f")
    row2("    Tokens input per particle",       _mean(smc_pp_tok_in),  _median(smc_pp_tok_in),  ".1f")

    print(f"\n  System-1.x  (per maze, single agent):")
    row2("    Tokens generated",               _mean(s1x_tok_gen), _median(s1x_tok_gen), ".1f")
    row2("    Tokens input",                   _mean(s1x_tok_in),  _median(s1x_tok_in),  ".1f")

    # =========================================================
    # 5. STATES EXPLORED (per particle)
    # =========================================================
    section("5. STATES EXPLORED (per particle)")

    # states_explored in SMC result is the sum over M particles; divide by M
    smc_states_pp = [r["states_explored"] / M for r in smc]
    s1x_states    = [r["states_explored"]     for r in s1x]

    row2("SMC states explored per particle (avg/median over mazes)",
         _mean(smc_states_pp), _median(smc_states_pp), ".1f")
    row2("System-1.x states explored per maze",
         _mean(s1x_states),    _median(s1x_states),    ".1f")

    # =========================================================
    # 6. SMC MODEL SWITCHING
    # =========================================================
    section("6. SMC MODEL SWITCHING")

    per_particle_switches = []   # total switches per particle (across all steps)
    all_switched          = []   # bool per particle-step
    all_model_used        = []   # "sys1" or "sys2" per particle-step
    all_stuck             = []   # bool per particle-step: state unchanged this step

    for r in smc:
        particle_switch_counts = defaultdict(int)
        particle_seen          = set()

        for step_log in r.get("steps", []):
            for plog in step_log["particles"]:
                pid = plog["particle_id"]
                particle_seen.add(pid)

                all_switched.append(plog["switched"])
                all_model_used.append(plog["model_used"])
                all_stuck.append(plog["state_before"] == plog["state_after"])

                if plog["switched"]:
                    particle_switch_counts[pid] += 1

        for pid in particle_seen:
            per_particle_switches.append(particle_switch_counts[pid])

    total_steps = len(all_model_used)
    sys1_steps  = sum(1 for m in all_model_used if m == "sys1")
    sys2_steps  = total_steps - sys1_steps

    row2("Switches per particle",
         _mean(per_particle_switches), _median(per_particle_switches), ".2f")
    row("Proportion of particle-steps with a switch",
        _mean(all_switched))
    row("Proportion of particle-steps using sys1",
        sys1_steps / total_steps if total_steps else float("nan"))
    row("Proportion of particle-steps using sys2",
        sys2_steps / total_steps if total_steps else float("nan"))
    row("Proportion of particle-steps where state did not change (proxy for stuck)",
        _mean(all_stuck))

    # =========================================================
    # 7. SMC TRAJECTORY QUALITY (best-of-trajectory)
    # =========================================================
    section("7. SMC TRAJECTORY QUALITY (best-of-trajectory selection)")

    all_fraction_kept  = []
    all_rewards        = []
    steps_to_complete  = []   # steps taken by each particle that completed

    for r in smc:
        particle_complete_step = {}   # pid -> first step index where complete=True

        for step_log in r.get("steps", []):
            t = step_log["step"]
            for plog in step_log["particles"]:
                pid = plog["particle_id"]

                if plog["fraction_kept"] is not None:
                    all_fraction_kept.append(plog["fraction_kept"])
                all_rewards.append(plog["reward"])

                if plog["complete"] and pid not in particle_complete_step:
                    particle_complete_step[pid] = t + 1  # 1-indexed step count

        for pid, s in particle_complete_step.items():
            steps_to_complete.append(s)

    row2("Fraction of generated trajectory kept (best-of-traj)",
         _mean(all_fraction_kept), _median(all_fraction_kept))
    row2("Reward per particle-step",
         _mean(all_rewards), _median(all_rewards))
    row2("Steps to completion (complete particles only)",
         _mean(steps_to_complete), _median(steps_to_complete), ".1f")
    row("Overall particle completion proportion (check vs section 1)",
        len(steps_to_complete) / (n_smc * M) if n_smc else float("nan"))

    # =========================================================
    # 8. SMC REVISIT PENALTY
    # =========================================================
    section("8. SMC REVISIT PENALTY")

    all_revisit = []
    for r in smc:
        for step_log in r.get("steps", []):
            for plog in step_log["particles"]:
                all_revisit.append(int(plog["revisit_penalty_applied"]))

    row("Proportion of particle-steps with revisit penalty applied", _mean(all_revisit))

    # =========================================================
    # 9. SYSTEM-1.x DECOMPOSITION
    # =========================================================
    section("9. SYSTEM-1.X DECOMPOSITION")

    subgoal_counts     = []
    sys1_sg_counts     = []
    sys2_sg_counts     = []
    fallback_count     = 0

    for r in s1x:
        sgs  = r.get("sub_goals_parsed", [])
        n_sg = len(sgs)
        subgoal_counts.append(n_sg)

        n_sys1 = sum(1 for sg in sgs if sg[2] == 1)
        sys1_sg_counts.append(n_sys1)
        sys2_sg_counts.append(n_sg - n_sys1)

        # Fallback: single subgoal, sys2, covers the whole problem
        if (n_sg == 1 and sgs[0][2] == 2
                and sgs[0][0] == str(r.get("start"))
                and sgs[0][1] == str(r.get("goal"))):
            fallback_count += 1

    total_sgs  = sum(subgoal_counts)
    total_sys1 = sum(sys1_sg_counts)

    row2("Subgoals per maze",
         _mean(subgoal_counts), _median(subgoal_counts), ".1f")
    row2("Sys1 subgoals per maze",
         _mean(sys1_sg_counts), _median(sys1_sg_counts), ".1f")
    row2("Sys2 subgoals per maze",
         _mean(sys2_sg_counts), _median(sys2_sg_counts), ".1f")
    row("Proportion of subgoals assigned to sys1",
        total_sys1 / total_sgs if total_sgs else float("nan"))
    row("Proportion of subgoals assigned to sys2",
        1 - total_sys1 / total_sgs if total_sgs else float("nan"))
    row("Controller parse fallback rate",
        fallback_count / n_s1x if n_s1x else float("nan"))

    # =========================================================
    # 10. DIFFICULTY BREAKDOWN (by gold plan length)
    # =========================================================
    section("10. DIFFICULTY BREAKDOWN (by gold_plan_len)")

    smc_by_len = defaultdict(list)
    s1x_by_len = defaultdict(list)
    for r in smc:
        smc_by_len[r["gold_plan_len"]].append(r)
    for r in s1x:
        if "gold_plan_len" in r:
            s1x_by_len[r["gold_plan_len"]].append(r)

    all_lens = sorted(set(list(smc_by_len) + list(s1x_by_len)))
    hdr = (f"  {'len':>4}  {'N':>3}  "
           f"{'SMC compl':>10}  {'SMC valid':>10}  {'S1x valid':>10}  "
           f"{'SMC opt':>8}  {'S1x opt':>8}")
    print(f"\n{hdr}")
    print(f"  {'─'*4}  {'─'*3}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}")
    for gl in all_lens:
        sr  = smc_by_len.get(gl, [])
        xr  = s1x_by_len.get(gl, [])
        n   = max(len(sr), len(xr))
        smc_comp = _mean([r["completion_rate"] for r in sr]) if sr else float("nan")
        smc_val  = _mean([int(r["valid"])       for r in sr]) if sr else float("nan")
        s1x_val  = _mean([int(r["valid"])       for r in xr]) if xr else float("nan")
        smc_opt  = _mean([int(r["optimal"])     for r in sr]) if sr else float("nan")
        s1x_opt  = _mean([int(r["optimal"])     for r in xr]) if xr else float("nan")
        print(f"  {gl:>4}  {n:>3}  "
              f"{smc_comp:>10.3f}  {smc_val:>10.3f}  {s1x_val:>10.3f}  "
              f"{smc_opt:>8.3f}  {s1x_opt:>8.3f}")

    # =========================================================
    # 11. HEAD-TO-HEAD
    # =========================================================
    section("11. HEAD-TO-HEAD (per maze, validity)")

    if n_smc == n_s1x:
        n = n_smc
        both   = sum(s and x for s, x in zip(smc_validity, s1x_validity))
        smc_o  = sum(s and not x for s, x in zip(smc_validity, s1x_validity))
        s1x_o  = sum(not s and x for s, x in zip(smc_validity, s1x_validity))
        nei    = sum(not s and not x for s, x in zip(smc_validity, s1x_validity))
        print(f"\n  Both valid:           {both:3d} / {n}  ({both/n:.3f})")
        print(f"  SMC only valid:       {smc_o:3d} / {n}  ({smc_o/n:.3f})")
        print(f"  System-1.x only:      {s1x_o:3d} / {n}  ({s1x_o/n:.3f})")
        print(f"  Neither valid:        {nei:3d} / {n}  ({nei/n:.3f})")
    else:
        print(f"\n  Cannot compare: SMC has {n_smc} results, System-1.x has {n_s1x}.")

    print()


if __name__ == "__main__":
    main()

"""Benchmark Brogue rollout throughput across environment-count sweeps."""

import argparse
import os
import time

import numpy as np

from broguegym.actions import Action, ActionKind
from broguegym.brogue import BrogueBackend


def find_explore_action() -> Action:
    """Return an EXPLORE action for benchmarking."""
    return Action(kind=ActionKind.EXPLORE)


def default_env_sweep(max_envs: int | None = None) -> list[int]:
    """Return a broad sweep around powers of two and CPU-count multiples."""

    cpu_count = os.cpu_count() or 1
    ceiling = max_envs if max_envs is not None else min(max(64, cpu_count * 4), 256)
    if ceiling < 1:
        msg = "max_envs must be at least 1"
        raise ValueError(msg)

    envs: set[int] = set()
    n = 1
    while n <= ceiling:
        envs.add(n)
        n *= 2
    for multiplier in (1, 2, 4):
        n = cpu_count * multiplier
        if n <= ceiling:
            envs.add(n)
    return sorted(envs)


def bench_brogue_backend(
    num_envs: int,
    steps_per_env: int,
    *,
    warmup_steps: int = 5,
    seed: int = 42,
) -> dict[str, float | int]:
    """Run one sweep point for process-isolated Brogue workers."""

    if num_envs < 1:
        msg = "num_envs must be at least 1"
        raise ValueError(msg)
    if steps_per_env < 1:
        msg = "steps_per_env must be at least 1"
        raise ValueError(msg)
    if warmup_steps < 0:
        msg = "warmup_steps must be non-negative"
        raise ValueError(msg)

    explore = find_explore_action()
    actions = [explore] * num_envs
    seeds = np.arange(seed, seed + num_envs, dtype=np.uint64)

    with BrogueBackend(num_envs) as backend:
        backend.reset_many(seed=seeds)
        for _ in range(warmup_steps):
            results = backend.step_many(actions)
            for env_id, result in enumerate(results):
                if result.terminated:
                    backend.reset(env_id, seed=np.array([seeds[env_id]], dtype=np.uint64))

        total_steps = 0
        t0 = time.perf_counter()
        for _ in range(steps_per_env):
            results = backend.step_many(actions)
            total_steps += num_envs
            for env_id, result in enumerate(results):
                if result.terminated:
                    backend.reset(env_id, seed=np.array([seeds[env_id]], dtype=np.uint64))
        elapsed = time.perf_counter() - t0

    return {
        "num_envs": num_envs,
        "steps_per_env": steps_per_env,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "elapsed": elapsed,
        "steps_per_sec": total_steps / elapsed,
        "batch_ms": elapsed * 1000.0 / steps_per_env,
        "env_step_us": elapsed * 1_000_000.0 / total_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Brogue rollout throughput")
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Measured steps per env for each sweep point",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Untimed warmup steps per env before each measurement",
    )
    parser.add_argument(
        "--envs",
        type=int,
        nargs="+",
        default=None,
        help="Explicit env-count sweep. Defaults to powers of two plus CPU-count multiples.",
    )
    parser.add_argument(
        "--max-envs",
        type=int,
        default=None,
        help="Maximum env count for the default sweep",
    )
    args = parser.parse_args()

    envs = sorted(set(args.envs)) if args.envs is not None else default_env_sweep(args.max_envs)

    print(f"{'='*86}")
    print(
        "Brogue Rollout Sweep  "
        f"(steps/env={args.steps}, warmup/env={args.warmup_steps}, cpu={os.cpu_count() or 1})",
    )
    print(f"{'='*86}")
    print(
        f"{'envs':>6}  {'total':>10}  {'elapsed':>9}  "
        f"{'steps/sec':>12}  {'batch ms':>10}  {'env-step us':>12}",
    )
    print(f"{'-'*86}")

    best: dict[str, float | int] | None = None
    for n in envs:
        result = bench_brogue_backend(
            n,
            args.steps,
            warmup_steps=args.warmup_steps,
        )
        if best is None or result["steps_per_sec"] > best["steps_per_sec"]:
            best = result
        print(
            f"{int(result['num_envs']):>6}  "
            f"{int(result['total_steps']):>10,}  "
            f"{float(result['elapsed']):>8.2f}s  "
            f"{float(result['steps_per_sec']):>12,.0f}  "
            f"{float(result['batch_ms']):>10.2f}  "
            f"{float(result['env_step_us']):>12.1f}",
        )

    if best is not None:
        print(f"{'-'*86}")
        print(
            "best throughput: "
            f"{float(best['steps_per_sec']):,.0f} steps/sec at {int(best['num_envs'])} envs",
        )
    print(f"{'='*86}")


if __name__ == "__main__":
    main()

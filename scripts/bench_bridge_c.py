"""Benchmark the Brogue C bridge directly, outside the Python backend."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from broguegym.brogue import _default_data_dir, _default_library_path


C_SOURCE = r"""
#include <dlfcn.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef void (*set_data_dir_fn)(const char *path);
typedef size_t (*observation_size_fn)(void);
typedef int (*reset_fn)(uint64_t seed, void *out);
typedef int (*step_fn)(long key, int control, int shift, void *out);
typedef void (*close_fn)(void);
typedef const char *(*last_error_fn)(void);

typedef struct bridge_api {
    set_data_dir_fn set_data_dir;
    observation_size_fn observation_size;
    reset_fn reset;
    step_fn step;
    close_fn close;
    last_error_fn last_error;
} bridge_api;

static double now_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        exit(2);
    }
    return (double) ts.tv_sec + (double) ts.tv_nsec / 1000000000.0;
}

static void *must_symbol(void *handle, const char *name) {
    dlerror();
    void *symbol = dlsym(handle, name);
    const char *error = dlerror();
    if (error != NULL || symbol == NULL) {
        fprintf(stderr, "missing symbol %s: %s\n", name, error ? error : "null");
        exit(2);
    }
    return symbol;
}

static int scaled_count(int base, int percent) {
    long count = ((long) base * (long) percent + 99L) / 100L;
    return count < 1L ? 1 : (int) count;
}

static int terminated(const void *observation, size_t observation_size) {
    const uint64_t *program_state = (const uint64_t *) (
        (const char *) observation + observation_size - 8 * sizeof(uint64_t)
    );
    return program_state[1] != 0;
}

static void benchmark_steps(const bridge_api *api,
                            const char *label,
                            long key,
                            int control,
                            int shift,
                            int base_steps,
                            int base_warmup,
                            int percent,
                            uint64_t seed,
                            void *observation,
                            size_t observation_size) {
    int steps = scaled_count(base_steps, percent);
    int warmup = scaled_count(base_warmup, percent);
    int measured_resets = 0;
    int measured_invalid = 0;
    int warmup_resets = 0;
    int rc = api->reset(seed, observation);
    if (rc != 0) {
        fprintf(stderr, "%s reset failed: %s\n", label, api->last_error());
        exit(2);
    }

    for (int i = 0; i < warmup; i++) {
        rc = api->step(key, control, shift, observation);
        if (rc < 0) {
            fprintf(stderr, "%s warmup step failed at %d: %s\n", label, i, api->last_error());
            exit(2);
        }
        if (terminated(observation, observation_size)) {
            warmup_resets++;
            rc = api->reset(seed + (uint64_t) warmup_resets, observation);
            if (rc != 0) {
                fprintf(stderr, "%s warmup reset failed: %s\n", label, api->last_error());
                exit(2);
            }
        }
    }

    double start = now_seconds();
    for (int i = 0; i < steps; i++) {
        rc = api->step(key, control, shift, observation);
        if (rc < 0) {
            fprintf(stderr, "%s step failed at %d: %s\n", label, i, api->last_error());
            exit(2);
        }
        measured_invalid += (rc == 1);
        if (terminated(observation, observation_size)) {
            measured_resets++;
            rc = api->reset(seed + (uint64_t) warmup_resets + (uint64_t) measured_resets, observation);
            if (rc != 0) {
                fprintf(stderr, "%s measured reset failed: %s\n", label, api->last_error());
                exit(2);
            }
        }
    }
    double elapsed = now_seconds() - start;
    printf("%-14s  %8d  %9.4fs  %12.0f  %11.2f  %6d  %7d\n",
           label,
           steps,
           elapsed,
           (double) steps / elapsed,
           elapsed * 1000000.0 / (double) steps,
           measured_resets,
           measured_invalid);
    api->close();
}

static void benchmark_resets(const bridge_api *api,
                             int base_resets,
                             int percent,
                             uint64_t seed,
                             void *observation) {
    int resets = scaled_count(base_resets, percent);
    double start = now_seconds();
    for (int i = 0; i < resets; i++) {
        int rc = api->reset(seed + (uint64_t) i, observation);
        if (rc != 0) {
            fprintf(stderr, "reset bench failed at %d: %s\n", i, api->last_error());
            exit(2);
        }
    }
    double elapsed = now_seconds() - start;
    printf("%-14s  %8d  %9.4fs  %12.0f  %11.2f  %6d  %7d\n",
           "reset",
           resets,
           elapsed,
           (double) resets / elapsed,
           elapsed * 1000000.0 / (double) resets,
           0,
           0);
    api->close();
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s LIBRARY_PATH DATA_DIR PERCENT\n", argv[0]);
        return 2;
    }

    const char *library_path = argv[1];
    const char *data_dir = argv[2];
    int percent = atoi(argv[3]);
    if (percent < 1) {
        fprintf(stderr, "percent must be at least 1\n");
        return 2;
    }

    void *handle = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
    if (handle == NULL) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 2;
    }

    bridge_api api;
    api.set_data_dir = (set_data_dir_fn) must_symbol(handle, "brh_set_data_dir");
    api.observation_size = (observation_size_fn) must_symbol(handle, "brh_observation_size");
    api.reset = (reset_fn) must_symbol(handle, "brh_reset");
    api.step = (step_fn) must_symbol(handle, "brh_step");
    api.close = (close_fn) must_symbol(handle, "brh_close");
    api.last_error = (last_error_fn) must_symbol(handle, "brh_last_error");

    api.set_data_dir(data_dir);
    size_t observation_size = api.observation_size();
    void *observation = calloc(1, observation_size);
    if (observation == NULL) {
        perror("calloc observation");
        return 2;
    }

    printf("Brogue C Bridge Direct Benchmark\n");
    printf("library: %s\n", library_path);
    printf("data dir: %s\n", data_dir);
    printf("observation: %zu bytes\n", observation_size);
    printf("scale: %d%%\n", percent);
    printf("\n");
    printf("%-14s  %8s  %10s  %12s  %11s  %6s  %7s\n",
           "case",
           "steps",
           "elapsed",
           "steps/sec",
           "us/step",
           "resets",
           "invalid");
    printf("--------------------------------------------------------------------------------\n");

    benchmark_steps(&api, "invalid-key", '!', 0, 0, 20000, 1000, percent, 1, observation, observation_size);
    benchmark_steps(&api, "rest", 'z', 0, 0, 5000, 200, percent, 1001, observation, observation_size);
    benchmark_steps(&api, "search", 's', 0, 0, 5000, 200, percent, 2001, observation, observation_size);
    benchmark_steps(&api, "explore", 'x', 0, 0, 1000, 50, percent, 3001, observation, observation_size);
    benchmark_steps(&api, "fast-explore", 'x', 1, 0, 1000, 50, percent, 4001, observation, observation_size);
    benchmark_resets(&api, 200, percent, 5001, observation);

    free(observation);
    dlclose(handle);
    return 0;
}
"""


def compile_harness(cc: str, binary_path: Path) -> None:
    """Compile the embedded native benchmark harness."""

    command = [
        cc,
        "-O3",
        "-Wall",
        "-Wextra",
        "-x",
        "c",
        "-o",
        str(binary_path),
        "-",
    ]
    if sys.platform.startswith("linux"):
        command.append("-ldl")
    subprocess.run(command, input=C_SOURCE, text=True, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile and run a direct C benchmark of the Brogue bridge API",
    )
    parser.add_argument(
        "--cc",
        default=os.environ.get("CC", "cc"),
        help="C compiler command to use",
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=_default_library_path(),
        help="Path to libbruhogue_brogue",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_default_data_dir(),
        help="Brogue data directory containing keymap/assets",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=100,
        help="Percentage of the default sample counts to run",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=None,
        help="Optional output path for the compiled harness",
    )
    args = parser.parse_args()

    if args.scale < 1:
        msg = "--scale must be at least 1"
        raise ValueError(msg)
    if not args.library.is_file():
        msg = f"Brogue bridge library does not exist: {args.library}"
        raise FileNotFoundError(msg)
    if not args.data_dir.is_dir():
        msg = f"Brogue data directory does not exist: {args.data_dir}"
        raise FileNotFoundError(msg)

    if args.binary is not None:
        binary_path = args.binary
        compile_harness(args.cc, binary_path)
        subprocess.run(
            [
                str(binary_path),
                str(args.library),
                str(args.data_dir),
                str(args.scale),
            ],
            check=True,
        )
        return

    with tempfile.TemporaryDirectory(prefix="broguegym-bridge-bench-") as temp_dir:
        binary_path = Path(temp_dir) / "bench_bridge_c"
        compile_harness(args.cc, binary_path)
        subprocess.run(
            [
                str(binary_path),
                str(args.library),
                str(args.data_dir),
                str(args.scale),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()

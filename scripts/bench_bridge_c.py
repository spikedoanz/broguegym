"""Benchmark the Brogue C bridge directly, outside the Python backend."""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from broguegym.brogue import (
    _BRIDGE_ABI_VERSION,
    _CObservation,
    _INVENTORY_SIZE,
    _INVENTORY_STR_LENGTH,
    _MAP_COLS,
    _MAP_ROWS,
    _SCREEN_COLS,
    _SCREEN_ROWS,
    _default_data_dir,
    _default_library_path,
)


C_SOURCE = r"""
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef void (*set_data_dir_fn)(const char *path);
typedef uint32_t (*abi_version_fn)(void);
typedef size_t (*observation_size_fn)(void);
typedef int (*dimension_fn)(void);
typedef int (*reset_fn)(uint64_t seed, void *out);
typedef int (*step_fn)(long key, int control, int shift, void *out);
typedef void (*close_fn)(void);
typedef const char *(*last_error_fn)(void);

typedef struct bridge_api {
    set_data_dir_fn set_data_dir;
    abi_version_fn abi_version;
    observation_size_fn observation_size;
    dimension_fn screen_cols;
    dimension_fn screen_rows;
    dimension_fn map_cols;
    dimension_fn map_rows;
    dimension_fn inventory_size;
    dimension_fn inventory_str_length;
    reset_fn reset;
    step_fn step;
    close_fn close;
    last_error_fn last_error;
} bridge_api;

typedef struct expected_abi {
    uint32_t abi_version;
    size_t observation_size;
    int screen_cols;
    int screen_rows;
    int map_cols;
    int map_rows;
    int inventory_size;
    int inventory_str_length;
    size_t program_state_offset;
} expected_abi;

typedef struct trial_result {
    int steps;
    double elapsed;
    int terminated;
    int invalid;
} trial_result;

static volatile unsigned int copy_checksum_sink = 0;

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

static int double_compare(const void *left, const void *right) {
    double a = *(const double *) left;
    double b = *(const double *) right;
    if (a < b) {
        return -1;
    }
    if (a > b) {
        return 1;
    }
    return 0;
}

static double median_sample(const double *sorted, int count) {
    if (count % 2 == 1) {
        return sorted[count / 2];
    }
    return (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0;
}

static double p95_sample(const double *sorted, int count) {
    int index = (95 * count + 99) / 100 - 1;
    if (index < 0) {
        index = 0;
    }
    if (index >= count) {
        index = count - 1;
    }
    return sorted[index];
}

static uint64_t parse_u64_arg(const char *name, const char *value) {
    char *end = NULL;
    uint64_t parsed = strtoull(value, &end, 10);
    if (end == value || *end != '\0') {
        fprintf(stderr, "invalid %s: %s\n", name, value);
        exit(2);
    }
    return parsed;
}

static int parse_int_arg(const char *name, const char *value) {
    uint64_t parsed = parse_u64_arg(name, value);
    if (parsed > 2147483647ULL) {
        fprintf(stderr, "%s is too large: %s\n", name, value);
        exit(2);
    }
    return (int) parsed;
}

static size_t parse_size_arg(const char *name, const char *value) {
    return (size_t) parse_u64_arg(name, value);
}

static void validate_bridge_abi(const bridge_api *api, const expected_abi *expected) {
    uint32_t abi_version = api->abi_version();
    size_t observation_size = api->observation_size();
    int screen_cols = api->screen_cols();
    int screen_rows = api->screen_rows();
    int map_cols = api->map_cols();
    int map_rows = api->map_rows();
    int inventory_size = api->inventory_size();
    int inventory_str_length = api->inventory_str_length();

    if (abi_version != expected->abi_version
        || observation_size != expected->observation_size
        || screen_cols != expected->screen_cols
        || screen_rows != expected->screen_rows
        || map_cols != expected->map_cols
        || map_rows != expected->map_rows
        || inventory_size != expected->inventory_size
        || inventory_str_length != expected->inventory_str_length
        || expected->program_state_offset + 8 * sizeof(uint64_t) > observation_size) {
        fprintf(stderr,
                "Brogue bridge ABI mismatch: "
                "abi=%u expected=%u, "
                "observation_size=%zu expected=%zu, "
                "screen=%dx%d expected=%dx%d, "
                "map=%dx%d expected=%dx%d, "
                "inventory=%d expected=%d, "
                "inventory_str_length=%d expected=%d, "
                "program_state_offset=%zu\n",
                abi_version,
                expected->abi_version,
                observation_size,
                expected->observation_size,
                screen_cols,
                screen_rows,
                expected->screen_cols,
                expected->screen_rows,
                map_cols,
                map_rows,
                expected->map_cols,
                expected->map_rows,
                inventory_size,
                expected->inventory_size,
                inventory_str_length,
                expected->inventory_str_length,
                expected->program_state_offset);
        exit(2);
    }
}

static int terminated(const void *observation, size_t program_state_offset) {
    const uint64_t *program_state = (const uint64_t *) (
        (const char *) observation + program_state_offset
    );
    return program_state[1] != 0;
}

static void run_unmeasured_warmup(const bridge_api *api,
                                  const char *label,
                                  long key,
                                  int control,
                                  int shift,
                                  int warmup,
                                  uint64_t seed,
                                  void *observation,
                                  size_t program_state_offset) {
    int rc = api->reset(seed, observation);
    if (rc != 0) {
        fprintf(stderr, "%s warmup reset failed: %s\n", label, api->last_error());
        exit(2);
    }

    int resets = 0;
    for (int i = 0; i < warmup; i++) {
        rc = api->step(key, control, shift, observation);
        if (rc < 0) {
            fprintf(stderr, "%s warmup step failed at %d: %s\n", label, i, api->last_error());
            exit(2);
        }
        if (terminated(observation, program_state_offset)) {
            resets++;
            rc = api->reset(seed + (uint64_t) resets, observation);
            if (rc != 0) {
                fprintf(stderr, "%s warmup reset failed: %s\n", label, api->last_error());
                exit(2);
            }
        }
    }
    api->close();
}

static trial_result run_step_trial(const bridge_api *api,
                                   const char *label,
                                   long key,
                                   int control,
                                   int shift,
                                   int target_steps,
                                   uint64_t seed,
                                   void *observation,
                                   size_t program_state_offset) {
    trial_result result = {0, 0.0, 0, 0};
    int rc = api->reset(seed, observation);
    if (rc != 0) {
        fprintf(stderr, "%s reset failed: %s\n", label, api->last_error());
        exit(2);
    }

    double start = now_seconds();
    for (int i = 0; i < target_steps; i++) {
        rc = api->step(key, control, shift, observation);
        if (rc < 0) {
            fprintf(stderr, "%s step failed at %d: %s\n", label, i, api->last_error());
            exit(2);
        }
        result.steps++;
        result.invalid += (rc == 1);
        if (terminated(observation, program_state_offset)) {
            result.terminated = 1;
            break;
        }
    }
    result.elapsed = now_seconds() - start;
    api->close();
    return result;
}

static void summarize_step_case(const char *label,
                                int target_steps,
                                int repeats,
                                const trial_result *results) {
    double *samples = (double *) calloc((size_t) repeats, sizeof(double));
    if (samples == NULL) {
        perror("calloc samples");
        exit(2);
    }

    int completed_steps = 0;
    int terminated_repeats = 0;
    int invalid = 0;
    for (int i = 0; i < repeats; i++) {
        if (results[i].steps < 1) {
            fprintf(stderr, "%s repeat %d completed no steps\n", label, i);
            exit(2);
        }
        samples[i] = results[i].elapsed * 1000000.0 / (double) results[i].steps;
        completed_steps += results[i].steps;
        terminated_repeats += results[i].terminated;
        invalid += results[i].invalid;
    }
    qsort(samples, (size_t) repeats, sizeof(double), double_compare);

    double min_us = samples[0];
    double median_us = median_sample(samples, repeats);
    double p95_us = p95_sample(samples, repeats);
    printf("%-14s  %8d  %7d  %8.2f  %8.2f  %8.2f  %10.0f  %6d  %7d  %7d\n",
           label,
           target_steps,
           repeats,
           median_us,
           min_us,
           p95_us,
           1000000.0 / median_us,
           completed_steps,
           terminated_repeats,
           invalid);
    free(samples);
}

static void benchmark_steps(const bridge_api *api,
                            const char *label,
                            long key,
                            int control,
                            int shift,
                            int base_steps,
                            int warmup,
                            int percent,
                            int repeats,
                            uint64_t seed,
                            void *observation,
                            size_t program_state_offset) {
    int target_steps = scaled_count(base_steps, percent);
    trial_result *results = (trial_result *) calloc((size_t) repeats, sizeof(trial_result));
    if (results == NULL) {
        perror("calloc results");
        exit(2);
    }

    run_unmeasured_warmup(api,
                          label,
                          key,
                          control,
                          shift,
                          warmup,
                          seed ^ 0x9e3779b97f4a7c15ULL,
                          observation,
                          program_state_offset);

    for (int repeat = 0; repeat < repeats; repeat++) {
        results[repeat] = run_step_trial(api,
                                         label,
                                         key,
                                         control,
                                         shift,
                                         target_steps,
                                         seed,
                                         observation,
                                         program_state_offset);
    }
    summarize_step_case(label, target_steps, repeats, results);
    free(results);
}

static void benchmark_resets(const bridge_api *api,
                             int base_resets,
                             int percent,
                             int repeats,
                             uint64_t seed,
                             void *observation) {
    int target_resets = scaled_count(base_resets, percent);
    double *samples = (double *) calloc((size_t) repeats, sizeof(double));
    if (samples == NULL) {
        perror("calloc reset samples");
        exit(2);
    }

    for (int repeat = 0; repeat < repeats; repeat++) {
        double start = now_seconds();
        for (int i = 0; i < target_resets; i++) {
            int rc = api->reset(seed + (uint64_t) i, observation);
            if (rc != 0) {
                fprintf(stderr, "reset bench failed at %d: %s\n", i, api->last_error());
                exit(2);
            }
        }
        double elapsed = now_seconds() - start;
        samples[repeat] = elapsed * 1000000.0 / (double) target_resets;
        api->close();
    }

    qsort(samples, (size_t) repeats, sizeof(double), double_compare);
    double min_us = samples[0];
    double median_us = median_sample(samples, repeats);
    double p95_us = p95_sample(samples, repeats);
    printf("%-14s  %8d  %7d  %8.2f  %8.2f  %8.2f  %10.0f  %6d  %7d  %7d\n",
           "reset",
           target_resets,
           repeats,
           median_us,
           min_us,
           p95_us,
           1000000.0 / median_us,
           target_resets * repeats,
           0,
           0);
    free(samples);
}

static void benchmark_copy_only(size_t observation_size, int percent, int repeats) {
    int target_copies = scaled_count(20000, percent);
    unsigned char *source = (unsigned char *) malloc(observation_size);
    unsigned char *dest = (unsigned char *) malloc(observation_size);
    double *samples = (double *) calloc((size_t) repeats, sizeof(double));
    if (source == NULL || dest == NULL || samples == NULL) {
        perror("copy-only allocation");
        exit(2);
    }
    for (size_t i = 0; i < observation_size; i++) {
        source[i] = (unsigned char) (i & 0xffU);
    }

    volatile unsigned int checksum = 0;
    for (int repeat = 0; repeat < repeats; repeat++) {
        double start = now_seconds();
        for (int i = 0; i < target_copies; i++) {
            memcpy(dest, source, observation_size);
            checksum += dest[((size_t) i * 97U) % observation_size];
        }
        double elapsed = now_seconds() - start;
        samples[repeat] = elapsed * 1000000.0 / (double) target_copies;
    }

    qsort(samples, (size_t) repeats, sizeof(double), double_compare);
    double min_us = samples[0];
    double median_us = median_sample(samples, repeats);
    double p95_us = p95_sample(samples, repeats);
    copy_checksum_sink = checksum;
    printf("%-14s  %8d  %7d  %8.2f  %8.2f  %8.2f  %10.0f  %6d  %7d  %7d\n",
           "copy-only",
           target_copies,
           repeats,
           median_us,
           min_us,
           p95_us,
           1000000.0 / median_us,
           target_copies * repeats,
           0,
           0);

    free(source);
    free(dest);
    free(samples);
}

int main(int argc, char **argv) {
    if (argc != 14) {
        fprintf(stderr,
                "usage: %s LIBRARY_PATH DATA_DIR PERCENT REPEATS "
                "ABI OBS_SIZE SCREEN_COLS SCREEN_ROWS MAP_COLS MAP_ROWS "
                "INVENTORY_SIZE INVENTORY_STR_LENGTH PROGRAM_STATE_OFFSET\n",
                argv[0]);
        return 2;
    }

    const char *library_path = argv[1];
    const char *data_dir = argv[2];
    int percent = parse_int_arg("percent", argv[3]);
    int repeats = parse_int_arg("repeats", argv[4]);
    expected_abi expected;
    expected.abi_version = (uint32_t) parse_u64_arg("expected abi", argv[5]);
    expected.observation_size = parse_size_arg("expected observation size", argv[6]);
    expected.screen_cols = parse_int_arg("expected screen cols", argv[7]);
    expected.screen_rows = parse_int_arg("expected screen rows", argv[8]);
    expected.map_cols = parse_int_arg("expected map cols", argv[9]);
    expected.map_rows = parse_int_arg("expected map rows", argv[10]);
    expected.inventory_size = parse_int_arg("expected inventory size", argv[11]);
    expected.inventory_str_length = parse_int_arg("expected inventory str length", argv[12]);
    expected.program_state_offset = parse_size_arg("program state offset", argv[13]);
    if (percent < 1) {
        fprintf(stderr, "percent must be at least 1\n");
        return 2;
    }
    if (repeats < 1) {
        fprintf(stderr, "repeats must be at least 1\n");
        return 2;
    }

    void *handle = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
    if (handle == NULL) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 2;
    }

    bridge_api api;
    api.set_data_dir = (set_data_dir_fn) must_symbol(handle, "brh_set_data_dir");
    api.abi_version = (abi_version_fn) must_symbol(handle, "brh_abi_version");
    api.observation_size = (observation_size_fn) must_symbol(handle, "brh_observation_size");
    api.screen_cols = (dimension_fn) must_symbol(handle, "brh_screen_cols");
    api.screen_rows = (dimension_fn) must_symbol(handle, "brh_screen_rows");
    api.map_cols = (dimension_fn) must_symbol(handle, "brh_map_cols");
    api.map_rows = (dimension_fn) must_symbol(handle, "brh_map_rows");
    api.inventory_size = (dimension_fn) must_symbol(handle, "brh_inventory_size");
    api.inventory_str_length = (dimension_fn) must_symbol(handle, "brh_inventory_str_length");
    api.reset = (reset_fn) must_symbol(handle, "brh_reset");
    api.step = (step_fn) must_symbol(handle, "brh_step");
    api.close = (close_fn) must_symbol(handle, "brh_close");
    api.last_error = (last_error_fn) must_symbol(handle, "brh_last_error");

    validate_bridge_abi(&api, &expected);
    api.set_data_dir(data_dir);
    void *observation = calloc(1, expected.observation_size);
    if (observation == NULL) {
        perror("calloc observation");
        return 2;
    }

    printf("Brogue C Bridge Direct Benchmark\n");
    printf("library: %s\n", library_path);
    printf("data dir: %s\n", data_dir);
    printf("observation: %zu bytes\n", expected.observation_size);
    printf("scale: %d%% measured trace length only; compare stateful cases at the same scale\n", percent);
    printf("repeats: %d\n", repeats);
    printf("step cases include full observation fill and export\n");
    printf("\n");
    printf("%-14s  %8s  %7s  %8s  %8s  %8s  %10s  %6s  %7s  %7s\n",
           "case",
           "target",
           "repeats",
           "med_us",
           "min_us",
           "p95_us",
           "med/sec",
           "done",
           "term",
           "invalid");
    printf("------------------------------------------------------------------------------------------------\n");

    benchmark_copy_only(expected.observation_size, percent, repeats);
    benchmark_steps(&api, "invalid-key", '!', 0, 0, 20000, 1000, percent, repeats, 1, observation, expected.program_state_offset);
    benchmark_steps(&api, "rest", 'z', 0, 0, 5000, 200, percent, repeats, 1001, observation, expected.program_state_offset);
    benchmark_steps(&api, "search", 's', 0, 0, 5000, 200, percent, repeats, 2001, observation, expected.program_state_offset);
    benchmark_steps(&api, "explore", 'x', 0, 0, 1000, 50, percent, repeats, 3001, observation, expected.program_state_offset);
    benchmark_steps(&api, "fast-explore", 'x', 1, 0, 1000, 50, percent, repeats, 4001, observation, expected.program_state_offset);
    benchmark_resets(&api, 200, percent, repeats, 5001, observation);

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


def harness_args(library: Path, data_dir: Path, scale: int, repeats: int) -> list[str]:
    """Return native harness arguments, including Python-side ABI expectations."""

    return [
        str(library),
        str(data_dir),
        str(scale),
        str(repeats),
        str(_BRIDGE_ABI_VERSION),
        str(ctypes.sizeof(_CObservation)),
        str(_SCREEN_COLS),
        str(_SCREEN_ROWS),
        str(_MAP_COLS),
        str(_MAP_ROWS),
        str(_INVENTORY_SIZE),
        str(_INVENTORY_STR_LENGTH),
        str(_CObservation.program_state.offset),
    ]


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
        help="Percentage of measured trace length to run; compare stateful cases at the same scale",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Timed repeats per case",
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
    if args.repeats < 1:
        msg = "--repeats must be at least 1"
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
            [str(binary_path), *harness_args(args.library, args.data_dir, args.scale, args.repeats)],
            check=True,
        )
        return

    with tempfile.TemporaryDirectory(prefix="broguegym-bridge-bench-") as temp_dir:
        binary_path = Path(temp_dir) / "bench_bridge_c"
        compile_harness(args.cc, binary_path)
        subprocess.run(
            [str(binary_path), *harness_args(args.library, args.data_dir, args.scale, args.repeats)],
            check=True,
        )


if __name__ == "__main__":
    main()

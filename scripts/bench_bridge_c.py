"""Benchmark the Brogue C bridge directly, outside the Python backend."""

from __future__ import annotations

import argparse
import ctypes
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime
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
typedef struct brh_env brh_env;
typedef struct brh_env_buffers {
    void *observations;
    long *actions;
    unsigned char *controls;
    unsigned char *shifts;
    float *rewards;
    float *terminals;
} brh_env_buffers;
typedef brh_env *(*env_create_fn)(const brh_env_buffers *buffers);
typedef int (*env_reset_fn)(brh_env *env, uint64_t seed);
typedef int (*env_step_fn)(brh_env *env, long key, int control, int shift);
typedef int (*env_num_agents_fn)(const brh_env *env);
typedef void (*env_close_fn)(brh_env *env);

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
    env_create_fn env_create;
    env_reset_fn env_reset;
    env_step_fn env_step;
    env_num_agents_fn env_num_agents;
    env_close_fn env_close;
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

typedef struct workload_profile {
    const char *name;
    int copy_count;
    int invalid_steps;
    int invalid_warmup;
    int rest_steps;
    int rest_warmup;
    int search_steps;
    int search_warmup;
    int explore_steps;
    int explore_warmup;
    int fast_explore_steps;
    int fast_explore_warmup;
    int reset_count;
} workload_profile;

typedef struct trial_result {
    int steps;
    double elapsed;
    int terminated;
    int invalid;
} trial_result;

static const workload_profile WORKLOAD_PROFILES[] = {
    {"smoke", 200, 200, 20, 50, 10, 50, 10, 10, 2, 10, 2, 2},
    {"standard", 20000, 20000, 1000, 5000, 200, 5000, 200, 1000, 50, 1000, 50, 200},
    {"long", 100000, 100000, 2000, 20000, 500, 20000, 500, 5000, 200, 5000, 200, 1000},
};

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

static double tail_sample(const double *sorted, int count) {
    if (count < 20) {
        return sorted[count - 1];
    }
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

static const workload_profile *find_profile(const char *name) {
    size_t count = sizeof(WORKLOAD_PROFILES) / sizeof(WORKLOAD_PROFILES[0]);
    for (size_t i = 0; i < count; i++) {
        if (strcmp(WORKLOAD_PROFILES[i].name, name) == 0) {
            return &WORKLOAD_PROFILES[i];
        }
    }
    fprintf(stderr, "unknown profile %s; expected smoke, standard, or long\n", name);
    exit(2);
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
                                  brh_env *env,
                                  const char *label,
                                  long key,
                                  int control,
                                  int shift,
                                  int warmup,
                                  uint64_t seed,
                                  void *observation,
                                  size_t program_state_offset) {
    int rc = api->env_reset(env, seed);
    if (rc != 0) {
        fprintf(stderr, "%s warmup reset failed: %s\n", label, api->last_error());
        exit(2);
    }

    int resets = 0;
    for (int i = 0; i < warmup; i++) {
        rc = api->env_step(env, key, control, shift);
        if (rc < 0) {
            fprintf(stderr, "%s warmup step failed at %d: %s\n", label, i, api->last_error());
            exit(2);
        }
        if (terminated(observation, program_state_offset)) {
            resets++;
            rc = api->env_reset(env, seed + (uint64_t) resets);
            if (rc != 0) {
                fprintf(stderr, "%s warmup reset failed: %s\n", label, api->last_error());
                exit(2);
            }
        }
    }
}

static trial_result run_step_trial(const bridge_api *api,
                                   brh_env *env,
                                   const char *label,
                                   long key,
                                   int control,
                                   int shift,
                                   int target_steps,
                                   uint64_t seed,
                                   void *observation,
                                   size_t program_state_offset) {
    trial_result result = {0, 0.0, 0, 0};
    int rc = api->env_reset(env, seed);
    if (rc != 0) {
        fprintf(stderr, "%s reset failed: %s\n", label, api->last_error());
        exit(2);
    }

    double start = now_seconds();
    for (int i = 0; i < target_steps; i++) {
        rc = api->env_step(env, key, control, shift);
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
    return result;
}

static void summarize_case(const char *label,
                           int target_steps,
                           int seed_count,
                           int trace_repeats,
                           int sample_count,
                           const trial_result *results) {
    double *samples = (double *) calloc((size_t) sample_count, sizeof(double));
    if (samples == NULL) {
        perror("calloc samples");
        exit(2);
    }

    int completed_steps = 0;
    int terminated_trials = 0;
    int invalid = 0;
    for (int i = 0; i < sample_count; i++) {
        if (results[i].steps < 1) {
            fprintf(stderr, "%s trial %d completed no steps\n", label, i);
            exit(2);
        }
        samples[i] = results[i].elapsed * 1000000.0 / (double) results[i].steps;
        completed_steps += results[i].steps;
        terminated_trials += results[i].terminated;
        invalid += results[i].invalid;
    }
    qsort(samples, (size_t) sample_count, sizeof(double), double_compare);

    double min_us = samples[0];
    double median_us = median_sample(samples, sample_count);
    double tail_us = tail_sample(samples, sample_count);
    printf("%-14s  %8d  %5d  %7d  %7d  %8.2f  %8.2f  %8.2f  %10.0f  %7d  %5d  %7d\n",
           label,
           target_steps,
           seed_count,
           trace_repeats,
           sample_count,
           median_us,
           min_us,
           tail_us,
           1000000.0 / median_us,
           completed_steps,
           terminated_trials,
           invalid);
    free(samples);
}

static void benchmark_steps(const bridge_api *api,
                            brh_env *env,
                            const char *label,
                            long key,
                            int control,
                            int shift,
                            int target_steps,
                            int warmup,
                            int trace_repeats,
                            int seed_count,
                            uint64_t seed_start,
                            uint64_t case_seed,
                            void *observation,
                            size_t program_state_offset) {
    int sample_count = trace_repeats * seed_count;
    trial_result *results = (trial_result *) calloc((size_t) sample_count, sizeof(trial_result));
    if (results == NULL) {
        perror("calloc results");
        exit(2);
    }

    run_unmeasured_warmup(api,
                          env,
                          label,
                          key,
                          control,
                          shift,
                          warmup,
                          (case_seed + seed_start) ^ 0x9e3779b97f4a7c15ULL,
                          observation,
                          program_state_offset);

    int result_index = 0;
    for (int seed_index = 0; seed_index < seed_count; seed_index++) {
        uint64_t seed = case_seed + seed_start + (uint64_t) seed_index;
        for (int repeat = 0; repeat < trace_repeats; repeat++) {
            results[result_index++] = run_step_trial(api,
                                                     env,
                                                     label,
                                                     key,
                                                     control,
                                                     shift,
                                                     target_steps,
                                                     seed,
                                                     observation,
                                                     program_state_offset);
        }
    }
    summarize_case(label, target_steps, seed_count, trace_repeats, sample_count, results);
    free(results);
}

static void benchmark_resets(const bridge_api *api,
                             brh_env *env,
                             int target_resets,
                             int trace_repeats,
                             int seed_count,
                             uint64_t seed_start) {
    int sample_count = trace_repeats * seed_count;
    trial_result *results = (trial_result *) calloc((size_t) sample_count, sizeof(trial_result));
    if (results == NULL) {
        perror("calloc reset results");
        exit(2);
    }

    int result_index = 0;
    for (int seed_index = 0; seed_index < seed_count; seed_index++) {
        for (int repeat = 0; repeat < trace_repeats; repeat++) {
            trial_result result = {target_resets, 0.0, 0, 0};
            uint64_t seed_base = seed_start + 5001ULL + (uint64_t) seed_index * 1000003ULL;
            double start = now_seconds();
            for (int i = 0; i < target_resets; i++) {
                int rc = api->env_reset(env, seed_base + (uint64_t) i);
                if (rc != 0) {
                    fprintf(stderr, "reset bench failed at %d: %s\n", i, api->last_error());
                    exit(2);
                }
            }
            result.elapsed = now_seconds() - start;
            results[result_index++] = result;
        }
    }
    summarize_case("reset", target_resets, seed_count, trace_repeats, sample_count, results);
    free(results);
}

static void benchmark_copy_only(size_t observation_size, int target_copies, int trace_repeats) {
    unsigned char *source = (unsigned char *) malloc(observation_size);
    unsigned char *dest = (unsigned char *) malloc(observation_size);
    trial_result *results = (trial_result *) calloc((size_t) trace_repeats, sizeof(trial_result));
    if (source == NULL || dest == NULL || results == NULL) {
        perror("copy-only allocation");
        exit(2);
    }
    for (size_t i = 0; i < observation_size; i++) {
        source[i] = (unsigned char) (i & 0xffU);
    }

    volatile unsigned int checksum = 0;
    for (int repeat = 0; repeat < trace_repeats; repeat++) {
        trial_result result = {target_copies, 0.0, 0, 0};
        double start = now_seconds();
        for (int i = 0; i < target_copies; i++) {
            memcpy(dest, source, observation_size);
            checksum += dest[((size_t) i * 97U) % observation_size];
        }
        result.elapsed = now_seconds() - start;
        results[repeat] = result;
    }

    copy_checksum_sink = checksum;
    summarize_case("copy-only", target_copies, 0, trace_repeats, trace_repeats, results);
    free(source);
    free(dest);
    free(results);
}

int main(int argc, char **argv) {
    if (argc != 16) {
        fprintf(stderr,
                "usage: %s LIBRARY_PATH DATA_DIR PROFILE TRACE_REPEATS SEED_COUNT SEED_START "
                "ABI OBS_SIZE SCREEN_COLS SCREEN_ROWS MAP_COLS MAP_ROWS "
                "INVENTORY_SIZE INVENTORY_STR_LENGTH PROGRAM_STATE_OFFSET\n",
                argv[0]);
        return 2;
    }

    const char *library_path = argv[1];
    const char *data_dir = argv[2];
    const workload_profile *profile = find_profile(argv[3]);
    int trace_repeats = parse_int_arg("trace repeats", argv[4]);
    int seed_count = parse_int_arg("seed count", argv[5]);
    uint64_t seed_start = parse_u64_arg("seed start", argv[6]);
    expected_abi expected;
    expected.abi_version = (uint32_t) parse_u64_arg("expected abi", argv[7]);
    expected.observation_size = parse_size_arg("expected observation size", argv[8]);
    expected.screen_cols = parse_int_arg("expected screen cols", argv[9]);
    expected.screen_rows = parse_int_arg("expected screen rows", argv[10]);
    expected.map_cols = parse_int_arg("expected map cols", argv[11]);
    expected.map_rows = parse_int_arg("expected map rows", argv[12]);
    expected.inventory_size = parse_int_arg("expected inventory size", argv[13]);
    expected.inventory_str_length = parse_int_arg("expected inventory str length", argv[14]);
    expected.program_state_offset = parse_size_arg("program state offset", argv[15]);
    if (trace_repeats < 1) {
        fprintf(stderr, "trace repeats must be at least 1\n");
        return 2;
    }
    if (seed_count < 1) {
        fprintf(stderr, "seed count must be at least 1\n");
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
    api.env_create = (env_create_fn) must_symbol(handle, "brh_env_create");
    api.env_reset = (env_reset_fn) must_symbol(handle, "brh_env_reset");
    api.env_step = (env_step_fn) must_symbol(handle, "brh_env_step");
    api.env_num_agents = (env_num_agents_fn) must_symbol(handle, "brh_env_num_agents");
    api.env_close = (env_close_fn) must_symbol(handle, "brh_env_close");

    validate_bridge_abi(&api, &expected);
    api.set_data_dir(data_dir);
    void *observation = calloc(1, expected.observation_size);
    if (observation == NULL) {
        perror("calloc observation");
        return 2;
    }
    long actions[1] = {0};
    unsigned char controls[1] = {0};
    unsigned char shifts[1] = {0};
    float rewards[1] = {0.0f};
    float terminals[1] = {0.0f};
    brh_env_buffers buffers = {
        observation,
        actions,
        controls,
        shifts,
        rewards,
        terminals,
    };
    brh_env *env = api.env_create(&buffers);
    if (env == NULL) {
        fprintf(stderr, "brh_env_create failed: %s\n", api.last_error());
        free(observation);
        dlclose(handle);
        return 2;
    }
    if (api.env_num_agents(env) != 1) {
        fprintf(stderr, "brh_env_num_agents returned %d; expected 1\n", api.env_num_agents(env));
        api.env_close(env);
        free(observation);
        dlclose(handle);
        return 2;
    }

    printf("Brogue C Bridge Direct Benchmark\n");
    printf("library: %s\n", library_path);
    printf("data dir: %s\n", data_dir);
    printf("observation: %zu bytes\n", expected.observation_size);
    printf("profile: %s fixed trace lengths\n", profile->name);
    printf("seed count: %d; trace repeats per seed: %d\n", seed_count, trace_repeats);
    printf("tail_us is p95 for >=20 samples, otherwise max\n");
    printf("step cases use scalar BrogueEnv ABI and include full observation fill/export\n");
    printf("no fill-only baseline: current bridge ABI exposes no observation-export-only call\n");
    printf("\n");
    printf("%-14s  %8s  %5s  %7s  %7s  %8s  %8s  %8s  %10s  %7s  %5s  %7s\n",
           "case",
           "target",
           "seeds",
           "repeats",
           "samples",
           "med_us",
           "min_us",
           "tail_us",
           "med/sec",
           "done",
           "term",
           "invalid");
    printf("----------------------------------------------------------------------------------------------------------------\n");

    benchmark_copy_only(expected.observation_size, profile->copy_count, trace_repeats);
    benchmark_steps(&api, env, "invalid-key", '!', 0, 0, profile->invalid_steps, profile->invalid_warmup, trace_repeats, seed_count, seed_start, 1, observation, expected.program_state_offset);
    benchmark_steps(&api, env, "rest", 'z', 0, 0, profile->rest_steps, profile->rest_warmup, trace_repeats, seed_count, seed_start, 1001, observation, expected.program_state_offset);
    benchmark_steps(&api, env, "search", 's', 0, 0, profile->search_steps, profile->search_warmup, trace_repeats, seed_count, seed_start, 2001, observation, expected.program_state_offset);
    benchmark_steps(&api, env, "explore", 'x', 0, 0, profile->explore_steps, profile->explore_warmup, trace_repeats, seed_count, seed_start, 3001, observation, expected.program_state_offset);
    benchmark_steps(&api, env, "fast-explore", 'x', 1, 0, profile->fast_explore_steps, profile->fast_explore_warmup, trace_repeats, seed_count, seed_start, 4001, observation, expected.program_state_offset);
    benchmark_resets(&api, env, profile->reset_count, trace_repeats, seed_count, seed_start);

    api.env_close(env);
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


def command_first_line(command: list[str], *, cwd: Path | None = None) -> str:
    """Return the first output line for a best-effort provenance command."""

    try:
        output = subprocess.check_output(
            command,
            cwd=cwd,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable ({exc})"
    first_line = output.splitlines()[0] if output.splitlines() else ""
    return first_line or "unavailable"


def git_status(root: Path) -> str:
    """Return a compact git status string for benchmark provenance."""

    try:
        output = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=root,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unknown ({exc})"
    lines = output.splitlines()
    if not lines:
        return "clean"
    return f"dirty ({len(lines)} status lines)"


def print_provenance(cc: str, library: Path) -> None:
    """Print run metadata that affects benchmark comparability."""

    root = Path(__file__).resolve().parents[1]
    library_stat = library.stat()
    modified = datetime.fromtimestamp(library_stat.st_mtime).isoformat(timespec="seconds")
    print("Benchmark provenance")
    print(f"repo: {command_first_line(['git', 'rev-parse', '--short', 'HEAD'], cwd=root)} {git_status(root)}")
    print(f"system: {platform.platform()} ({platform.machine()})")
    print(f"python: {platform.python_version()}")
    print(f"harness compiler: {cc}")
    print(f"harness compiler version: {command_first_line([cc, '--version'])}")
    print("harness flags: -O3 -Wall -Wextra")
    print(f"bridge library: {library} ({library_stat.st_size} bytes, mtime {modified})")
    print(
        "bridge build env: "
        f"CC={os.environ.get('CC', '')!r} "
        f"CFLAGS={os.environ.get('CFLAGS', '')!r} "
        f"CPPFLAGS={os.environ.get('CPPFLAGS', '')!r} "
        f"LDLIBS={os.environ.get('LDLIBS', '')!r}",
    )
    print(flush=True)


def harness_args(
    library: Path,
    data_dir: Path,
    profile: str,
    trace_repeats: int,
    seed_count: int,
    seed_start: int,
) -> list[str]:
    """Return native harness arguments, including Python-side ABI expectations."""

    return [
        str(library),
        str(data_dir),
        profile,
        str(trace_repeats),
        str(seed_count),
        str(seed_start),
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
        "--profile",
        choices=("smoke", "standard", "long"),
        default="standard",
        help="Fixed workload profile to run",
    )
    parser.add_argument(
        "--trace-repeats",
        "--repeats",
        dest="trace_repeats",
        type=int,
        default=3,
        help="Timed repeats per seed for each case",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=1,
        help="Number of consecutive seeds to measure for each stateful case",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="Offset added to each case's deterministic base seed",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=None,
        help="Optional output path for the compiled harness",
    )
    args = parser.parse_args()

    if args.trace_repeats < 1:
        msg = "--trace-repeats must be at least 1"
        raise ValueError(msg)
    if args.seed_count < 1:
        msg = "--seed-count must be at least 1"
        raise ValueError(msg)
    if args.seed_start < 0:
        msg = "--seed-start must be non-negative"
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
        print_provenance(args.cc, args.library)
        subprocess.run(
            [
                str(binary_path),
                *harness_args(
                    args.library,
                    args.data_dir,
                    args.profile,
                    args.trace_repeats,
                    args.seed_count,
                    args.seed_start,
                ),
            ],
            check=True,
        )
        return

    with tempfile.TemporaryDirectory(prefix="broguegym-bridge-bench-") as temp_dir:
        binary_path = Path(temp_dir) / "bench_bridge_c"
        compile_harness(args.cc, binary_path)
        print_provenance(args.cc, args.library)
        subprocess.run(
            [
                str(binary_path),
                *harness_args(
                    args.library,
                    args.data_dir,
                    args.profile,
                    args.trace_repeats,
                    args.seed_count,
                    args.seed_start,
                ),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()

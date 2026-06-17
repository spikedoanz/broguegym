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

#define BRH_SCREEN_COLS 100
#define BRH_SCREEN_ROWS 34
#define BRH_MAP_COLS 79
#define BRH_MAP_ROWS 29
#define BRH_TERRAIN_LAYERS 4
#define BRH_OBS_CELLS (BRH_SCREEN_COLS * BRH_SCREEN_ROWS)
#define BRH_MAP_CELLS (BRH_MAP_COLS * BRH_MAP_ROWS)
#define BRH_MAP_LAYER_CELLS (BRH_MAP_CELLS * BRH_TERRAIN_LAYERS)
#define BRH_COLOR_CELLS (BRH_OBS_CELLS * 3)
#define BRH_MAP_COLOR_CELLS (BRH_MAP_CELLS * 3)
#define BRH_BLSTATS_SIZE 21
#define BRH_MESSAGE_SIZE 256
#define BRH_PROGRAM_STATE_SIZE 8
#define BRH_INVENTORY_SIZE 26
#define BRH_INVENTORY_STR_LENGTH 80
#define BRH_INVENTORY_STR_CELLS (BRH_INVENTORY_SIZE * BRH_INVENTORY_STR_LENGTH)
#define BRH_OBSERVATION_MASK_SCREEN ((uint64_t) 1U << 0)
#define BRH_OBSERVATION_MASK_SEMANTIC_VISIBLE ((uint64_t) 1U << 1)
#define BRH_OBSERVATION_MASK_INVENTORY ((uint64_t) 1U << 2)
#define BRH_OBSERVATION_MASK_FULL (BRH_OBSERVATION_MASK_SCREEN \
                                   | BRH_OBSERVATION_MASK_SEMANTIC_VISIBLE \
                                   | BRH_OBSERVATION_MASK_INVENTORY)

typedef void (*set_data_dir_fn)(const char *path);
typedef uint32_t (*abi_version_fn)(void);
typedef size_t (*observation_size_fn)(void);
typedef int (*dimension_fn)(void);
typedef int (*reset_fn)(uint64_t seed, void *out);
typedef int (*step_fn)(long key, int control, int shift, void *out);
typedef uint64_t (*supported_observation_mask_fn)(void);
typedef int (*register_observation_buffers_fn)(const void *buffers, uint64_t field_mask);
typedef void (*clear_observation_buffers_fn)(void);
typedef int (*reset_registered_fn)(uint64_t seed);
typedef int (*step_registered_fn)(long key, int control, int shift);
typedef void (*close_fn)(void);
typedef const char *(*last_error_fn)(void);

typedef struct brh_observation {
    int16_t glyphs[BRH_OBS_CELLS];
    uint32_t chars[BRH_OBS_CELLS];
    uint8_t colors_fg[BRH_COLOR_CELLS];
    uint8_t colors_bg[BRH_COLOR_CELLS];
    uint8_t specials[BRH_OBS_CELLS];
    uint16_t map_layers[BRH_MAP_LAYER_CELLS];
    uint64_t map_flags[BRH_MAP_CELLS];
    uint16_t map_volume[BRH_MAP_CELLS];
    uint8_t map_machine[BRH_MAP_CELLS];
    int16_t map_light[BRH_MAP_COLOR_CELLS];
    uint8_t map_has_item[BRH_MAP_CELLS];
    uint16_t map_item_category[BRH_MAP_CELLS];
    int16_t map_item_kind[BRH_MAP_CELLS];
    int16_t map_item_quantity[BRH_MAP_CELLS];
    uint64_t map_item_flags[BRH_MAP_CELLS];
    uint8_t map_has_monster[BRH_MAP_CELLS];
    int16_t map_monster_kind[BRH_MAP_CELLS];
    int16_t map_monster_hp[BRH_MAP_CELLS];
    int16_t map_monster_state[BRH_MAP_CELLS];
    uint8_t inventory_present[BRH_INVENTORY_SIZE];
    uint8_t inventory_letters[BRH_INVENTORY_SIZE];
    uint8_t inventory_strs[BRH_INVENTORY_STR_CELLS];
    uint16_t inventory_category[BRH_INVENTORY_SIZE];
    int16_t inventory_kind[BRH_INVENTORY_SIZE];
    int16_t inventory_quantity[BRH_INVENTORY_SIZE];
    uint64_t inventory_flags[BRH_INVENTORY_SIZE];
    int16_t inventory_enchant1[BRH_INVENTORY_SIZE];
    int16_t inventory_enchant2[BRH_INVENTORY_SIZE];
    int16_t inventory_charges[BRH_INVENTORY_SIZE];
    int64_t blstats[BRH_BLSTATS_SIZE];
    uint8_t message[BRH_MESSAGE_SIZE];
    uint64_t program_state[BRH_PROGRAM_STATE_SIZE];
} brh_observation;

typedef struct brh_observation_buffers {
    int16_t *glyphs;
    uint32_t *chars;
    uint8_t *colors_fg;
    uint8_t *colors_bg;
    uint8_t *specials;
    uint16_t *map_layers;
    uint64_t *map_flags;
    uint16_t *map_volume;
    uint8_t *map_machine;
    int16_t *map_light;
    uint8_t *map_has_item;
    uint16_t *map_item_category;
    int16_t *map_item_kind;
    int16_t *map_item_quantity;
    uint64_t *map_item_flags;
    uint8_t *map_has_monster;
    int16_t *map_monster_kind;
    int16_t *map_monster_hp;
    int16_t *map_monster_state;
    uint8_t *inventory_present;
    uint8_t *inventory_letters;
    uint8_t *inventory_strs;
    uint16_t *inventory_category;
    int16_t *inventory_kind;
    int16_t *inventory_quantity;
    uint64_t *inventory_flags;
    int16_t *inventory_enchant1;
    int16_t *inventory_enchant2;
    int16_t *inventory_charges;
    int64_t *blstats;
    uint8_t *message;
    uint64_t *program_state;
} brh_observation_buffers;

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
    supported_observation_mask_fn supported_observation_mask;
    register_observation_buffers_fn register_observation_buffers;
    clear_observation_buffers_fn clear_observation_buffers;
    reset_registered_fn reset_registered;
    step_registered_fn step_registered;
    close_fn close;
    last_error_fn last_error;
    int use_registered;
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

static int parse_api_mode(const char *name) {
    if (strcmp(name, "legacy") == 0) {
        return 0;
    }
    if (strcmp(name, "registered") == 0) {
        return 1;
    }
    fprintf(stderr, "unknown api %s; expected legacy or registered\n", name);
    exit(2);
}

static uint64_t parse_observation_profile(const char *name) {
    if (strcmp(name, "full") == 0) {
        return BRH_OBSERVATION_MASK_FULL;
    }
    if (strcmp(name, "screen") == 0) {
        return BRH_OBSERVATION_MASK_SCREEN;
    }
    fprintf(stderr, "unknown observation profile %s; expected full or screen\n", name);
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

static int bridge_reset_for_mode(const bridge_api *api, uint64_t seed, void *observation) {
    if (api->use_registered) {
        return api->reset_registered(seed);
    }
    return api->reset(seed, observation);
}

static int bridge_step_for_mode(const bridge_api *api,
                                long key,
                                int control,
                                int shift,
                                void *observation) {
    if (api->use_registered) {
        (void) observation;
        return api->step_registered(key, control, shift);
    }
    return api->step(key, control, shift, observation);
}

static void register_observation_buffers(const bridge_api *api,
                                         brh_observation *observation,
                                         uint64_t observation_mask) {
    brh_observation_buffers buffers;
    uint64_t supported_mask = api->supported_observation_mask();

    if ((observation_mask & ~supported_mask) != 0) {
        fprintf(stderr,
                "bridge does not support requested observation mask 0x%llx; supported=0x%llx\n",
                (unsigned long long) observation_mask,
                (unsigned long long) supported_mask);
        exit(2);
    }

    buffers.glyphs = observation->glyphs;
    buffers.chars = observation->chars;
    buffers.colors_fg = observation->colors_fg;
    buffers.colors_bg = observation->colors_bg;
    buffers.specials = observation->specials;
    buffers.map_layers = observation->map_layers;
    buffers.map_flags = observation->map_flags;
    buffers.map_volume = observation->map_volume;
    buffers.map_machine = observation->map_machine;
    buffers.map_light = observation->map_light;
    buffers.map_has_item = observation->map_has_item;
    buffers.map_item_category = observation->map_item_category;
    buffers.map_item_kind = observation->map_item_kind;
    buffers.map_item_quantity = observation->map_item_quantity;
    buffers.map_item_flags = observation->map_item_flags;
    buffers.map_has_monster = observation->map_has_monster;
    buffers.map_monster_kind = observation->map_monster_kind;
    buffers.map_monster_hp = observation->map_monster_hp;
    buffers.map_monster_state = observation->map_monster_state;
    buffers.inventory_present = observation->inventory_present;
    buffers.inventory_letters = observation->inventory_letters;
    buffers.inventory_strs = observation->inventory_strs;
    buffers.inventory_category = observation->inventory_category;
    buffers.inventory_kind = observation->inventory_kind;
    buffers.inventory_quantity = observation->inventory_quantity;
    buffers.inventory_flags = observation->inventory_flags;
    buffers.inventory_enchant1 = observation->inventory_enchant1;
    buffers.inventory_enchant2 = observation->inventory_enchant2;
    buffers.inventory_charges = observation->inventory_charges;
    buffers.blstats = observation->blstats;
    buffers.message = observation->message;
    buffers.program_state = observation->program_state;

    if (api->register_observation_buffers(&buffers, observation_mask) != 0) {
        fprintf(stderr, "registered observation buffer setup failed: %s\n", api->last_error());
        exit(2);
    }
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
    int rc = bridge_reset_for_mode(api, seed, observation);
    if (rc != 0) {
        fprintf(stderr, "%s warmup reset failed: %s\n", label, api->last_error());
        exit(2);
    }

    int resets = 0;
    for (int i = 0; i < warmup; i++) {
        rc = bridge_step_for_mode(api, key, control, shift, observation);
        if (rc < 0) {
            fprintf(stderr, "%s warmup step failed at %d: %s\n", label, i, api->last_error());
            exit(2);
        }
        if (terminated(observation, program_state_offset)) {
            resets++;
            rc = bridge_reset_for_mode(api, seed + (uint64_t) resets, observation);
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
    int rc = bridge_reset_for_mode(api, seed, observation);
    if (rc != 0) {
        fprintf(stderr, "%s reset failed: %s\n", label, api->last_error());
        exit(2);
    }

    double start = now_seconds();
    for (int i = 0; i < target_steps; i++) {
        rc = bridge_step_for_mode(api, key, control, shift, observation);
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
                             int target_resets,
                             int trace_repeats,
                             int seed_count,
                             uint64_t seed_start,
                             void *observation) {
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
                int rc = bridge_reset_for_mode(api, seed_base + (uint64_t) i, observation);
                if (rc != 0) {
                    fprintf(stderr, "reset bench failed at %d: %s\n", i, api->last_error());
                    exit(2);
                }
            }
            result.elapsed = now_seconds() - start;
            api->close();
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
    if (argc != 18) {
        fprintf(stderr,
                "usage: %s LIBRARY_PATH DATA_DIR PROFILE API OBS_PROFILE "
                "TRACE_REPEATS SEED_COUNT SEED_START "
                "ABI OBS_SIZE SCREEN_COLS SCREEN_ROWS MAP_COLS MAP_ROWS "
                "INVENTORY_SIZE INVENTORY_STR_LENGTH PROGRAM_STATE_OFFSET\n",
                argv[0]);
        return 2;
    }

    const char *library_path = argv[1];
    const char *data_dir = argv[2];
    const workload_profile *profile = find_profile(argv[3]);
    int use_registered = parse_api_mode(argv[4]);
    const char *observation_profile = argv[5];
    uint64_t observation_mask = parse_observation_profile(observation_profile);
    int trace_repeats = parse_int_arg("trace repeats", argv[6]);
    int seed_count = parse_int_arg("seed count", argv[7]);
    uint64_t seed_start = parse_u64_arg("seed start", argv[8]);
    expected_abi expected;
    expected.abi_version = (uint32_t) parse_u64_arg("expected abi", argv[9]);
    expected.observation_size = parse_size_arg("expected observation size", argv[10]);
    expected.screen_cols = parse_int_arg("expected screen cols", argv[11]);
    expected.screen_rows = parse_int_arg("expected screen rows", argv[12]);
    expected.map_cols = parse_int_arg("expected map cols", argv[13]);
    expected.map_rows = parse_int_arg("expected map rows", argv[14]);
    expected.inventory_size = parse_int_arg("expected inventory size", argv[15]);
    expected.inventory_str_length = parse_int_arg("expected inventory str length", argv[16]);
    expected.program_state_offset = parse_size_arg("program state offset", argv[17]);
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
    api.supported_observation_mask = (supported_observation_mask_fn) must_symbol(handle, "brh_supported_observation_mask");
    api.register_observation_buffers = (register_observation_buffers_fn) must_symbol(handle, "brh_register_observation_buffers");
    api.clear_observation_buffers = (clear_observation_buffers_fn) must_symbol(handle, "brh_clear_observation_buffers");
    api.reset_registered = (reset_registered_fn) must_symbol(handle, "brh_reset_registered");
    api.step_registered = (step_registered_fn) must_symbol(handle, "brh_step_registered");
    api.close = (close_fn) must_symbol(handle, "brh_close");
    api.last_error = (last_error_fn) must_symbol(handle, "brh_last_error");
    api.use_registered = use_registered;

    validate_bridge_abi(&api, &expected);
    api.set_data_dir(data_dir);
    void *observation = calloc(1, expected.observation_size);
    if (observation == NULL) {
        perror("calloc observation");
        return 2;
    }
    if (api.use_registered) {
        register_observation_buffers(&api, (brh_observation *) observation, observation_mask);
    }

    printf("Brogue C Bridge Direct Benchmark\n");
    printf("library: %s\n", library_path);
    printf("data dir: %s\n", data_dir);
    printf("observation: %zu bytes\n", expected.observation_size);
    printf("api: %s\n", api.use_registered ? "registered" : "legacy");
    printf("observation profile: %s (mask=0x%llx)\n",
           observation_profile,
           (unsigned long long) observation_mask);
    printf("profile: %s fixed trace lengths\n", profile->name);
    printf("seed count: %d; trace repeats per seed: %d\n", seed_count, trace_repeats);
    printf("tail_us is p95 for >=20 samples, otherwise max\n");
    printf("step cases include %s observation fill and export\n",
           api.use_registered ? observation_profile : "full");
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
    benchmark_steps(&api, "invalid-key", '!', 0, 0, profile->invalid_steps, profile->invalid_warmup, trace_repeats, seed_count, seed_start, 1, observation, expected.program_state_offset);
    benchmark_steps(&api, "rest", 'z', 0, 0, profile->rest_steps, profile->rest_warmup, trace_repeats, seed_count, seed_start, 1001, observation, expected.program_state_offset);
    benchmark_steps(&api, "search", 's', 0, 0, profile->search_steps, profile->search_warmup, trace_repeats, seed_count, seed_start, 2001, observation, expected.program_state_offset);
    benchmark_steps(&api, "explore", 'x', 0, 0, profile->explore_steps, profile->explore_warmup, trace_repeats, seed_count, seed_start, 3001, observation, expected.program_state_offset);
    benchmark_steps(&api, "fast-explore", 'x', 1, 0, profile->fast_explore_steps, profile->fast_explore_warmup, trace_repeats, seed_count, seed_start, 4001, observation, expected.program_state_offset);
    benchmark_resets(&api, profile->reset_count, trace_repeats, seed_count, seed_start, observation);

    free(observation);
    if (api.use_registered) {
        api.clear_observation_buffers();
    }
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
    api: str,
    obs_profile: str,
    trace_repeats: int,
    seed_count: int,
    seed_start: int,
) -> list[str]:
    """Return native harness arguments, including Python-side ABI expectations."""

    return [
        str(library),
        str(data_dir),
        profile,
        api,
        obs_profile,
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
        "--api",
        choices=("legacy", "registered"),
        default="legacy",
        help="Bridge API path to benchmark",
    )
    parser.add_argument(
        "--obs-profile",
        choices=("full", "screen"),
        default="full",
        help="Observation group mask for registered API runs",
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
                    args.api,
                    args.obs_profile,
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
                    args.api,
                    args.obs_profile,
                    args.trace_repeats,
                    args.seed_count,
                    args.seed_start,
                ),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()

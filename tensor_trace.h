#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "hash.h"
#include "tensor.h"

/*
 * Combined profiler + tensor trace.
 *
 * PROFILE=1: records cumulative time per named op (fnv1a hash table).
 * TRACE=1:   prints tensor debug string at each trace point.
 * TRACE=prefix: only print trace points matching prefix.
 *
 * tensor_trace(t, "name") records profiler timing (if PROFILE=1)
 * and prints trace output (if TRACE=1). t may be NULL for timing-only.
 */

#define PROFILER_HT_SIZE 512
#define PROFILER_HT_MASK (PROFILER_HT_SIZE - 1)

struct profiler_entry {
	const char *name;
	uint64_t total_ns;
};

extern uint64_t profiler_last;
extern struct profiler_entry profiler_entries[PROFILER_HT_SIZE];
extern int tensor_trace_on;
extern int profiler_on;

void tensor_trace_init(void);
void tensor_trace(const tensor_t *t, const char *fmt, ...)
	__attribute__((format(printf, 2, 3)));

static inline uint64_t profiler_now(void)
{
	struct timespec ts = {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static inline double profiler_to_sec(uint64_t t)
{
	return (double)t / 1000000000.0;
}

void tensor_trace_shutdown(void);

#include "tensor_trace.h"

#include <stdlib.h>
#include <stdarg.h>
#include <math.h>

uint64_t profiler_last;
struct profiler_entry profiler_entries[PROFILER_HT_SIZE] = {};
int tensor_trace_on = 0;
int profiler_on = 0;
static const char *tensor_trace_filter = NULL;

void tensor_trace_init(void)
{
	const char *env = getenv("TRACE");
	if (env) {
		tensor_trace_on = 1;
		if (strcmp(env, "1") != 0)
			tensor_trace_filter = env;
	}
	if (getenv("PROFILE")) {
		profiler_on = 1;
		profiler_last = profiler_now();
	}
}

static void profiler_record(const char *name)
{
	uint64_t now = profiler_now();
	uint64_t elapsed = now - profiler_last;
	profiler_last = now;

	uint32_t h = fnv1a_str(name) & PROFILER_HT_MASK;

	while (profiler_entries[h].name) {
		if (profiler_entries[h].name == name ||
		    strcmp(profiler_entries[h].name, name) == 0) {
			profiler_entries[h].total_ns += elapsed;
			return;
		}
		h = (h + 1) & PROFILER_HT_MASK;
	}

	profiler_entries[h].name = name;
	profiler_entries[h].total_ns = elapsed;
}

void tensor_trace(const tensor_t *t, const char *fmt, ...)
{
	char name[64];
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(name, sizeof(name), fmt, ap);
	va_end(ap);

	if (profiler_on)
		profiler_record(fmt);

	if (!tensor_trace_on)
		return;
	if (tensor_trace_filter && strncmp(name, tensor_trace_filter,
					   strlen(tensor_trace_filter)) != 0)
		return;
	if (!t || t->type != TENSOR_F32 || t->ndim < 1)
		return;

	char *dbg = tensor_to_debug_string(t);
	fprintf(stderr, "TRACE %s: %s\n", name, dbg);
	free(dbg);
}

void tensor_trace_shutdown(void)
{
	if (!profiler_on)
		return;

	uint64_t total = 0;
	for (int i = 0; i < PROFILER_HT_SIZE; i++) {
		if (!profiler_entries[i].name)
			continue;
		fprintf(stderr, "%.9fs %s\n",
			profiler_to_sec(profiler_entries[i].total_ns),
			profiler_entries[i].name);
		total += profiler_entries[i].total_ns;
	}
	fprintf(stderr, "total=%fs\n", profiler_to_sec(total));
}

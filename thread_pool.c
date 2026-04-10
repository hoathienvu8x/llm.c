#include "thread_pool.h"

#define _GNU_SOURCE
#include <assert.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#endif

/* No clear guidance on these, TODO - move to env/config? */
#define MIN_FLOPS_THREADED  (16 * 1024 * 1024)
#define MIN_FLOPS_PER_THREAD (512 * 1024)

struct thread_pool {
	int nthreads;
	pthread_t *threads;
	pthread_barrier_t start_barrier;
	pthread_barrier_t end_barrier;
	tp_work_fn fn;
	void *arg;
	int active_threads;
	int quit;
};

struct worker_arg {
	struct thread_pool *tp;
	int tidx;
};

static struct thread_pool g_tp;

static int detect_physical_cores(void)
{
	int seen[4096] = {};
	int ncpu = sysconf(_SC_NPROCESSORS_ONLN);
	int count = 0;

	for (int cpu = 0; cpu < ncpu; cpu++) {
		char path[128];
		int core_id, pkg = 0;
		FILE *f;

		snprintf(path, sizeof(path),
			 "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
		f = fopen(path, "r");
		if (!f)
			return 0;
		if (fscanf(f, "%d", &core_id) != 1) {
			fclose(f);
			return 0;
		}
		fclose(f);

		snprintf(path, sizeof(path),
			 "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
		f = fopen(path, "r");
		if (f) {
			fscanf(f, "%d", &pkg);
			fclose(f);
		}

		int key = pkg * 1024 + core_id;
		assert(key >= 0 && key < (int)ARRAY_SIZE(seen));
		if (!seen[key]) {
			seen[key] = 1;
			count++;
		}
	}
	return count;
}

static void *worker(void *p)
{
	struct worker_arg *wa = p;
	struct thread_pool *tp = wa->tp;
	int idx = wa->tidx;

	for (;;) {
		pthread_barrier_wait(&tp->start_barrier);
		if (tp->quit)
			return NULL;
		if (idx < tp->active_threads)
			tp->fn(tp->arg, idx, tp->active_threads);
		pthread_barrier_wait(&tp->end_barrier);
	}

	free(wa);
}

__attribute__((constructor))
static void thread_pool_init(void)
{
	int ncpu = sysconf(_SC_NPROCESSORS_ONLN);
	if (ncpu < 1) ncpu = 1;

	int phys = detect_physical_cores();
	if (phys > 0)
		ncpu = phys;

	g_tp.nthreads = ncpu;
	g_tp.threads = calloc(ncpu, sizeof(pthread_t));

	pthread_barrier_init(&g_tp.start_barrier, NULL, ncpu + 1);
	pthread_barrier_init(&g_tp.end_barrier, NULL, ncpu + 1);

	for (int i = 0; i < ncpu; i++) {
		struct worker_arg *wa = malloc(sizeof(*wa));
		wa->tp = &g_tp;
		wa->tidx = i;
		pthread_create(&g_tp.threads[i], NULL, worker, wa);
	}
}

__attribute__((destructor))
static void thread_pool_fini(void)
{
	g_tp.quit = 1;
	pthread_barrier_wait(&g_tp.start_barrier);

	for (int i = 0; i < g_tp.nthreads; i++)
		pthread_join(g_tp.threads[i], NULL);

	pthread_barrier_destroy(&g_tp.start_barrier);
	pthread_barrier_destroy(&g_tp.end_barrier);
	free(g_tp.threads);
}

static int pick_nthreads(size_t m, size_t k, size_t n)
{
	size_t flops = 2 * m * k * n;
	if (flops < MIN_FLOPS_THREADED)
		return 1;

	int by_flops = flops / MIN_FLOPS_PER_THREAD;
	int nt = by_flops < n ? by_flops : n;
	if (nt > g_tp.nthreads)
		nt = g_tp.nthreads;
	if (nt < 1)
		nt = 1;
	return nt;
}

void thread_pool_run(tp_work_fn fn, void *arg,
		     size_t m, size_t k, size_t n)
{
	int nt = pick_nthreads(m, k, n);

	if (nt <= 1) {
		fn(arg, 0, 1);
		return;
	}

	g_tp.fn = fn;
	g_tp.arg = arg;
	g_tp.active_threads = nt;
	pthread_barrier_wait(&g_tp.start_barrier);
	pthread_barrier_wait(&g_tp.end_barrier);
}

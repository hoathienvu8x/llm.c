#define _GNU_SOURCE
#include "thread_pool.h"

#include <assert.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>

#ifdef __x86_64__
#include <immintrin.h>
#define cpu_pause() _mm_pause()
#else
#define cpu_pause() ((void)0)
#endif

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#endif

/* No clear guidance on these, TODO - move to env/config? */
#define MIN_FLOPS_THREADED  (16 * 1024 * 1024)
#define MIN_FLOPS_PER_THREAD (512 * 1024)
#define CACHE_LINE 64

struct thread_pool {
	int nthreads;
	pthread_t *threads;
	int *cpus;

	tp_work_fn fn;
	void *arg;
	int active_threads;
	int quit;

	_Alignas(CACHE_LINE) atomic_int generation;
	_Alignas(CACHE_LINE) atomic_int n_done;
};

static struct thread_pool g_tp;

static inline void spin_wait_done(int n)
{
	while (atomic_load_explicit(&g_tp.n_done, memory_order_acquire) < n)
		cpu_pause();
}

static int detect_cores(int *cpus, int max)
{
	int seen[4096] = {};
	int ncpu = sysconf(_SC_NPROCESSORS_ONLN);
	int count = 0;

	for (int cpu = 0; cpu < ncpu && count < max; cpu++) {
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
			if (cpus)
				cpus[count] = cpu;
			count++;
		}
	}
	return count;
}

static void *worker(void *p)
{
	int idx = (int)(intptr_t)p;
	struct thread_pool *tp = &g_tp;
	int gen = 0;

	if (tp->cpus) {
		cpu_set_t cpuset;
		CPU_ZERO(&cpuset);
		CPU_SET(tp->cpus[idx], &cpuset);
		pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
	}

	for (;;) {
		while (atomic_load_explicit(&tp->generation,
					    memory_order_acquire) == gen)
			cpu_pause();
		gen++;

		if (tp->quit)
			return NULL;

		if (idx < tp->active_threads)
			tp->fn(tp->arg, idx, tp->active_threads);

		atomic_fetch_add_explicit(&tp->n_done, 1, memory_order_release);
	}
}

__attribute__((constructor))
static void thread_pool_init(void)
{
	int ncpu = sysconf(_SC_NPROCESSORS_ONLN);
	if (ncpu < 1) ncpu = 1;

	int *cpus = calloc(ncpu, sizeof(int));
	int phys = detect_cores(cpus, ncpu);
	if (phys > 0)
		ncpu = phys;
	else
		for (int i = 0; i < ncpu; i++)
			cpus[i] = i;

	g_tp.nthreads = ncpu;
	g_tp.threads = calloc(ncpu, sizeof(pthread_t));
	g_tp.cpus = cpus;
	atomic_store(&g_tp.generation, 0);
	atomic_store(&g_tp.n_done, 0);

	for (int i = 0; i < ncpu; i++)
		pthread_create(&g_tp.threads[i], NULL, worker,
			       (void *)(intptr_t)i);
}

__attribute__((destructor))
static void thread_pool_fini(void)
{
	g_tp.quit = 1;
	atomic_fetch_add_explicit(&g_tp.generation, 1, memory_order_release);

	for (int i = 0; i < g_tp.nthreads; i++)
		pthread_join(g_tp.threads[i], NULL);

	free(g_tp.threads);
	free(g_tp.cpus);
}

static int pick_nthreads(size_t m, size_t k, size_t n)
{
	size_t flops = 2 * m * k * n;
	if (flops < MIN_FLOPS_THREADED)
		return 1;

	return g_tp.nthreads;
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
	atomic_store_explicit(&g_tp.n_done, 0, memory_order_release);

	atomic_fetch_add_explicit(&g_tp.generation, 1, memory_order_release);

	spin_wait_done(g_tp.nthreads);
}

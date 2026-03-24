#pragma once

#include <stddef.h>

typedef void (*tp_work_fn)(void *arg, int tidx, int nthreads);

void thread_pool_run(tp_work_fn fn, void *arg, size_t m, size_t k, size_t n);

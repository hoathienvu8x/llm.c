#include "matmul.h"
#include "thread_pool.h"
#include "quant.h"
#include "simd.h"

#include <stdlib.h>
#include <string.h>

#ifdef USE_CBLAS
#include <cblas.h>
#endif

static void mma_init(tensor_t *ret, const tensor_t *add, size_t m, size_t n)
{
	ret->ndim = 2;
	ret->dim[0] = m;
	ret->dim[1] = n;
	ret->totlen = m * n;

	if (!add) {
		memset(ret->data, 0, m * n * sizeof(scalar_t));
		return;
	}

	if (add->totlen == m * n) {
		memcpy(ret->data, add->data, m * n * sizeof(scalar_t));
	} else {
		assert(add->totlen == n);
		for (size_t i = 0; i < m; i++)
			memcpy(&ret->data[i * n], add->data,
			       n * sizeof(scalar_t));
	}
}

struct matmul_work {
	scalar_t *result;
	const scalar_t *lhs;
	const tensor_t *rhs;
	size_t m, k, n;
};

static void nontrans_thread_fn(void *arg, int tidx, int nthreads)
{
	struct matmul_work *w = arg;
	size_t m = w->m, k = w->k, n = w->n;
	const scalar_t *lhs = w->lhs;
	const scalar_t *rhs = w->rhs->data;
	scalar_t *result = w->result;

	size_t chunk = (m + nthreads - 1) / nthreads;
	size_t i_start = tidx * chunk;
	size_t i_end = i_start + chunk;
	if (i_end > m) i_end = m;
	if (i_start >= m) return;

	for_each_vec(j0, n) {
		for (size_t i = i_start; i < i_end; i++) {
			vector_t acc;
			vector_load(&acc, &result[i * n + j0]);

			for (size_t kk = 0; kk < k; kk++) {
				vector_t bv, av;
				vector_set(&av, lhs[i * k + kk]);
				vector_load(&bv, (scalar_t *)&rhs[kk * n + j0]);
				vector_fma(&acc, &av, &bv, &acc);
			}
			vector_store(&result[i * n + j0], &acc);
		}
	}

	for (size_t j = vector_batches(n); j < n; j++) {
		for (size_t i = i_start; i < i_end; i++) {
			scalar_t sum = result[i * n + j];
			for (size_t kk = 0; kk < k; kk++)
				sum += lhs[i * k + kk] * rhs[kk * n + j];
			result[i * n + j] = sum;
		}
	}
}

struct gemv_work {
	const scalar_t *lhs;
#ifdef Q8_DOT
	const block_q8_K *q8;
#endif
	const tensor_t *rhs;
	scalar_t *result;
	size_t k, n;
};

static void transposed_gemv_thread_fn(void *arg, int tidx, int nthreads)
{
	struct gemv_work *w = arg;
	size_t chunk = (w->n + nthreads - 1) / nthreads;
	size_t j_start = tidx * chunk;
	size_t j_end = j_start + chunk;
	if (j_end > w->n) j_end = w->n;

	for (size_t j = j_start; j < j_end; j++)
#ifdef Q8_DOT
		w->result[j] += dot_q8_quant(w->q8, w->lhs,
					     w->rhs->qdata,
					     w->rhs->type, j, w->k);
#else
		w->result[j] += dot_f32_quant(w->lhs,
					      w->rhs->qdata,
					      w->rhs->type, j, w->k);
#endif
}

/* result[m,n] = lhs[m,k] @ rhs[n,k].T parallelized along n */
static void transposed_thread_fn(void *arg, int tidx, int nthreads)
{
	struct matmul_work *w = arg;
	size_t m = w->m, k = w->k, n = w->n;
	const scalar_t *lhs = w->lhs;
	scalar_t *result = w->result;

	size_t chunk = (n + nthreads - 1) / nthreads;
	size_t j_start = tidx * chunk;
	size_t j_end = j_start + chunk;
	if (j_end > n) j_end = n;
	if (j_start >= n) return;

	scalar_t *scratch = NULL;
	if (w->rhs->type != TENSOR_F32) {
		scratch = aligned_alloc(VECTOR_ALIGN,
			MMA_NR * k * sizeof(scalar_t));
	}

	for (size_t j0 = j_start; j0 < j_end; j0 += MMA_NR) {
		size_t nr = MMA_NR;
		if (j0 + nr > j_end) nr = j_end - j0;

		const scalar_t *b;
		if (w->rhs->type != TENSOR_F32) {
			for (size_t j = 0; j < nr; j++)
				dequant_row(w->rhs->qdata, w->rhs->type,
					    j0 + j, &scratch[j * k], k);
			b = scratch;
		} else {
			b = &w->rhs->data[j0 * k];
		}

		for (size_t i0 = 0; i0 < m; i0 += MMA_MR) {
			size_t mr = MMA_MR;
			if (i0 + mr > m) mr = m - i0;

			/* k-outer, j-inner: load each lhs vector once,
			 * broadcast across MMA_NR rhs rows. */
			vector_t acc[MMA_MR * MMA_NR];
			for (size_t r = 0; r < mr; r++)
				for (size_t j = 0; j < nr; j++)
					vector_set(&acc[r * MMA_NR + j], 0);

			for_each_vec(kk, k) {
				for (size_t r = 0; r < mr; r++) {
					vector_t av;
					vector_load(&av,
						(scalar_t *)&lhs[(i0 + r) * k + kk]);
					for (size_t j = 0; j < nr; j++) {
						vector_t bv;
						vector_load(&bv,
							(scalar_t *)&b[j * k + kk]);
						vector_fma(&acc[r * MMA_NR + j],
							   &av, &bv,
							   &acc[r * MMA_NR + j]);
					}
				}
			}

			for (size_t r = 0; r < mr; r++) {
				for (size_t j = 0; j < nr; j++) {
					scalar_t sum = vector_reduce_sum(
						&acc[r * MMA_NR + j]);
					for (size_t kk = vector_batches(k);
					     kk < k; kk++) {
						sum += lhs[(i0 + r) * k + kk] *
						       b[j * k + kk];
					}
					result[(i0 + r) * n + j0 + j] += sum;
				}
			}
		}
	}

	free(scratch);
}

void tensor_mma_2x2_tp(
	tensor_t *ret,
	const tensor_t *lhs,
	const tensor_t *rhs,
	const tensor_t *add)
{
	assert(ret != lhs && ret != rhs);
	assert(rhs->ndim == 2 && lhs->ndim == 2);
	assert(lhs->dim[1] == rhs->dim[0]);

	size_t m = lhs->dim[0];
	size_t k = lhs->dim[1];
	size_t n = rhs->dim[1];

	mma_init(ret, add, m, n);

	struct matmul_work w = {
		.result = ret->data, .lhs = lhs->data, .rhs = rhs,
		.m = m, .k = k, .n = n,
	};
	thread_pool_run(nontrans_thread_fn, &w, m, k, n);
}

void tensor_mma_transposed_2x2_tp(
	tensor_t *ret,
	const tensor_t *lhs,
	const tensor_t *rhs,
	const tensor_t *add)
{
	assert(ret != lhs && ret != rhs);
	assert(rhs->ndim == 2 && lhs->ndim == 2);
	assert(lhs->dim[1] == rhs->dim[1]);

	size_t m = lhs->dim[0];
	size_t k = lhs->dim[1];
	size_t n = rhs->dim[0];

	mma_init(ret, add, m, n);

	struct matmul_work w = {
		.result = ret->data, .lhs = lhs->data, .rhs = rhs,
		.m = m, .k = k, .n = n,
	};

	if (m == 1 && rhs->type != TENSOR_F32) {
		struct gemv_work gw = {
			.lhs = lhs->data, .rhs = rhs,
			.result = ret->data, .k = k, .n = n,
		};
#ifdef Q8_DOT
		if (k % QK_K == 0) {
			block_q8_K *q8 = tensor_scratch(lhs, (k / QK_K) * sizeof(block_q8_K));
			quantize_row_q8(lhs->data, q8, k);
			gw.q8 = q8;
		}
#endif
		thread_pool_run(transposed_gemv_thread_fn, &gw, m, k, n);
	} else {
		thread_pool_run(transposed_thread_fn, &w, m, k, n);
	}
}

#ifdef USE_CBLAS
void tensor_mma_2x2_cblas(
	tensor_t *ret,
	const tensor_t *lhs,
	const tensor_t *rhs,
	const tensor_t *add)
{
	assert(ret != lhs && ret != rhs);
	assert(rhs->ndim == 2 && lhs->ndim == 2);
	assert(lhs->dim[1] == rhs->dim[0]);

	size_t m = lhs->dim[0];
	size_t k = lhs->dim[1];
	size_t n = rhs->dim[1];
	float beta = 0.0;

	mma_init(ret, add, m, n);
	if (add) beta = 1.0;

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		    m, n, k, 1.0,
		    lhs->data, k,
		    rhs->data, n,
		    beta, ret->data, n);
}

void tensor_mma_transposed_2x2_cblas(
	tensor_t *ret,
	const tensor_t *lhs,
	const tensor_t *rhs,
	const tensor_t *add)
{
	assert(ret != lhs && ret != rhs);
	assert(rhs->ndim == 2 && lhs->ndim == 2);
	assert(lhs->dim[1] == rhs->dim[1]);

	size_t m = lhs->dim[0];
	size_t k = lhs->dim[1];
	size_t n = rhs->dim[0];

	mma_init(ret, add, m, n);

	if (rhs->type == TENSOR_F32) {
		float beta = add ? 1.0 : 0.0;

		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
			    m, n, k, 1.0,
			    lhs->data, k,
			    rhs->data, rhs->dim[1],
			    beta, ret->data, n);
	} else {
		struct matmul_work w = {
			.result = ret->data, .lhs = lhs->data, .rhs = rhs,
			.m = m, .k = k, .n = n,
		};
		thread_pool_run(transposed_thread_fn, &w, m, k, n);
	}
}
#endif

void tensor_mma_2x2_naive(
	tensor_t *ret,
	const tensor_t *lhs,
	const tensor_t *rhs,
	const tensor_t *add)
{
	assert(ret != lhs && ret != rhs);
	assert(rhs->ndim == 2 && lhs->ndim == 2);
	assert(lhs->dim[1] == rhs->dim[0]);

	size_t m = lhs->dim[0];
	size_t k = lhs->dim[1];
	size_t n = rhs->dim[1];

	mma_init(ret, add, m, n);

	for (size_t i = 0; i < m; i++)
		for (size_t j = 0; j < n; j++) {
			scalar_t sum = 0;
			for (size_t kk = 0; kk < k; kk++)
				sum += lhs->data[i * k + kk] *
				       rhs->data[kk * n + j];
			ret->data[i * n + j] += sum;
		}
}

void tensor_mma_transposed_2x2_naive(
	tensor_t *ret,
	const tensor_t *lhs,
	const tensor_t *rhs,
	const tensor_t *add)
{
	assert(ret != lhs && ret != rhs);
	assert(rhs->ndim == 2 && lhs->ndim == 2);
	assert(lhs->dim[1] == rhs->dim[1]);

	size_t m = lhs->dim[0];
	size_t k = lhs->dim[1];
	size_t n = rhs->dim[0];

	mma_init(ret, add, m, n);

	for (size_t i = 0; i < m; i++)
		for (size_t j = 0; j < n; j++) {
			scalar_t sum = 0;
			for (size_t kk = 0; kk < k; kk++)
				sum += lhs->data[i * k + kk] *
				       rhs->data[j * k + kk];
			ret->data[i * n + j] += sum;
		}
}

void tensor_mma_2x2(
	tensor_t *ret,
	const tensor_t *lhs,
	const tensor_t *rhs,
	const tensor_t *add)
{
#ifdef USE_CBLAS
	tensor_mma_2x2_cblas(ret, lhs, rhs, add);
#else
	tensor_mma_2x2_tp(ret, lhs, rhs, add);
#endif
}

void tensor_mma_transposed_2x2(
	tensor_t *ret,
	const tensor_t *lhs,
	const tensor_t *rhs,
	const tensor_t *add)
{
#ifdef USE_CBLAS
	tensor_mma_transposed_2x2_cblas(ret, lhs, rhs, add);
#else
	tensor_mma_transposed_2x2_tp(ret, lhs, rhs, add);
#endif
}

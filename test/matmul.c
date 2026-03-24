#include "tensor.h"
#include "matmul.h"
#include "quant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "test/test.h"

static void quantize_q8_0_row(const float *src, block_q8_0 *dst, size_t n)
{
	for (size_t i = 0; i < n / GGML_QK; i++) {
		float amax = 0.0f;
		for (int j = 0; j < GGML_QK; j++) {
			float v = fabsf(src[i * GGML_QK + j]);
			if (v > amax) amax = v;
		}
		float d = amax / 127.0f;
		float id = d ? 1.0f / d : 0.0f;

		union { float f; uint32_t u; } u = { .f = d };
		uint16_t f16 = ((u.u >> 16) & 0x8000) |
			       (((u.u >> 23) - 127 + 15) << 10) |
			       ((u.u >> 13) & 0x3FF);
		dst[i].d = f16;
		for (int j = 0; j < GGML_QK; j++) {
			int v = roundf(src[i * GGML_QK + j] * id);
			if (v > 127) v = 127;
			if (v < -128) v = -128;
			dst[i].qs[j] = v;
		}
	}
}

static void fill_random(float *data, size_t n)
{
	for (size_t i = 0; i < n; i++)
		data[i] = (float)drand48() * 2.0f - 1.0f;
}

static void bench_transposed(size_t M, size_t K, size_t N, int rounds)
{
	tensor_t *lhs = tensor_new_zero(2, M, K);
	tensor_t *ret = tensor_new_zero(2, M, N);
	fill_random(lhs->data, M * K);

	tensor_t *rhs_f32 = tensor_new_zero(2, N, K);
	fill_random(rhs_f32->data, N * K);

	size_t nblocks = K / GGML_QK;
	block_q8_0 *q8_data = aligned_alloc(VECTOR_ALIGN,
		N * nblocks * sizeof(block_q8_0));
	for (size_t i = 0; i < N; i++)
		quantize_q8_0_row(&rhs_f32->data[i * K],
				  &q8_data[i * nblocks], K);
	tensor_t *rhs_q8 = tensor_new_mapped(q8_data, N * K, TENSOR_Q8_0);
	rhs_q8->ndim = 2;
	rhs_q8->dim[0] = N;
	rhs_q8->dim[1] = K;

	char name[64];
	snprintf(name, sizeof(name), "%zux%zu @ %zux%zu.T", M, K, N, K);
	bench_begin(name);

	uint64_t start, t, base;

#ifdef USE_CBLAS
	start = now();
	for (int i = 0; i < rounds; i++)
		tensor_mma_transposed_2x2_cblas(ret, lhs, rhs_f32, NULL);
	base = now() - start;
	bench_entry("f32_cblas", rounds, base, 0);
#else
	base = 0;
#endif

	start = now();
	for (int i = 0; i < rounds; i++)
		tensor_mma_transposed_2x2_tp(ret, lhs, rhs_f32, NULL);
	t = now() - start;
	bench_entry("f32_tp", rounds, t, base);

	start = now();
	for (int i = 0; i < rounds; i++)
		tensor_mma_transposed_2x2_tp(ret, lhs, rhs_q8, NULL);
	t = now() - start;
	bench_entry("q8_tp", rounds, t, base);

	bench_end();

	tensor_free(lhs);
	tensor_free(ret);
	tensor_free(rhs_f32);
	free(q8_data);
	free((void *)rhs_q8);
}

static void bench_nontransposed(size_t M, size_t K, size_t N, int rounds)
{
	tensor_t *lhs = tensor_new_zero(2, M, K);
	tensor_t *rhs = tensor_new_zero(2, K, N);
	tensor_t *ret = tensor_new_zero(2, M, N);
	fill_random(lhs->data, M * K);
	fill_random(rhs->data, K * N);

	char name[64];
	snprintf(name, sizeof(name), "%zux%zu @ %zux%zu", M, K, K, N);
	bench_begin(name);

	uint64_t start, t, base;

#ifdef USE_CBLAS
	start = now();
	for (int i = 0; i < rounds; i++)
		tensor_mma_2x2_cblas(ret, lhs, rhs, NULL);
	base = now() - start;
	bench_entry("f32_cblas", rounds, base, 0);
#else
	base = 0;
#endif

	start = now();
	for (int i = 0; i < rounds; i++)
		tensor_mma_2x2_tp(ret, lhs, rhs, NULL);
	t = now() - start;
	bench_entry("f32_tp", rounds, t, base);

	bench_end();

	tensor_free(lhs);
	tensor_free(rhs);
	tensor_free(ret);
}

int main(void)
{
	srand48(42);

	bench_transposed(1,  768, 768,  1000);
	bench_transposed(1,  768, 2304, 1000);
	bench_transposed(1,  768, 3072, 1000);
	bench_transposed(8,  768, 768,  500);
	bench_transposed(8,  768, 3072, 500);
	bench_transposed(64, 768, 768,  200);
	bench_transposed(64, 768, 3072, 200);

	bench_nontransposed(1,   64,  64,  2000);
	bench_nontransposed(64,  64,  64,  1000);
	bench_nontransposed(128, 128, 128, 500);
	bench_nontransposed(64,  768, 768, 200);

	return 0;
}

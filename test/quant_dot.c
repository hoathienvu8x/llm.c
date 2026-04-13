#include "quant.h"
#include "quant_cpu.h"
#if defined(__AVX2__)
#include "quant_avx2.h"
#endif
#include "tensor.h"
#include "test/test.h"

#include <math.h>

/* fill test data with deterministic patterns */
static void fill_q4_K(block_q4_K *qw, size_t nb)
{
	for (size_t i = 0; i < nb; i++) {
		qw[i].d = 0x3C00;    /* 1.0 in f16 */
		qw[i].dmin = 0x3800;  /* 0.5 in f16 */
		for (int j = 0; j < 12; j++)
			qw[i].scales[j] = 5 + (j * 3) % 40;
		for (int j = 0; j < 128; j++)
			qw[i].qs[j] = ((i + j) * 7 + 3) & 0xFF;
	}
}

static void fill_q6_K(block_q6_K *qw, size_t nb)
{
	for (size_t i = 0; i < nb; i++) {
		qw[i].d = 0x3C00;
		for (int j = 0; j < 128; j++)
			qw[i].ql[j] = ((i + j) * 11 + 7) & 0xFF;
		for (int j = 0; j < 64; j++)
			qw[i].qh[j] = ((i + j) * 5 + 2) & 0xFF;
		for (int j = 0; j < 16; j++)
			qw[i].scales[j] = (int8_t)(((i + j) * 3) % 61 - 30);
	}
}

static void test_q8k_roundtrip(void)
{
	float x[256];
	for (int i = 0; i < 256; i++)
		x[i] = sinf(i * 0.1f) * 0.5f;

	block_q8_K q8;
	quantize_row_q8(x, &q8, 256);

	double max_err = 0;
	for (int i = 0; i < 256; i++) {
		double err = fabs(x[i] - q8.d * q8.qs[i]);
		if (err > max_err)
			max_err = err;
	}
	assert(max_err < 0.01);

	for (int j = 0; j < 16; j++) {
		int sum = 0;
		for (int k = 0; k < 16; k++)
			sum += q8.qs[j * 16 + k];
		assert(q8.bsums[j] == sum);
	}
	printf("  q8k roundtrip: ok (max_err=%.6f)\n", max_err);
}

static void test_cpu_q4k(void)
{
	size_t K = 256;
	size_t nb = K / QK_K;
	block_q4_K qw[1];
	fill_q4_K(qw, nb);

	float x[256];
	for (int i = 0; i < 256; i++)
		x[i] = sinf(i * 0.037f) * 0.1f;

	block_q8_K q8[1];
	quantize_row_q8(x, q8, K);

	float ref = dot_f32_quant(x, qw, TENSOR_Q4_K, 0, K);
	float cpu = cpu_dot_q4_K_q8_K(qw, q8, K);
	float rel = fabsf(cpu - ref) / (fabsf(ref) + 1e-9f);
	assert(rel < 0.02f);
	printf("  cpu q4k dot: ok (ref=%.4f cpu=%.4f rel=%.6f)\n", ref, cpu, rel);
}

static void test_cpu_q6k(void)
{
	size_t K = 256;
	size_t nb = K / QK_K;
	block_q6_K qw[1];
	fill_q6_K(qw, nb);

	float x[256];
	for (int i = 0; i < 256; i++)
		x[i] = sinf(i * 0.037f) * 0.1f;

	block_q8_K q8[1];
	quantize_row_q8(x, q8, K);

	float ref = dot_f32_quant(x, qw, TENSOR_Q6_K, 0, K);
	float cpu = cpu_dot_q6_K_q8_K(qw, q8, K);
	float rel = fabsf(cpu - ref) / (fabsf(ref) + 1e-9f);
	assert(rel < 0.02f);
	printf("  cpu q6k dot: ok (ref=%.4f cpu=%.4f rel=%.6f)\n", ref, cpu, rel);
}

#if defined(__AVX2__)
static void test_avx2_q4k(void)
{
	size_t K = 4096;
	size_t nb = K / QK_K;
	block_q4_K *qw = calloc(nb, sizeof(block_q4_K));
	fill_q4_K(qw, nb);

	float *x = aligned_alloc(64, K * sizeof(float));
	for (size_t i = 0; i < K; i++)
		x[i] = sinf(i * 0.037f) * 0.1f;

	block_q8_K *q8 = aligned_alloc(64, nb * sizeof(block_q8_K));
	quantize_row_q8(x, q8, K);

	float cpu = cpu_dot_q4_K_q8_K(qw, q8, K);
	float avx = avx2_dot_q4_K_q8_K(qw, q8, K);
	assert(cpu == avx);
	printf("  avx2 q4k dot: ok (cpu=%.4f avx2=%.4f)\n", cpu, avx);

	free(qw);
	free(x);
	free(q8);
}

static void test_avx2_q6k(void)
{
	size_t K = 4096;
	size_t nb = K / QK_K;
	block_q6_K *qw = calloc(nb, sizeof(block_q6_K));
	fill_q6_K(qw, nb);

	float *x = aligned_alloc(64, K * sizeof(float));
	for (size_t i = 0; i < K; i++)
		x[i] = sinf(i * 0.037f) * 0.1f;

	block_q8_K *q8 = aligned_alloc(64, nb * sizeof(block_q8_K));
	quantize_row_q8(x, q8, K);

	float cpu = cpu_dot_q6_K_q8_K(qw, q8, K);
	float avx = avx2_dot_q6_K_q8_K(qw, q8, K);
	assert(cpu == avx);
	printf("  avx2 q6k dot: ok (cpu=%.4f avx2=%.4f)\n", cpu, avx);

	free(qw);
	free(x);
	free(q8);
}
#endif

static void bench_dot(int rounds)
{
	size_t K = 4096;
	size_t nb = K / QK_K;

	block_q4_K *q4w = calloc(nb, sizeof(block_q4_K));
	block_q6_K *q6w = calloc(nb, sizeof(block_q6_K));
	float *x = aligned_alloc(64, K * sizeof(float));
	fill_q4_K(q4w, nb);
	fill_q6_K(q6w, nb);
	for (size_t i = 0; i < K; i++)
		x[i] = sinf(i * 0.037f) * 0.1f;

	block_q8_K *q8 = aligned_alloc(64, nb * sizeof(block_q8_K));
	quantize_row_q8(x, q8, K);

	volatile float sink = 0;
	uint64_t t0, dt_f32, dt_q8k;

	bench_begin("q4k dot (4096)");
	t0 = now();
	for (int r = 0; r < rounds; r++)
		sink = dot_f32_quant(x, q4w, TENSOR_Q4_K, 0, K);
	dt_f32 = now() - t0;
	bench_entry("f32", rounds, dt_f32, 0);
	t0 = now();
	for (int r = 0; r < rounds; r++)
		sink = dot_q8_quant(q8, x, q4w, TENSOR_Q4_K, 0, K);
	dt_q8k = now() - t0;
	bench_entry("q8k", rounds, dt_q8k, dt_f32);
	bench_end();

	bench_begin("q6k dot (4096)");
	t0 = now();
	for (int r = 0; r < rounds; r++)
		sink = dot_f32_quant(x, q6w, TENSOR_Q6_K, 0, K);
	dt_f32 = now() - t0;
	bench_entry("f32", rounds, dt_f32, 0);
	t0 = now();
	for (int r = 0; r < rounds; r++)
		sink = dot_q8_quant(q8, x, q6w, TENSOR_Q6_K, 0, K);
	dt_q8k = now() - t0;
	bench_entry("q8k", rounds, dt_q8k, dt_f32);
	bench_end();

	free(q4w);
	free(q6w);
	free(x);
	free(q8);
	(void)sink;
}

int main(void)
{
	printf("quant dot tests:\n");
	test_q8k_roundtrip();
	test_cpu_q4k();
	test_cpu_q6k();
#if defined(__AVX2__)
	test_avx2_q4k();
	test_avx2_q6k();
#endif
	printf("quant dot: all tests passed\n\n");

	bench_dot(1000000);

	return 0;
}

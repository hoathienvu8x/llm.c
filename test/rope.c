#include "nn.h"
#include "tensor.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>

/*
 * RoPE test: verify our rope_apply matches ggml's ROPE_TYPE_NORM
 * (consecutive pairs) by computing expected values manually.
 *
 * Standard RoPE rotates consecutive pairs (2i, 2i+1):
 *   out[2i]   = x[2i]   * cos(pos * freq_i) - x[2i+1] * sin(pos * freq_i)
 *   out[2i+1] = x[2i]   * sin(pos * freq_i) + x[2i+1] * cos(pos * freq_i)
 * where freq_i = 1 / theta^(2i / head_dim)
 */

static void rope_reference(float *head, int pos, int hlen, float theta)
{
	int half = hlen / 2;
	for (int i = 0; i < half; i++) {
		float freq = 1.0f / powf(theta, (float)(2 * i) / (float)hlen);
		float angle = (float)pos * freq;
		float cos_a = cosf(angle);
		float sin_a = sinf(angle);

		float x0 = head[2*i];
		float x1 = head[2*i + 1];
		head[2*i]     = x0 * cos_a - x1 * sin_a;
		head[2*i + 1] = x0 * sin_a + x1 * cos_a;
	}
}

static void test_rope_basic(void)
{
	/* Single head, HLEN=8, position 0: should be identity */
	float data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
	float orig[8];
	memcpy(orig, data, sizeof(data));

	tensor_t t = { .data = data, .ndim = 2, .dim = {1, 8}, .totlen = 8 };
	rope_apply(&t, 0, 8, 10000.0f);

	/* pos=0: angle=0, cos=1, sin=0, so output should equal input */
	for (int i = 0; i < 8; i++)
		assert(fabsf(data[i] - orig[i]) < 1e-6f);
	printf("  rope pos=0 identity: ok\n");
}

static void test_rope_rotation(void)
{
	/* Single head, HLEN=8, position 5, theta=10000 */
	float data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
	float expected[8];
	memcpy(expected, data, sizeof(data));
	rope_reference(expected, 5, 8, 10000.0f);

	tensor_t t = { .data = data, .ndim = 2, .dim = {1, 8}, .totlen = 8 };
	rope_apply(&t, 5, 8, 10000.0f);

	for (int i = 0; i < 8; i++) {
		assert(fabsf(data[i] - expected[i]) < 1e-5f);
	}
	printf("  rope pos=5 rotation: ok\n");
}

static void test_rope_preserves_norm(void)
{
	/* RoPE is a rotation: it should preserve the vector norm */
	float data[128];
	for (int i = 0; i < 128; i++)
		data[i] = sinf(i * 0.1f);

	float norm_before = 0;
	for (int i = 0; i < 128; i++)
		norm_before += data[i] * data[i];

	tensor_t t = { .data = data, .ndim = 2, .dim = {1, 128}, .totlen = 128 };
	rope_apply(&t, 13, 128, 1000000.0f);

	float norm_after = 0;
	for (int i = 0; i < 128; i++)
		norm_after += data[i] * data[i];

	assert(fabsf(norm_before - norm_after) < 1e-4f);
	printf("  rope preserves norm: ok (%.6f -> %.6f)\n", norm_before, norm_after);
}

static void test_rope_consecutive_pairs(void)
{
	/* Verify RoPE rotates consecutive pairs (0,1), (2,3), (4,5), ...
	 * NOT halved pairs (0,4), (1,5), (2,6), ... (GPT-NeoX style).
	 * This is ROPE_TYPE_NORM in llama.cpp. */
	float data[8] = {1, 0, 0, 0, 0, 0, 0, 0};

	/* With only x[0]=1, after RoPE:
	 * pair (0,1): x0=1, x1=0
	 *   out[0] = 1*cos - 0*sin = cos(angle_0)
	 *   out[1] = 1*sin + 0*cos = sin(angle_0)
	 * All other pairs: x0=0, x1=0 -> output = 0 */
	tensor_t t = { .data = data, .ndim = 2, .dim = {1, 8}, .totlen = 8 };
	rope_apply(&t, 7, 8, 10000.0f);

	float freq0 = 1.0f / powf(10000.0f, 0.0f / 8.0f); /* = 1.0 */
	float angle0 = 7.0f * freq0; /* = 7.0 */
	float expected_0 = cosf(angle0);
	float expected_1 = sinf(angle0);

	assert(fabsf(data[0] - expected_0) < 1e-6f);
	assert(fabsf(data[1] - expected_1) < 1e-6f);
	/* Elements 2-7 should be 0 (other pairs had zero input) */
	for (int i = 2; i < 8; i++)
		assert(fabsf(data[i]) < 1e-6f);

	printf("  rope consecutive pairs: ok (out[0]=%.4f==cos(7), out[1]=%.4f==sin(7))\n",
		data[0], data[1]);
}

int main(void)
{
	printf("rope tests:\n");
	test_rope_basic();
	test_rope_rotation();
	test_rope_preserves_norm();
	test_rope_consecutive_pairs();
	printf("rope: all tests passed\n");
	return 0;
}

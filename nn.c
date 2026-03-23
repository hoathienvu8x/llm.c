#include "nn.h"
#include "simd.h"

#include <math.h>

void rms_norm(tensor_t *ln, tensor_t *tmp_mat, const tensor_t *weight)
{
	for (size_t i = 0; i < tensor_len(tmp_mat); i++) {
		tensor_t row;

		tensor_at(tmp_mat, i, &row);

		vector_t s;
		vector_set(&s, 0);

		size_t len = tensor_len(&row);

		for (size_t j = 0; j < vector_batches(len); j += VECTOR_BATCH) {
			vector_t tmp;

			vector_load(&tmp, &row.data[j]);
			vector_mul(&tmp, &tmp, &tmp);
			vector_add(&s, &s, &tmp);
		}
		scalar_t sum = vector_reduce_sum(&s);
		for (size_t j = vector_batches(len); j < len; j++)
			sum += row.data[j] * row.data[j];

		scalar_t rms = sqrtf(sum / len + 1e-5f);

		vector_t vrms;
		vector_set(&vrms, rms);

		for (size_t j = 0; j < vector_batches(len); j += VECTOR_BATCH) {
			vector_t tmp;

			vector_load(&tmp, &row.data[j]);
			vector_div(&tmp, &tmp, &vrms);
			vector_store(&row.data[j], &tmp);
		}
		for (size_t j = vector_batches(len); j < len; j++)
			row.data[j] = row.data[j] / rms;

		tensor_t ln_row;
		tensor_at(ln, i, &ln_row);

		tensor_mul(&ln_row, &row, weight);
	}
}

void layer_norm(
	tensor_t *ln,
	tensor_t *tmp_mat,
	const tensor_t *weight,
	const tensor_t *bias)
{
	for (size_t i = 0; i < tensor_len(tmp_mat); i++) {
		tensor_t row;

		tensor_at(tmp_mat, i, &row);

		scalar_t row_mean = tensor_mean(&row);

		vector_t s, e;

		vector_set(&s, 0);
		vector_set(&e, row_mean);

		size_t len = tensor_len(&row);

		for (size_t j = 0; j < vector_batches(len); j += VECTOR_BATCH) {
			vector_t tmp;

			vector_load(&tmp, &row.data[j]);
			vector_sub(&tmp, &tmp, &e);
			vector_mul(&tmp, &tmp, &tmp);
			vector_add(&s, &s, &tmp);
		}
		scalar_t sum = vector_reduce_sum(&s);
		for (size_t j = vector_batches(len); j < len; j++) {
			scalar_t tmp = row.data[j] - row_mean;
			sum += tmp * tmp;
		}

		scalar_t var = sum / len;
		scalar_t var_sqrt = sqrtf(var + 1e-5);

		vector_t vsqrt;
		vector_set(&vsqrt, var_sqrt);

		for (size_t j = 0; j < vector_batches(len); j += VECTOR_BATCH) {
			vector_t tmp;

			vector_load(&tmp, &row.data[j]);
			vector_sub(&tmp, &tmp, &e);
			vector_div(&tmp, &tmp, &vsqrt);
			vector_store(&row.data[j], &tmp);
		}
		for (size_t j = vector_batches(len); j < len; j++) {
			row.data[j] = (row.data[j] - row_mean) / var_sqrt;
		}

		tensor_t ln_row;
		tensor_at(ln, i, &ln_row);

		tensor_mul(&ln_row, &row, weight);
		tensor_add(&ln_row, &ln_row, bias);
	}
}

#define GELU_K1 0.7978845608028654 /* (sqrt(2.0 / M_PI)) */
#define GELU_K2 0.044715

void gelua(tensor_t *t)
{
	assert(t->totlen % VECTOR_BATCH == 0);

	vector_t vinp;
	vector_t va;

	vector_t k1;
	vector_set(&k1, GELU_K1);

	vector_t k2;
	vector_set(&k2, GELU_K2);

	vector_t one;
	vector_set(&one, 1.0);

	vector_t half;
	vector_set(&half, 0.5);

	for (size_t i = 0; i < vector_batches(t->totlen); i += VECTOR_BATCH) {
		vector_load(&vinp, &t->data[i]);

		/* 1.0 + GELU_K2 * inp * inp */
		vector_mul(&va, &vinp, &vinp);
		vector_mul(&va, &va, &k2);
		vector_add(&va, &va, &one);

		/* tanh() */
		vector_mul(&va, &va, &vinp);
		vector_mul(&va, &va, &k1);
		vector_tanh(&va, &va);

		/* 1.0 + tanh() */
		vector_add(&va, &va, &one);

		/* 0.5 * (1.0 + tanh()) */
		vector_mul(&va, &va, &half);

		/* inp * 0.5 * (1.0 * tanh()) */
		vector_mul(&va, &va, &vinp);

		vector_store(&t->data[i], &va);
	}

	for (size_t i = vector_batches(t->totlen); i < t->totlen; i++) {
		scalar_t inp;

		inp = t->data[i];
		t->data[i] = 0.5 * inp * (1.0 + tanhf(GELU_K1 * inp * (1.0 + GELU_K2 * inp * inp)));
	}
}

void silu(tensor_t *t)
{
	for (size_t i = 0; i < vector_batches(t->totlen); i += VECTOR_BATCH) {
		vector_t vinp, vneg, vexp, vone, vdenom;

		vector_load(&vinp, &t->data[i]);

		/* sigmoid(x) = 1 / (1 + exp(-x)) */
		vector_set(&vone, 1.0);
		vector_set(&vneg, 0);
		vector_sub(&vneg, &vneg, &vinp); /* -x */
		vector_exp(&vexp, &vneg);        /* exp(-x) */
		vector_add(&vdenom, &vone, &vexp); /* 1 + exp(-x) */
		vector_div(&vdenom, &vinp, &vdenom); /* x / (1 + exp(-x)) */

		vector_store(&t->data[i], &vdenom);
	}

	for (size_t i = vector_batches(t->totlen); i < t->totlen; i++) {
		scalar_t x = t->data[i];
		t->data[i] = x / (1.0f + expf(-x));
	}
}

void rope_apply(tensor_t *t, int pos, size_t head_len, float theta)
{
	/* t is (H, HLEN), apply RoPE to each head */
	assert(t->ndim == 2);
	size_t H = t->dim[0];
	size_t HLEN = t->dim[1];
	assert(HLEN == head_len);

	size_t half = HLEN / 2;

	for (size_t h = 0; h < H; h++) {
		scalar_t *head = &t->data[h * HLEN];

		for (size_t i = 0; i < half; i++) {
			float freq = 1.0f / powf(theta, (float)(2 * i) / (float)HLEN);
			float angle = (float)pos * freq;
			float cos_a = cosf(angle);
			float sin_a = sinf(angle);

			scalar_t x0 = head[i];
			scalar_t x1 = head[i + half];

			head[i]        = x0 * cos_a - x1 * sin_a;
			head[i + half] = x0 * sin_a + x1 * cos_a;
		}
	}
}

void softmax_1d(tensor_t *t)
{
	size_t len = tensor_len(t);
	vector_t vsum, vmax;
	scalar_t max;

	assert(t->ndim == 1);

	/* https://discuss.pytorch.org/t/how-to-implement-the-exactly-same-softmax-as-f-softmax-by-pytorch/44263/2 */

	max = tensor_max(t, NULL);

	vector_set(&vsum, 0);
	vector_set(&vmax, max);

	for (size_t i = 0; i < vector_batches(len); i += VECTOR_BATCH) {
		vector_t vtmp;

		vector_load(&vtmp, &t->data[i]);
		vector_sub(&vtmp, &vtmp, &vmax);
		vector_exp(&vtmp, &vtmp);
		vector_store(&t->data[i], &vtmp);
		vector_add(&vsum, &vsum, &vtmp);
	}
	scalar_t sum = vector_reduce_sum(&vsum);
	for (size_t i = vector_batches(len); i < len; i++) {
		t->data[i] = expf(t->data[i] - max);
		sum += t->data[i];
	}

	vector_set(&vsum, sum);
	for (size_t i = 0; i < vector_batches(len); i += VECTOR_BATCH) {
		vector_t tmp;

		vector_load(&tmp, &t->data[i]);
		vector_div(&tmp, &tmp, &vsum);
		vector_store(&t->data[i], &tmp);
	}
	for (size_t i = vector_batches(len); i < len; i++) {
		t->data[i] = t->data[i] / sum;
	}
}

void softmax_2d(tensor_t *t)
{
	assert(t->ndim == 2);

	for (size_t i = 0; i < tensor_len(t); i++) {
		tensor_t row;

		tensor_at(t, i, &row);
		softmax_1d(&row);
	}
}

void top_k(tensor_t *f, size_t *top_n, scalar_t *top_v, size_t k)
{
	assert(k <= f->totlen);

	for (size_t i = 0; i < k; i++) {
		top_n[i] = 0;
		top_v[i] = f->data[0];
	}

	for (size_t i = 1; i < f->totlen; i++) {
		scalar_t new_v = f->data[i];
		int new_p = -1;

		for (size_t j = 0; j < k; j++) {
			if (new_v > top_v[j])
				new_p = j;
		}

		if (new_p < 0)
			continue;

		for (size_t j = 0; j < k; j++) {
			if (j < new_p) {
				top_n[j] = top_n[j+1];
				top_v[j] = top_v[j+1];
			} else if (j == new_p) {
				top_n[j] = i;
				top_v[j] = new_v;
				break;
			}
		}
	}
}

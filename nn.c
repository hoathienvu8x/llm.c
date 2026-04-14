#include "nn.h"
#include "vector.h"
#include "quant.h"
#include "thread_pool.h"

#include <math.h>

void rms_norm(tensor_t *ln, tensor_t *tmp_mat, const tensor_t *weight)
{
	for (size_t i = 0; i < tensor_len(tmp_mat); i++) {
		tensor_t row;

		tensor_at(tmp_mat, i, &row);

		vector_t s;
		vector_set(&s, 0);

		size_t len = tensor_len(&row);

		for_each_vec(j, len) {
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

		for_each_vec(j, len) {
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

		for_each_vec(j, len) {
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

		for_each_vec(j, len) {
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

	for_each_vec(i, t->totlen) {
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
	for_each_vec(i, t->totlen) {
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

			/* Standard RoPE: rotate consecutive pairs (2i, 2i+1) */
			scalar_t x0 = head[2*i];
			scalar_t x1 = head[2*i + 1];

			head[2*i]     = x0 * cos_a - x1 * sin_a;
			head[2*i + 1] = x0 * sin_a + x1 * cos_a;
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

	for_each_vec(i, len) {
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
	for_each_vec(i, len) {
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

struct fused_seg {
	scalar_t *out;
	const tensor_t *weight;
	size_t n;
	size_t offset;
};

#ifdef FUSED_GEMV
#define FUSED_MAX_SEGS 4

struct fused_gemv_work {
	const scalar_t *input;
#ifdef Q8_DOT
	const block_q8_K *q8;
#endif
	size_t k;
	size_t total_n;
	int nsegs;
	struct fused_seg segs[FUSED_MAX_SEGS];
};

static void fused_gemv_fn(void *arg, int tidx, int nthreads)
{
	struct fused_gemv_work *w = arg;
	size_t chunk = (w->total_n + nthreads - 1) / nthreads;
	size_t j_start = tidx * chunk;
	size_t j_end = j_start + chunk;
	if (j_end > w->total_n)
		j_end = w->total_n;

	for (size_t j = j_start; j < j_end; j++) {
		for (int s = 0; s < w->nsegs; s++) {
			if (j < w->segs[s].offset + w->segs[s].n) {
				size_t local = j - w->segs[s].offset;
#ifdef Q8_DOT
				w->segs[s].out[local] = dot_q8_quant(
					w->q8, w->input,
					w->segs[s].weight->qdata,
					w->segs[s].weight->type, local, w->k);
#else
				w->segs[s].out[local] = dot_f32_quant(
					w->input,
					w->segs[s].weight->qdata,
					w->segs[s].weight->type, local, w->k);
#endif
				break;
			}
		}
	}
}

static int fused_gemv_run(const tensor_t *in, struct fused_seg *segs, int nsegs)
{
	if (in->dim[0] != 1)
		return 0;

	for (int i = 0; i < nsegs; i++)
		if (segs[i].weight->type == TENSOR_F32 ||
		    segs[i].weight->dim[1] != in->dim[1])
			return 0;

	size_t k = in->dim[1];

	struct fused_gemv_work w = {
		.input = in->data,
		.k = k,
		.nsegs = nsegs,
	};

#ifdef Q8_DOT
	block_q8_K *q8 = NULL;
	if (k % QK_K == 0) {
		q8 = tensor_scratch(in, (k / QK_K) * sizeof(block_q8_K));
		quantize_row_q8(in->data, q8, k);
	}
	w.q8 = q8;
#endif

	size_t total = 0;
	for (int i = 0; i < nsegs; i++) {
		w.segs[i] = segs[i];
		w.segs[i].offset = total;
		total += segs[i].n;
	}
	w.total_n = total;

	thread_pool_run(fused_gemv_fn, &w, 1, k, total);
	return 1;
}
#else
static int fused_gemv_run(const tensor_t *in, struct fused_seg *segs, int nsegs)
{
	return 0;
}
#endif

int fused_gemv2(
	const tensor_t *in,
	tensor_t *out1, const tensor_t *w1,
	tensor_t *out2, const tensor_t *w2)
{
	struct fused_seg segs[] = {
		{ out1->data, w1, w1->dim[0] },
		{ out2->data, w2, w2->dim[0] },
	};
	return fused_gemv_run(in, segs, 2);
}

int fused_gemv3(
	const tensor_t *in,
	tensor_t *out1, const tensor_t *w1,
	tensor_t *out2, const tensor_t *w2,
	tensor_t *out3, const tensor_t *w3)
{
	struct fused_seg segs[] = {
		{ out1->data, w1, w1->dim[0] },
		{ out2->data, w2, w2->dim[0] },
		{ out3->data, w3, w3->dim[0] },
	};
	return fused_gemv_run(in, segs, 3);
}

#ifdef FUSED_GEMV
struct fused_ffn_work {
	const scalar_t *input;
#ifdef Q8_DOT
	const block_q8_K *q8;
#endif
	scalar_t *gate;
	scalar_t *up;
	scalar_t *out;
	const tensor_t *gate_w;
	const tensor_t *up_w;
	const tensor_t *down_w;
	size_t k;          /* input dim (e.g. 4096) */
	size_t intermediate; /* gate/up dim (e.g. 14336) */
	size_t out_n;      /* output dim (e.g. 4096) */
};

static void fused_ffn_fn(void *arg, int tidx, int nthreads)
{
	struct fused_ffn_work *w = arg;
	size_t inter = w->intermediate;

	/* Phase 1: gate+up GEMV — each thread computes its share of rows */
	size_t chunk = (inter + nthreads - 1) / nthreads;
	size_t j_start = tidx * chunk;
	size_t j_end = j_start + chunk;
	if (j_end > inter) j_end = inter;

	for (size_t j = j_start; j < j_end; j++) {
#ifdef Q8_DOT
		w->gate[j] = dot_q8_quant(w->q8, w->input,
			w->gate_w->qdata, w->gate_w->type, j, w->k);
		w->up[j] = dot_q8_quant(w->q8, w->input,
			w->up_w->qdata, w->up_w->type, j, w->k);
#else
		w->gate[j] = dot_f32_quant(w->input,
			w->gate_w->qdata, w->gate_w->type, j, w->k);
		w->up[j] = dot_f32_quant(w->input,
			w->up_w->qdata, w->up_w->type, j, w->k);
#endif
	}

	/* Phase 2: silu(gate) * up — each thread does its portion */
	for (size_t j = j_start; j < j_end; j++) {
		float x = w->gate[j];
		w->gate[j] = (x / (1.0f + expf(-x))) * w->up[j];
	}

	/* Barrier: all threads need the full intermediate for down proj */
	thread_pool_barrier(nthreads);

	/* Phase 3: down projection — each thread computes its share of output rows.
	 * Uses float dot since the intermediate isn't Q8_K-quantized. */
	chunk = (w->out_n + nthreads - 1) / nthreads;
	j_start = tidx * chunk;
	j_end = j_start + chunk;
	if (j_end > w->out_n) j_end = w->out_n;

	for (size_t j = j_start; j < j_end; j++)
		w->out[j] = dot_f32_quant(w->gate,
			w->down_w->qdata, w->down_w->type, j, inter);
}
#endif /* FUSED_GEMV */

int fused_ffn_silu(
	const tensor_t *in, tensor_t *out,
	tensor_t *gate, const tensor_t *gate_w,
	tensor_t *up, const tensor_t *up_w,
	const tensor_t *down_w)
{
#ifdef FUSED_GEMV
	if (in->dim[0] != 1)
		return 0;
	if (gate_w->type == TENSOR_F32 || up_w->type == TENSOR_F32 ||
	    down_w->type == TENSOR_F32)
		return 0;

	size_t k = in->dim[1];
	size_t inter = gate_w->dim[0];

	struct fused_ffn_work w = {
		.input = in->data,
		.gate = gate->data,
		.up = up->data,
		.out = out->data,
		.gate_w = gate_w,
		.up_w = up_w,
		.down_w = down_w,
		.k = k,
		.intermediate = inter,
		.out_n = down_w->dim[0],
	};

#ifdef Q8_DOT
	block_q8_K *q8 = NULL;
	if (k % QK_K == 0) {
		q8 = tensor_scratch(in, (k / QK_K) * sizeof(block_q8_K));
		quantize_row_q8(in->data, q8, k);
	}
	w.q8 = q8;
#endif

	thread_pool_run(fused_ffn_fn, &w, 1, k, inter);
	return 1;
#else
	return 0;
#endif
}

#ifdef FLASH_ATTENTION
#define FA_BLOCK 64

void flash_attention(tensor_t *out, const tensor_t *q, const tensor_t *k,
		     const tensor_t *v, scalar_t scale,
		     size_t cache_size, size_t swa)
{
	size_t T = q->dim[0];
	size_t AT = k->dim[0];
	size_t D = q->dim[1];

	/* Copy Q if aliased with output */
	scalar_t qbuf[T * D];
	const scalar_t *qdata = q->data;
	if (tensor_aliases(out, q)) {
		memcpy(qbuf, q->data, T * D * sizeof(scalar_t));
		qdata = qbuf;
	}

	/* Per-row running state for online softmax */
	scalar_t m[T], l[T];
	for (size_t i = 0; i < T; i++) {
		m[i] = -INFINITY;
		l[i] = 0;
	}
	memset(out->data, 0, T * D * sizeof(scalar_t));

	scalar_t s[T * FA_BLOCK];

	for (size_t j0 = 0; j0 < AT; j0 += FA_BLOCK) {
		size_t Bk = (j0 + FA_BLOCK <= AT) ? FA_BLOCK : AT - j0;

		/* S = Q[T,D] @ K[j0:j0+Bk, D]^T → [T, Bk], scaled */
		for (size_t i = 0; i < T; i++)
			for (size_t j = 0; j < Bk; j++) {
				const scalar_t *qi = &qdata[i * D];
				const scalar_t *kj = &k->data[(j0 + j) * D];
				vector_t acc;
				vector_set(&acc, 0);
				for_each_vec(d, D) {
					vector_t vq, vk;
					vector_load(&vq, (scalar_t *)&qi[d]);
					vector_load(&vk, (scalar_t *)&kj[d]);
					vector_fma(&acc, &vq, &vk, &acc);
				}
				s[i * Bk + j] = vector_reduce_sum(&acc) * scale;
			}

		/* Causal + SWA mask */
		for (size_t i = 0; i < T; i++)
			for (size_t j = 0; j < Bk; j++) {
				size_t kpos = j0 + j;
				if (kpos > cache_size + i)
					s[i * Bk + j] = -INFINITY;
				if (swa && (int)(cache_size + i) - (int)kpos >= (int)swa)
					s[i * Bk + j] = -INFINITY;
			}

		/* Online softmax + accumulate output */
		for (size_t i = 0; i < T; i++) {
			scalar_t *oi = &out->data[i * D];

			/* Block max */
			scalar_t mi = -INFINITY;
			for (size_t j = 0; j < Bk; j++)
				if (s[i * Bk + j] > mi)
					mi = s[i * Bk + j];

			scalar_t m_new = m[i] > mi ? m[i] : mi;
			scalar_t correction = expf(m[i] - m_new);

			/* Rescale previous accumulator */
			vector_t vcorr;
			vector_set(&vcorr, correction);
			for_each_vec(d, D) {
				vector_t vo;
				vector_load(&vo, &oi[d]);
				vector_mul(&vo, &vo, &vcorr);
				vector_store(&oi[d], &vo);
			}
			l[i] *= correction;

			/* Accumulate this block */
			for (size_t j = 0; j < Bk; j++) {
				scalar_t p = expf(s[i * Bk + j] - m_new);
				l[i] += p;
				const scalar_t *vj = &v->data[(j0 + j) * D];
				vector_t vp;
				vector_set(&vp, p);
				for_each_vec(d, D) {
					vector_t vo, vv;
					vector_load(&vo, &oi[d]);
					vector_load(&vv, (scalar_t *)&vj[d]);
					vector_fma(&vo, &vp, &vv, &vo);
					vector_store(&oi[d], &vo);
				}
			}

			m[i] = m_new;
		}
	}

	/* Final normalize */
	for (size_t i = 0; i < T; i++) {
		scalar_t *oi = &out->data[i * D];
		vector_t vinv;
		vector_set(&vinv, 1.0f / l[i]);
		for_each_vec(d, D) {
			vector_t vo;
			vector_load(&vo, &oi[d]);
			vector_mul(&vo, &vo, &vinv);
			vector_store(&oi[d], &vo);
		}
	}
}
#endif

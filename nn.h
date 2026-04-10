#pragma once

#include "tensor.h"

void layer_norm(
	tensor_t *ln,
	tensor_t *tmp_mat,
	const tensor_t *weight,
	const tensor_t *bias);

void softmax_1d(tensor_t *t);
void softmax_2d(tensor_t *t);
void gelua(tensor_t *t);
void silu(tensor_t *t);
void rms_norm(tensor_t *ln, tensor_t *tmp_mat, const tensor_t *weight);
void rope_apply(tensor_t *t, int pos, size_t head_len, float theta);
void top_k(tensor_t *f, size_t *top_n, scalar_t *top_v, size_t k);

void flash_attention(
	tensor_t *out,          /* [T, HLEN] result */
	const tensor_t *q,      /* [T, HLEN] query (one head) */
	const tensor_t *k,      /* [AT, HLEN] cached keys */
	const tensor_t *v,      /* [AT, HLEN] cached values */
	scalar_t scale,         /* 1/sqrt(HLEN) */
	size_t cache_size,      /* positions already in cache (for causal mask) */
	size_t swa);            /* sliding window, 0=disabled */

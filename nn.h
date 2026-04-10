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

/* Fused multi-GEMV: multiple output = input @ weight^T in one thread
 * dispatch. Returns 1 on success, 0 if not applicable (wrong dims,
 * F32 weights, or FUSED_GEMV disabled). */
int fused_gemv2(
	const tensor_t *in,
	tensor_t *out1, const tensor_t *w1,
	tensor_t *out2, const tensor_t *w2);

int fused_gemv3(
	const tensor_t *in,
	tensor_t *out1, const tensor_t *w1,
	tensor_t *out2, const tensor_t *w2,
	tensor_t *out3, const tensor_t *w3);

/*
 * Flash Attention (https://github.com/dao-ailab/flash-attention)
 *
 * Standard attention materializes the full [T, AT] score matrix:
 *   S = Q @ K^T / sqrt(d)
 *   S = mask(S)
 *   P = softmax(S)          ← [T, AT], the bottleneck
 *   O = P @ V
 *
 * Flash attention avoids this by processing K/V in blocks and
 * computing softmax incrementally ("online softmax"):
 *
 *   for each K_block, V_block of size [Bk, D]:
 *       S = Q @ K_block^T / sqrt(d)        # small [T, Bk] tile
 *       S = mask(S)
 *       m_new = max(m, rowmax(S))          # update running max
 *       correction = exp(m_old - m_new)    # rescale previous results
 *       P = exp(S - m_new)                 # local softmax numerators
 *       O = O * correction + P @ V_block   # accumulate weighted values
 *       l = l * correction + rowsum(P)     # accumulate denominator
 *   O = O / l                              # final normalize
 *
 * Memory: O(T * Bk) instead of O(T * AT). For a 4K context with
 * Bk=64, this is 256KB instead of 64MB per head.
 */
void flash_attention(
	tensor_t *out,          /* [T, HLEN] result */
	const tensor_t *q,      /* [T, HLEN] query (one head) */
	const tensor_t *k,      /* [AT, HLEN] cached keys */
	const tensor_t *v,      /* [AT, HLEN] cached values */
	scalar_t scale,         /* 1/sqrt(HLEN) */
	size_t cache_size,      /* positions already in cache (for causal mask) */
	size_t swa);            /* sliding window, 0=disabled */

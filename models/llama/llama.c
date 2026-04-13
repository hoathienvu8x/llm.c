#include "llama.h"
#include "nn.h"
#include "matmul.h"
#include "gguf.h"
#include "vocab.h"
#include "kvcache.h"
#include "vector.h"
#include "tensor_trace.h"
#include "tools.h"
#include "json.h"
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

struct llama {
	struct gguf *gguf;

	struct {
		size_t context;
		size_t q_heads;       /* query heads (32) */
		size_t kv_heads;      /* key/value heads (8) */
		size_t head_len;      /* head dimension (128) */
		size_t layers;
		size_t embeddings;
		size_t vocab_len;
		size_t num_experts;
		size_t top_k_experts;
		size_t expert_intermediate;
	};

	int bos_id;
	int eos_id;
	int pos;
	scalar_t hlen_sq;
	float rope_theta;
	size_t sliding_window; /* 0 = full attention */
	int tool_calls_token;  /* -1 = no tool support */

	const tensor_t *wte;      /* token_embd.weight */
	const tensor_t *lm_head;  /* output.weight */

	struct llama_layer {
		const tensor_t *attn_norm;  /* RMSNorm */

		const tensor_t *q_weight;   /* [q_heads * HLEN, E] */
		const tensor_t *k_weight;   /* [kv_heads * HLEN, E] */
		const tensor_t *v_weight;   /* [kv_heads * HLEN, E] */
		const tensor_t *o_weight;   /* [E, q_heads * HLEN] */

		const tensor_t *ffn_norm;   /* pre-MoE RMSNorm */
		const tensor_t *gate_inp;   /* router: [num_experts, E] */
		const tensor_t **gate_exp;  /* per-expert: [EI, E] each */
		const tensor_t **up_exp;    /* per-expert: [EI, E] each */
		const tensor_t **down_exp;  /* per-expert: [E, EI] each */
	} *hl;

	const tensor_t *output_norm; /* final RMSNorm */

	struct {
		tensor_t *output;
		tensor_t *hidden;
		tensor_t *q, *k, *v;
		tensor_t *qh;
		tensor_t *masked_attn;
		tensor_t *attn;
		tensor_t *attn_residual;
		tensor_t *gate_logits;
		tensor_t *expert_gate;
		tensor_t *expert_up;
		tensor_t *expert_out;
		tensor_t *moe_out;
		tensor_t *logits;
	} state;

	struct kvcache *cache;
};

void *llama_load(struct gguf *g)
{
	struct llama *model;

	model = calloc(1, sizeof(*model));
	if (!model)
		return NULL;

	model->gguf = g;

	model->context = gguf_get_uint32(g, "llama.context_length");
	if (model->context > 8192)
		model->context = 8192;
	model->embeddings = gguf_get_uint32(g, "llama.embedding_length");
	model->q_heads = gguf_get_uint32(g, "llama.attention.head_count");
	model->kv_heads = gguf_get_uint32(g, "llama.attention.head_count_kv");
	model->layers = gguf_get_uint32(g, "llama.block_count");
	model->head_len = model->embeddings / model->q_heads;
	model->vocab_len = gguf_get_arr_n(g, "tokenizer.ggml.tokens");
	model->num_experts = gguf_get_uint32_or(g, "llama.expert_count", 0);
	model->top_k_experts = gguf_get_uint32_or(g, "llama.expert_used_count", 0);
	model->expert_intermediate = gguf_get_uint32(g, "llama.feed_forward_length");
	model->rope_theta = gguf_get_float32(g, "llama.rope.freq_base");
	model->sliding_window = gguf_get_uint32_or(g, "llama.attention.sliding_window", 0);
	model->bos_id = gguf_get_uint32(g, "tokenizer.ggml.bos_token_id");
	model->eos_id = gguf_get_uint32(g, "tokenizer.ggml.eos_token_id");
	model->tool_calls_token = vocab_decode(g, "[TOOL_CALLS]", NULL);

	size_t C = model->context;
	size_t HLEN = model->head_len;
	size_t QH = model->q_heads;
	size_t KVH = model->kv_heads;
	size_t E = model->embeddings;
	size_t NE = model->num_experts;
	size_t EI = model->expert_intermediate;

	model->hlen_sq = sqrt((scalar_t)HLEN);

	assert(QH * HLEN == E);
	assert(QH % KVH == 0);

	if (NE > 0)
		fprintf(stderr, "llama: C=%zu E=%zu QH=%zu KVH=%zu HLEN=%zu L=%zu V=%zu NE=%zu topk=%zu EI=%zu theta=%.1f SWA=%zu\n",
		        C, E, QH, KVH, HLEN, model->layers, model->vocab_len,
		        NE, model->top_k_experts, EI, model->rope_theta, model->sliding_window);
	else
		fprintf(stderr, "llama: C=%zu E=%zu QH=%zu KVH=%zu HLEN=%zu L=%zu V=%zu EI=%zu theta=%.1f SWA=%zu\n",
		        C, E, QH, KVH, HLEN, model->layers, model->vocab_len,
		        EI, model->rope_theta, model->sliding_window);

	model->hl = calloc(model->layers, sizeof(*model->hl));
	assert(model->hl);

	model->wte = gguf_tensor_2d(g, model->vocab_len, E, "token_embd.weight");
	model->lm_head = gguf_tensor_2d(g, model->vocab_len, E, "output.weight");
	if (!model->lm_head) {
		fprintf(stderr, "llama: using tied embeddings (token_embd as lm_head)\n");
		model->lm_head = model->wte;
	}
	model->output_norm = gguf_tensor_1d(g, E, "output_norm.weight");

	for (size_t i = 0; i < model->layers; i++) {
		model->hl[i].attn_norm = gguf_tensor_1d(g, E, "blk.%zu.attn_norm.weight", i);

		model->hl[i].q_weight = gguf_tensor_2d(g, QH * HLEN, E, "blk.%zu.attn_q.weight", i);
		model->hl[i].k_weight = gguf_tensor_2d(g, KVH * HLEN, E, "blk.%zu.attn_k.weight", i);
		model->hl[i].v_weight = gguf_tensor_2d(g, KVH * HLEN, E, "blk.%zu.attn_v.weight", i);
		model->hl[i].o_weight = gguf_tensor_2d(g, E, QH * HLEN, "blk.%zu.attn_output.weight", i);

		model->hl[i].ffn_norm = gguf_tensor_1d(g, E, "blk.%zu.ffn_norm.weight", i);
		if (NE > 0) {
			model->hl[i].gate_inp = gguf_tensor_2d(g, NE, E, "blk.%zu.ffn_gate_inp.weight", i);
			model->hl[i].gate_exp = calloc(NE, sizeof(tensor_t *));
			model->hl[i].up_exp = calloc(NE, sizeof(tensor_t *));
			model->hl[i].down_exp = calloc(NE, sizeof(tensor_t *));
			for (size_t e = 0; e < NE; e++) {
				model->hl[i].gate_exp[e] = gguf_tensor_2d(g, EI, E, "blk.%zu.ffn_gate.%zu.weight", i, e);
				model->hl[i].up_exp[e] = gguf_tensor_2d(g, EI, E, "blk.%zu.ffn_up.%zu.weight", i, e);
				model->hl[i].down_exp[e] = gguf_tensor_2d(g, E, EI, "blk.%zu.ffn_down.%zu.weight", i, e);
			}
		} else {
			model->hl[i].gate_exp = calloc(1, sizeof(tensor_t *));
			model->hl[i].up_exp = calloc(1, sizeof(tensor_t *));
			model->hl[i].down_exp = calloc(1, sizeof(tensor_t *));
			model->hl[i].gate_exp[0] = gguf_tensor_2d(g, EI, E, "blk.%zu.ffn_gate.weight", i);
			model->hl[i].up_exp[0] = gguf_tensor_2d(g, EI, E, "blk.%zu.ffn_up.weight", i);
			model->hl[i].down_exp[0] = gguf_tensor_2d(g, E, EI, "blk.%zu.ffn_down.weight", i);
		}
	}

	/* KV cache uses kv_heads, not q_heads */
	model->state.output = tensor_new_zero(2, C, E);
	model->state.hidden = tensor_new_zero(2, C, E);
	model->state.q = tensor_new_zero(2, C, QH * HLEN);
	model->state.k = tensor_new_zero(2, C, KVH * HLEN);
	model->state.v = tensor_new_zero(2, C, KVH * HLEN);
	model->state.qh = tensor_new_zero(2, C, HLEN);
	model->state.masked_attn = tensor_new_zero(2, C, C);
	model->state.attn = tensor_new_zero(2, C, E);
	model->state.attn_residual = tensor_new_zero(2, C, E);
	if (NE > 0) {
		model->state.gate_logits = tensor_new_zero(2, C, NE);
		model->state.expert_gate = tensor_new_zero(2, 1, EI);
		model->state.expert_up = tensor_new_zero(2, 1, EI);
		model->state.expert_out = tensor_new_zero(2, 1, E);
	} else {
		model->state.expert_gate = tensor_new_zero(2, C, EI);
		model->state.expert_up = tensor_new_zero(2, C, EI);
		model->state.expert_out = tensor_new_zero(2, C, E);
	}
	model->state.moe_out = tensor_new_zero(2, C, E);
	model->state.logits = tensor_new_zero(1, model->vocab_len);

	model->cache = kvcache_alloc(model->layers, C, KVH, HLEN);
	assert(model->cache);
	model->cache->sliding_window = model->sliding_window;

	uint64_t totmem = 0;
	totmem += model->state.output->maxcap;
	totmem += model->state.hidden->maxcap;
	totmem += model->state.q->maxcap;
	totmem += model->state.k->maxcap;
	totmem += model->state.v->maxcap;
	totmem += model->state.qh->maxcap;
	totmem += model->state.masked_attn->maxcap;
	totmem += model->state.attn->maxcap;
	totmem += model->state.attn_residual->maxcap;
	if (model->state.gate_logits)
		totmem += model->state.gate_logits->maxcap;
	totmem += model->state.expert_gate->maxcap;
	totmem += model->state.expert_up->maxcap;
	totmem += model->state.expert_out->maxcap;
	totmem += model->state.moe_out->maxcap;
	totmem += model->state.logits->maxcap;

	uint64_t cachemem = 0;
	for (size_t i = 0; i < model->layers; i++) {
		cachemem += model->cache->hl[i].k->maxcap;
		cachemem += model->cache->hl[i].v->maxcap;
	}

	fprintf(stderr, "runtime memory: %luMB + %luMB KV cache\n",
	        totmem / 1024 / 1024,
	        cachemem / 1024 / 1024);

	return model;
}

/* GQA attention: q_heads > kv_heads.
 * Each KV head serves (q_heads / kv_heads) Q heads.
 * We iterate over KV heads and process the Q head group for each. */
static void transformer(struct llama *model, tensor_t *q, tensor_t *k, tensor_t *v,
                         tensor_t *output, size_t l, int *pos, enum kv_mode mode)
{
	tensor_t *qh = model->state.qh;
	tensor_t *masked_attn = model->state.masked_attn;

	struct kvcache *cache = model->cache;

	bool decode = mode == KV_DECODE;
	bool prefill = !decode;

	size_t T = tensor_len(q);
	size_t AT = cache->size + T;  /* total context length after this step */
	size_t QH = model->q_heads;
	size_t KVH = model->kv_heads;
	size_t HLEN = model->head_len;
	size_t group_size = QH / KVH;

	tensor_resize(qh, T);
	tensor_resize_2d(masked_attn, AT, AT);

	for (size_t kv_idx = 0; kv_idx < KVH; kv_idx++) {
		tensor_t cache_k, cache_v;

		kvcache_get_k(cache, l, kv_idx, &cache_k);
		kvcache_get_v(cache, l, kv_idx, &cache_v);

		/* Write K and V into cache (once per KV head) */
		if (decode) {
			tensor_t kv;
			tensor_at(k, 0, &kv);
			tensor_reshape_2d(&kv, KVH, HLEN);
			tensor_at(&kv, kv_idx, &kv);
			tensor_set_inner(&cache_k, cache->size, &kv);

			tensor_t vv;
			tensor_at(v, 0, &vv);
			tensor_reshape_2d(&vv, KVH, HLEN);
			tensor_at(&vv, kv_idx, &vv);
			tensor_set_inner(&cache_v, cache->size, &vv);
		} else {
			for (size_t t_idx = 0; t_idx < T; t_idx++) {
				tensor_t kv;
				tensor_at(k, t_idx, &kv);
				tensor_reshape_2d(&kv, KVH, HLEN);
				tensor_at(&kv, kv_idx, &kv);
				tensor_set_inner(&cache_k, cache->size + t_idx, &kv);

				tensor_t vv;
				tensor_at(v, t_idx, &vv);
				tensor_reshape_2d(&vv, KVH, HLEN);
				tensor_at(&vv, kv_idx, &vv);
				tensor_set_inner(&cache_v, cache->size + t_idx, &vv);
			}
		}

		tensor_resize(&cache_k, AT);
		tensor_resize(&cache_v, AT);

		/* Process each Q head in this KV group */
		for (size_t g = 0; g < group_size; g++) {
			size_t q_idx = kv_idx * group_size + g;

			/* Extract Q head */
			if (decode) {
				tensor_t qv;
				tensor_at(q, 0, &qv);
				tensor_reshape_2d(&qv, QH, HLEN);
				tensor_at(&qv, q_idx, &qv);
				tensor_set_inner(qh, 0, &qv);
			} else {
				for (size_t t_idx = 0; t_idx < T; t_idx++) {
					tensor_t qv;
					tensor_at(q, t_idx, &qv);
					tensor_reshape_2d(&qv, QH, HLEN);
					tensor_at(&qv, q_idx, &qv);
					tensor_set_inner(qh, t_idx, &qv);
				}
			}

			tensor_assert_2d(qh, T, HLEN);
			tensor_assert_2d(&cache_k, AT, HLEN);
			tensor_assert_2d(&cache_v, AT, HLEN);

#ifdef FLASH_ATTENTION
			flash_attention(qh, qh, &cache_k, &cache_v,
					1.0f / model->hlen_sq, cache->size,
					model->sliding_window);
#else
			tensor_mma_transposed_2x2(masked_attn, qh, &cache_k, NULL);
			tensor_assert_2d(masked_attn, T, AT);
			tensor_div_scalar(masked_attn, masked_attn, model->hlen_sq);

			if (prefill) {
				size_t swa = model->sliding_window;
				for (size_t i = 0; i < T; i++)
					for (size_t j = 0; j < AT; j++) {
						int causal = j > cache->size + i;
						int outside = swa && pos[i] - (int)j >= (int)swa;
						if (causal || outside)
							masked_attn->data[i * AT + j] = -INFINITY;
					}
			}

			softmax_2d(masked_attn);

			tensor_mma_2x2(qh, masked_attn, &cache_v, NULL);
#endif
			tensor_assert_2d(qh, T, HLEN);

			/* Write back to output at the correct Q head position */
			size_t out_start = prefill ? 0 : AT - 1;
			size_t out_end = prefill ? T : AT;
			for (size_t t_idx = out_start; t_idx < out_end; t_idx++) {
				tensor_t row;
				tensor_at(qh, prefill ? t_idx : 0, &row);
				tensor_assert_1d(&row, HLEN);

				tensor_t attn_tok;
				tensor_at(output, prefill ? t_idx : 0, &attn_tok);
				tensor_reshape_2d(&attn_tok, QH, HLEN);
				tensor_set_inner(&attn_tok, q_idx, &row);
			}
		}
	}
}

static void dense_ffn(struct llama *model, tensor_t *input, tensor_t *output, size_t l)
{
	struct llama_layer *hl = &model->hl[l];
	tensor_t *gate = model->state.expert_gate;
	tensor_t *up = model->state.expert_up;

	size_t T = tensor_len(input);
	tensor_resize(gate, T);
	tensor_resize(up, T);

	if (!fused_gemv2(input,
			       gate, hl->gate_exp[0],
			       up, hl->up_exp[0])) {
		tensor_mma_transposed_2x2(gate, input, hl->gate_exp[0], NULL);
		tensor_mma_transposed_2x2(up, input, hl->up_exp[0], NULL);
	}

	silu(gate);
	tensor_mul(gate, gate, up);
	tensor_mma_transposed_2x2(output, gate, hl->down_exp[0], NULL);
}

static void moe_ffn(struct llama *model, tensor_t *input, tensor_t *output, size_t l)
{
	struct llama_layer *hl = &model->hl[l];
	size_t T = tensor_len(input);
	size_t E = model->embeddings;
	size_t EI = model->expert_intermediate;
	size_t NE = model->num_experts;
	size_t topk = model->top_k_experts;

	tensor_t *gate_logits = model->state.gate_logits;
	tensor_t *expert_gate = model->state.expert_gate;
	tensor_t *expert_up = model->state.expert_up;
	tensor_t *expert_out = model->state.expert_out;

	tensor_set(output, 0);

	tensor_resize(gate_logits, T);
	tensor_mma_transposed_2x2(gate_logits, input, hl->gate_inp, NULL);
	tensor_assert_2d(gate_logits, T, NE);

	for (size_t t = 0; t < T; t++) {
		tensor_t tok_logits;
		tensor_at(gate_logits, t, &tok_logits);
		softmax_1d(&tok_logits);

		size_t expert_ids[topk];
		scalar_t expert_weights[topk];
		top_k(&tok_logits, expert_ids, expert_weights, topk);

		/* Normalize expert weights to sum to 1 */
		scalar_t weight_sum = 0;
		for (size_t e = 0; e < topk; e++)
			weight_sum += expert_weights[e];
		for (size_t e = 0; e < topk; e++)
			expert_weights[e] /= weight_sum;

		tensor_t tok_in;
		tensor_at(input, t, &tok_in);
		tensor_reshape_2d(&tok_in, 1, E);

		for (size_t e = 0; e < topk; e++) {
			const tensor_t *gate_w = hl->gate_exp[expert_ids[e]];
			const tensor_t *up_w = hl->up_exp[expert_ids[e]];
			const tensor_t *down_w = hl->down_exp[expert_ids[e]];

			tensor_mma_transposed_2x2(expert_gate, &tok_in, gate_w, NULL);
			tensor_mma_transposed_2x2(expert_up, &tok_in, up_w, NULL);

			silu(expert_gate);
			tensor_mul(expert_gate, expert_gate, expert_up);

			tensor_mma_transposed_2x2(expert_out, expert_gate, down_w, NULL);

			tensor_t tok_out;
			tensor_at(output, t, &tok_out);

			tensor_reshape_1d(expert_out, E);
			tensor_div_scalar(expert_out, expert_out, 1.0f / expert_weights[e]);
			tensor_add(&tok_out, &tok_out, expert_out);
			tensor_reshape_2d(expert_out, 1, E);
		}
	}
}

static void llama_forward(struct llama *model, int *tok, int *pos, size_t T,
                             tensor_t *output, enum kv_mode mode)
{
	size_t E = model->embeddings;
	size_t QH = model->q_heads;
	size_t KVH = model->kv_heads;
	size_t HLEN = model->head_len;

	tensor_t *hidden = model->state.hidden;
	tensor_t *q = model->state.q;
	tensor_t *k = model->state.k;
	tensor_t *v = model->state.v;
	tensor_t *attn = model->state.attn;
	tensor_t *attn_residual = model->state.attn_residual;
	tensor_t *moe_out = model->state.moe_out;

	tensor_pick_rows(hidden, model->wte, tok, T);
	tensor_assert_2d(hidden, T, E);

	tensor_resize(q, T);
	tensor_resize(k, T);
	tensor_resize(v, T);
	tensor_resize(attn, T);
	tensor_resize(attn_residual, T);
	tensor_resize(moe_out, T);
	tensor_resize(output, T);

	tensor_trace(hidden, "embed");

	for (size_t l = 0; l < model->layers; l++) {
		struct llama_layer *hl = &model->hl[l];

		/* Pre-attention RMSNorm */
		tensor_copy(output, hidden);
		rms_norm(q, output, hl->attn_norm);
		tensor_trace(q, "L%zu.norm1", l);

		/* Separate Q/K/V projections */
		tensor_copy(output, q);
		if (!fused_gemv3(output,
				       q, hl->q_weight,
				       k, hl->k_weight,
				       v, hl->v_weight)) {
			tensor_mma_transposed_2x2(q, output, hl->q_weight, NULL);
			tensor_mma_transposed_2x2(k, output, hl->k_weight, NULL);
			tensor_mma_transposed_2x2(v, output, hl->v_weight, NULL);
		}
		tensor_assert_2d(q, T, QH * HLEN);
		tensor_assert_2d(k, T, KVH * HLEN);
		tensor_assert_2d(v, T, KVH * HLEN);
		tensor_trace(q, "L%zu.Q", l);
		tensor_trace(k, "L%zu.K", l);
		tensor_trace(v, "L%zu.V", l);

		/* Apply RoPE to Q and K */
		for (size_t t_idx = 0; t_idx < T; t_idx++) {
			tensor_t qt;
			tensor_at(q, t_idx, &qt);
			tensor_reshape_2d(&qt, QH, HLEN);
			rope_apply(&qt, pos[t_idx], HLEN, model->rope_theta);

			tensor_t kt;
			tensor_at(k, t_idx, &kt);
			tensor_reshape_2d(&kt, KVH, HLEN);
			rope_apply(&kt, pos[t_idx], HLEN, model->rope_theta);
		}
		tensor_trace(NULL, "L%zu.rope", l);

		/* GQA multi-head attention */
		transformer(model, q, k, v, attn, l, pos, mode);
		tensor_trace(attn, "L%zu.attn", l);

		/* Output projection + residual */
		tensor_mma_transposed_2x2(output, attn, hl->o_weight, NULL);
		tensor_trace(output, "L%zu.oproj", l);
		tensor_add(hidden, hidden, output);
		tensor_copy(attn_residual, hidden);

		/* Pre-MoE RMSNorm */
		tensor_copy(output, hidden);
		rms_norm(moe_out, output, hl->ffn_norm);
		tensor_trace(moe_out, "L%zu.norm2", l);

		if (model->num_experts > 0)
			moe_ffn(model, moe_out, output, l);
		else
			dense_ffn(model, moe_out, output, l);
		tensor_trace(output, "L%zu.ffn", l);

		/* Residual */
		tensor_add(hidden, attn_residual, output);
	}

	/* Final RMSNorm */
	tensor_copy(output, hidden);
	rms_norm(hidden, output, model->output_norm);
	tensor_copy(output, hidden);
	tensor_trace(output, "final_norm");

}

void llama_prefill(struct llama *model, int *tok, int *pos, size_t T, tensor_t *output)
{
	llama_forward(model, tok, pos, T, output, KV_PREFILL);
	model->cache->size += T;
}

void llama_decode(struct llama *model, int tok, int pos, tensor_t *output)
{
	if (model->cache->size >= model->cache->context)
		kvcache_rotate(model->cache);

	if (pos >= (int)model->cache->context)
		pos = model->cache->size;

	llama_forward(model, &tok, &pos, 1, output, KV_DECODE);
	model->cache->size++;
}

void llama_generate(void *ctx, const char *text, int num, pick_token_t f, void *cb_ctx)
{
	struct llama *model = ctx;
	int tok_sz;

	size_t E = model->embeddings;
	size_t C = model->context;

	uint64_t total_begin = profiler_now();

	tensor_t *output = model->state.output;
	tensor_t *logits = model->state.logits;

	int *toks = malloc(C * sizeof(int));
	int *poss = malloc(C * sizeof(int));
	assert(toks && poss);

	int T = 0;
	int tok;

	if (model->pos == 0) {
		toks[T] = model->bos_id;
		poss[T] = T;
		T++;
	}

	T += vocab_tokenize(model->gguf, text, &toks[T], C - T);
	for (int i = (model->pos == 0) ? 1 : 0; i < T; i++)
		poss[i] = model->pos + i;

	uint64_t prefill_begin = profiler_now();

	if (getenv("DEBUG_TOKENS")) {
		fprintf(stderr, "TOKENS: pos=%d cache=%zu T=%d\n",
			model->pos, model->cache->size, T);
		for (int i = 0; i < T; i++)
			fprintf(stderr, "  [%d] tok=%d pos=%d '%s'\n",
				i, toks[i], poss[i], vocab_encode(model->gguf, toks[i]));
	}

	llama_prefill(model, toks, poss, T, output);
	model->pos += T;

	uint64_t prefill_end = profiler_now();

	tensor_t last_row;
	tensor_at(output, T - 1, &last_row);
	tensor_assert_1d(logits, model->vocab_len);
	tensor_assert_1d(&last_row, E);
	tensor_reshape_2d(&last_row, 1, E);
	tensor_mma_transposed_2x2(logits, &last_row, model->lm_head, NULL);
	tensor_reshape_1d(logits, model->vocab_len);

	tok = f(cb_ctx, logits);

	uint64_t decode_begin = profiler_now();
	uint64_t batch_begin = decode_begin;
	while (num-- && tok != model->eos_id) {
		llama_decode(model, tok, model->pos, output);
		model->pos++;

		tensor_at(output, 0, &last_row);
		tensor_assert_1d(logits, model->vocab_len);
		tensor_assert_1d(&last_row, E);
		tensor_reshape_2d(&last_row, 1, E);
		tensor_mma_transposed_2x2(logits, &last_row, model->lm_head, NULL);
		tensor_reshape_1d(logits, model->vocab_len);
		tok = f(cb_ctx, logits);

		if (model->pos && model->pos % 100 == 0) {
			uint64_t end = profiler_now();
			jsonw_t *j = trace_begin("decode");
			jsonw_num(j, "tokens", model->pos);
			jsonw_num(j, "tok_per_sec", 100/profiler_to_sec(end-batch_begin));
			trace_end(j);
			batch_begin = end;
		}
	}
	uint64_t decode_end = profiler_now();

	jsonw_t *j = trace_begin("generate");
	jsonw_num(j, "prefill_sec", profiler_to_sec(prefill_end - prefill_begin));
	jsonw_num(j, "prefill_tokens", T);
	jsonw_num(j, "decode_sec", profiler_to_sec(decode_end - decode_begin));
	jsonw_num(j, "total_sec", profiler_to_sec(decode_end - total_begin));
	trace_end(j);

	free(toks);
	free(poss);
}

void llama_close(void *ctx)
{
	struct llama *model = ctx;

	tensor_free_mapped(model->wte);
	if (model->lm_head != model->wte)
		tensor_free_mapped(model->lm_head);
	tensor_free_mapped(model->output_norm);

	for (size_t i = 0; i < model->layers; i++) {
		tensor_free_mapped(model->hl[i].attn_norm);
		tensor_free_mapped(model->hl[i].q_weight);
		tensor_free_mapped(model->hl[i].k_weight);
		tensor_free_mapped(model->hl[i].v_weight);
		tensor_free_mapped(model->hl[i].o_weight);
		tensor_free_mapped(model->hl[i].ffn_norm);
		if (model->hl[i].gate_inp)
			tensor_free_mapped(model->hl[i].gate_inp);
		size_t ne = model->num_experts > 0 ? model->num_experts : 1;
		for (size_t e = 0; e < ne; e++) {
			tensor_free_mapped(model->hl[i].gate_exp[e]);
			tensor_free_mapped(model->hl[i].up_exp[e]);
			tensor_free_mapped(model->hl[i].down_exp[e]);
		}
		free(model->hl[i].gate_exp);
		free(model->hl[i].up_exp);
		free(model->hl[i].down_exp);
	}
	free(model->hl);

	tensor_free(model->state.output);
	tensor_free(model->state.hidden);
	tensor_free(model->state.q);
	tensor_free(model->state.k);
	tensor_free(model->state.v);
	tensor_free(model->state.qh);
	tensor_free(model->state.masked_attn);
	tensor_free(model->state.attn);
	tensor_free(model->state.attn_residual);
	if (model->state.gate_logits)
		tensor_free(model->state.gate_logits);
	tensor_free(model->state.expert_gate);
	tensor_free(model->state.expert_up);
	tensor_free(model->state.expert_out);
	tensor_free(model->state.moe_out);
	tensor_free(model->state.logits);

	kvcache_free(model->cache);

	free(model);
}

/* Mistral tool calling format:
 *   [AVAILABLE_TOOLS] [json] [/AVAILABLE_TOOLS]
 *   [TOOL_CALLS] [json]</s>
 *   [TOOL_RESULTS] result [/TOOL_RESULTS] */

static char *llama_tools_format(void *ctx)
{
	struct llama *model = ctx;
	int n = tools_get_count();
	if (model->tool_calls_token < 0 || n == 0)
		return strdup("");

	jsonw_t *j = jsonw_new();
	/* Raw prefix — not JSON, so write directly */
	fprintf(j->f, "[AVAILABLE_TOOLS] ");

	jsonw_arr(j);
	for (int i = 0; i < n; i++) {
		const struct tool *t = tools_get(i);
		jsonw_obj(j);
		jsonw_str(j, "type", "function");
		jsonw_key(j, "function");
		jsonw_obj(j);
		jsonw_str(j, "name", t->name);
		jsonw_str(j, "description", t->description);
		tools_format_params(j, t);
		jsonw_obj_end(j);
		jsonw_obj_end(j);
	}
	jsonw_arr_end(j);

	fprintf(j->f, " [/AVAILABLE_TOOLS]");
	return jsonw_done(j);
}

static int llama_tools_detect(void *ctx, int tok)
{
	struct llama *model = ctx;
	return model->tool_calls_token >= 0 && tok == model->tool_calls_token;
}

static char *llama_tools_wrap_result(void *ctx, const char *result)
{
	jsonw_t *j = jsonw_new();
	fprintf(j->f, "[TOOL_RESULTS]");
	jsonw_obj(j);
	jsonw_key(j, "content");
	fprintf(j->f, "%s", result);
	j->need_comma = 1;
	jsonw_obj_end(j);
	fprintf(j->f, "[/TOOL_RESULTS]");
	return jsonw_done(j);
}

static const struct model llama_model = {
	.name = "llama",  /* Mixtral uses llama architecture in GGUF */
	.load = llama_load,
	.generate = llama_generate,
	.close = llama_close,
	.tools_format = llama_tools_format,
	.tools_detect = llama_tools_detect,
	.tools_wrap_result = llama_tools_wrap_result,
};

__attribute__((constructor))
static void llama_register(void)
{
	register_model(&llama_model);
}

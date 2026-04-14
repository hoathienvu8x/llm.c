#include "olmoe.h"
#include "nn.h"
#include "matmul.h"
#include "gguf.h"
#include "vocab.h"
#include "kvcache.h"
#include "vector.h"
#include "tensor_trace.h"
#include "json.h"
#include "quant.h"

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* Like tensor_at(), but works for quantized tensors by computing
 * the correct byte offset into qdata. For F32 tensors it falls
 * back to the normal data pointer arithmetic. */
static void tensor_at_q(const tensor_t *t, size_t idx, tensor_t *view)
{
	assert(t->ndim > 1);
	assert(idx < t->dim[0]);

	size_t inner_totlen = t->totlen / t->dim[0];

	view->ndim = t->ndim - 1;
	view->dim[0] = t->dim[1];
	view->dim[1] = t->dim[2];
	view->dim[2] = t->dim[3];
	view->totlen = inner_totlen;
	view->maxcap = inner_totlen;
	view->type = t->type;

	if (t->type == TENSOR_F32) {
		view->data = &t->data[inner_totlen * idx];
		view->qdata = NULL;
	} else {
		view->data = NULL;

		/* Compute byte offset: each row of the innermost dimension
		 * consists of (cols / elements_per_block) quantized blocks. */
		size_t cols = t->dim[t->ndim - 1];
		size_t rows_per_slice = inner_totlen / cols;
		size_t epb, bsz;

		switch (t->type) {
		case TENSOR_Q8_0: epb = GGML_QK; bsz = sizeof(block_q8_0); break;
		case TENSOR_Q4_0: epb = GGML_QK; bsz = sizeof(block_q4_0); break;
		case TENSOR_Q4_K: epb = QK_K;    bsz = sizeof(block_q4_K); break;
		case TENSOR_Q5_K: epb = QK_K;    bsz = sizeof(block_q5_K); break;
		case TENSOR_Q6_K: epb = QK_K;    bsz = sizeof(block_q6_K); break;
		default: abort();
		}

		size_t blocks_per_row = cols / epb;
		size_t byte_offset = idx * rows_per_slice * blocks_per_row * bsz;
		view->qdata = (char *)t->qdata + byte_offset;
	}
}

struct olmoe {
	struct gguf *gguf;

	struct {
		size_t context;
		size_t heads;
		size_t head_len;
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

	const tensor_t *wte;      /* token_embd.weight */
	const tensor_t *lm_head;  /* output.weight (untied) */

	struct olmoe_layer {
		const tensor_t *attn_norm;  /* RMSNorm */

		const tensor_t *q_weight;
		const tensor_t *k_weight;
		const tensor_t *v_weight;
		const tensor_t *q_norm;     /* per-head Q RMSNorm [E] */
		const tensor_t *k_norm;     /* per-head K RMSNorm [E] */
		const tensor_t *o_weight;

		const tensor_t *ffn_norm;   /* pre-MoE RMSNorm */
		const tensor_t *gate_inp;   /* router: [64, E] */
		const tensor_t *gate_exps;  /* [64, intermediate, E] */
		const tensor_t *up_exps;    /* [64, intermediate, E] */
		const tensor_t *down_exps;  /* [64, E, intermediate] */
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

void *olmoe_load(struct gguf *g)
{
	struct olmoe *model;

	model = calloc(1, sizeof(*model));
	if (!model)
		return NULL;

	model->gguf = g;

	model->context = gguf_get_uint32(g, "olmoe.context_length");
	model->embeddings = gguf_get_uint32(g, "olmoe.embedding_length");
	model->heads = gguf_get_uint32(g, "olmoe.attention.head_count");
	model->layers = gguf_get_uint32(g, "olmoe.block_count");
	model->head_len = model->embeddings / model->heads;
	model->vocab_len = gguf_get_arr_n(g, "tokenizer.ggml.tokens");
	model->num_experts = gguf_get_uint32(g, "olmoe.expert_count");
	model->top_k_experts = gguf_get_uint32(g, "olmoe.expert_used_count");
	model->expert_intermediate = gguf_get_uint32(g, "olmoe.feed_forward_length");
	model->rope_theta = gguf_get_float32(g, "olmoe.rope.freq_base");
	model->bos_id = gguf_get_uint32(g, "tokenizer.ggml.bos_token_id");
	model->eos_id = gguf_get_uint32(g, "tokenizer.ggml.eos_token_id");

	size_t C = model->context;
	size_t HLEN = model->head_len;
	size_t H = model->heads;
	size_t E = model->embeddings;
	size_t NE = model->num_experts;
	size_t EI = model->expert_intermediate;

	model->hlen_sq = sqrt((scalar_t)HLEN);

	assert(H * HLEN == E);

	fprintf(stderr, "olmoe: C=%zu E=%zu H=%zu HLEN=%zu L=%zu V=%zu NE=%zu topk=%zu EI=%zu theta=%.1f\n",
	        C, E, H, HLEN, model->layers, model->vocab_len, NE, model->top_k_experts, EI, model->rope_theta);

	model->hl = calloc(model->layers, sizeof(*model->hl));
	assert(model->hl);

	model->wte = gguf_tensor_2d(g, model->vocab_len, E, "token_embd.weight");
	model->lm_head = gguf_tensor_2d(g, model->vocab_len, E, "output.weight");
	model->output_norm = gguf_tensor_1d(g, E, "output_norm.weight");

	for (size_t i = 0; i < model->layers; i++) {
		model->hl[i].attn_norm = gguf_tensor_1d(g, E, "blk.%zu.attn_norm.weight", i);

		model->hl[i].q_weight = gguf_tensor_2d(g, E, E, "blk.%zu.attn_q.weight", i);
		model->hl[i].k_weight = gguf_tensor_2d(g, E, E, "blk.%zu.attn_k.weight", i);
		model->hl[i].v_weight = gguf_tensor_2d(g, E, E, "blk.%zu.attn_v.weight", i);
		model->hl[i].q_norm = gguf_tensor_1d(g, E, "blk.%zu.attn_q_norm.weight", i);
		model->hl[i].k_norm = gguf_tensor_1d(g, E, "blk.%zu.attn_k_norm.weight", i);
		model->hl[i].o_weight = gguf_tensor_2d(g, E, E, "blk.%zu.attn_output.weight", i);

		model->hl[i].ffn_norm = gguf_tensor_1d(g, E, "blk.%zu.ffn_norm.weight", i);
		model->hl[i].gate_inp = gguf_tensor_2d(g, NE, E, "blk.%zu.ffn_gate_inp.weight", i);
		model->hl[i].gate_exps = gguf_tensor_3d(g, NE, EI, E, "blk.%zu.ffn_gate_exps.weight", i);
		model->hl[i].up_exps = gguf_tensor_3d(g, NE, EI, E, "blk.%zu.ffn_up_exps.weight", i);
		model->hl[i].down_exps = gguf_tensor_3d(g, NE, E, EI, "blk.%zu.ffn_down_exps.weight", i);
	}

	model->state.output = tensor_new_zero(2, C, E);
	model->state.hidden = tensor_new_zero(2, C, E);
	model->state.q = tensor_new_zero(2, C, E);
	model->state.k = tensor_new_zero(2, C, E);
	model->state.v = tensor_new_zero(2, C, E);
	model->state.qh = tensor_new_zero(2, C, HLEN);
	model->state.masked_attn = tensor_new_zero(2, C, C);
	model->state.attn = tensor_new_zero(2, C, E);
	model->state.attn_residual = tensor_new_zero(2, C, E);
	model->state.gate_logits = tensor_new_zero(2, C, NE);
	model->state.expert_gate = tensor_new_zero(2, 1, EI);
	model->state.expert_up = tensor_new_zero(2, 1, EI);
	model->state.expert_out = tensor_new_zero(2, 1, E);
	model->state.moe_out = tensor_new_zero(2, C, E);
	model->state.logits = tensor_new_zero(1, model->vocab_len);

	model->cache = kvcache_alloc(model->layers, C, H, HLEN);
	assert(model->cache);

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

static void transformer(struct olmoe *model, tensor_t *q, tensor_t *k, tensor_t *v,
                         tensor_t *output, size_t l, enum kv_mode mode)
{
	tensor_t *qh = model->state.qh;
	tensor_t *masked_attn = model->state.masked_attn;

	struct kvcache *cache = model->cache;

	bool decode = mode == KV_DECODE;
	bool prefill = !decode;

	size_t T = tensor_len(q);
	size_t AT = cache->size + T;
	size_t H = model->heads;
	size_t HLEN = model->head_len;
	size_t E = model->embeddings;

	tensor_resize(qh, T);
	tensor_resize_2d(masked_attn, AT, AT);

	for (size_t h_idx = 0; h_idx < H; h_idx++) {
		tensor_t cache_k, cache_v;

		kvcache_get_k(cache, l, h_idx, &cache_k);
		kvcache_get_v(cache, l, h_idx, &cache_v);

		if (decode) {
			/* Q: extract head from single token */
			tensor_t qv;
			tensor_at(q, 0, &qv);
			tensor_reshape_2d(&qv, H, HLEN);
			tensor_at(&qv, h_idx, &qv);
			tensor_set_inner(qh, 0, &qv);

			/* K: write into cache */
			tensor_t kv;
			tensor_at(k, 0, &kv);
			tensor_reshape_2d(&kv, H, HLEN);
			tensor_at(&kv, h_idx, &kv);
			tensor_set_inner(&cache_k, cache->size, &kv);

			/* V: write into cache */
			tensor_t vv;
			tensor_at(v, 0, &vv);
			tensor_reshape_2d(&vv, H, HLEN);
			tensor_at(&vv, h_idx, &vv);
			tensor_set_inner(&cache_v, cache->size, &vv);
		} else {
			for (size_t t_idx = 0; t_idx < T; t_idx++) {
				tensor_t qv;
				tensor_at(q, t_idx, &qv);
				tensor_reshape_2d(&qv, H, HLEN);
				tensor_at(&qv, h_idx, &qv);
				tensor_set_inner(qh, t_idx, &qv);

				tensor_t kv;
				tensor_at(k, t_idx, &kv);
				tensor_reshape_2d(&kv, H, HLEN);
				tensor_at(&kv, h_idx, &kv);
				tensor_set_inner(&cache_k, cache->size + t_idx, &kv);

				tensor_t vv;
				tensor_at(v, t_idx, &vv);
				tensor_reshape_2d(&vv, H, HLEN);
				tensor_at(&vv, h_idx, &vv);
				tensor_set_inner(&cache_v, cache->size + t_idx, &vv);
			}
		}

		tensor_resize(&cache_k, AT);
		tensor_resize(&cache_v, AT);

		tensor_assert_2d(qh, T, HLEN);
		tensor_assert_2d(&cache_k, AT, HLEN);
		tensor_assert_2d(&cache_v, AT, HLEN);
		tensor_mma_transposed_2x2(masked_attn, qh, &cache_k, NULL);
		tensor_assert_2d(masked_attn, T, AT);
		tensor_div_scalar(masked_attn, masked_attn, model->hlen_sq);

		if (prefill) {
			for (size_t i = 0; i < T; i++)
				for (size_t j = 0; j < AT; j++)
					if (j > cache->size + i)
						masked_attn->data[i * AT + j] = -1.0000e+04;
		}

		softmax_2d(masked_attn);

		tensor_assert_2d(masked_attn, T, AT);
		tensor_assert_2d(&cache_v, AT, HLEN);

		tensor_mma_2x2(qh, masked_attn, &cache_v, NULL);
		tensor_assert_2d(qh, T, HLEN);

		size_t out_start = prefill ? 0 : AT - 1;
		size_t out_end = prefill ? T : AT;
		for (size_t t_idx = out_start; t_idx < out_end; t_idx++) {
			tensor_t row;
			tensor_at(qh, prefill ? t_idx : 0, &row);
			tensor_assert_1d(&row, HLEN);

			tensor_t attn_tok;
			tensor_at(output, prefill ? t_idx : 0, &attn_tok);
			tensor_reshape_2d(&attn_tok, H, HLEN);
			tensor_set_inner(&attn_tok, h_idx, &row);
		}
	}
}

static void moe_ffn(struct olmoe *model, tensor_t *input, tensor_t *output, size_t l)
{
	struct olmoe_layer *hl = &model->hl[l];
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

	/* Compute router logits: input @ gate_inp^T -> (T, NE) */
	tensor_resize(gate_logits, T);
	tensor_mma_transposed_2x2(gate_logits, input, hl->gate_inp, NULL);
	tensor_assert_2d(gate_logits, T, NE);

	for (size_t t = 0; t < T; t++) {
		tensor_t tok_logits;
		tensor_at(gate_logits, t, &tok_logits);
		softmax_1d(&tok_logits);

		/* Select top-k experts */
		size_t expert_ids[topk];
		scalar_t expert_weights[topk];
		top_k(&tok_logits, expert_ids, expert_weights, topk);

		/* Get token input as (1, E) for matmul */
		tensor_t tok_in;
		tensor_at(input, t, &tok_in);
		tensor_reshape_2d(&tok_in, 1, E);

		for (size_t e = 0; e < topk; e++) {
			tensor_t gate_w, up_w, down_w;

			/* Get expert weight views from 3D tensors */
			tensor_at_q(hl->gate_exps, expert_ids[e], &gate_w);  /* (EI, E) */
			tensor_at_q(hl->up_exps, expert_ids[e], &up_w);      /* (EI, E) */
			tensor_at_q(hl->down_exps, expert_ids[e], &down_w);  /* (E, EI) */

			if (!fused_ffn_silu(&tok_in, expert_out,
					    expert_gate, &gate_w,
					    expert_up, &up_w, &down_w)) {
				tensor_mma_transposed_2x2(expert_gate, &tok_in, &gate_w, NULL);
				tensor_mma_transposed_2x2(expert_up, &tok_in, &up_w, NULL);
				silu(expert_gate);
				tensor_mul(expert_gate, expert_gate, expert_up);
				tensor_mma_transposed_2x2(expert_out, expert_gate, &down_w, NULL);
			}

			/* Accumulate: output[t] += weight * expert_out */
			tensor_t tok_out;
			tensor_at(output, t, &tok_out);

			tensor_reshape_1d(expert_out, E);
			tensor_div_scalar(expert_out, expert_out, 1.0f / expert_weights[e]);
			tensor_add(&tok_out, &tok_out, expert_out);
			tensor_reshape_2d(expert_out, 1, E);
		}
	}
}

static void olmoe_forward(struct olmoe *model, int *tok, int *pos, size_t T,
                           tensor_t *output, enum kv_mode mode)
{
	size_t E = model->embeddings;
	size_t H = model->heads;
	size_t HLEN = model->head_len;

	tensor_t *hidden = model->state.hidden;
	tensor_t *q = model->state.q;
	tensor_t *k = model->state.k;
	tensor_t *v = model->state.v;
	tensor_t *attn = model->state.attn;
	tensor_t *attn_residual = model->state.attn_residual;
	tensor_t *moe_out = model->state.moe_out;

	/* Embed tokens (no position embeddings — RoPE is used instead) */
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
		struct olmoe_layer *hl = &model->hl[l];

		/* Pre-attention RMSNorm */
		tensor_copy(output, hidden);
		rms_norm(q, output, hl->attn_norm);
		tensor_trace(q, "L%zu.norm1", l);

		/* Separate Q/K/V projections */
		tensor_copy(output, q);
		tensor_mma_transposed_2x2(q, output, hl->q_weight, NULL);
		tensor_mma_transposed_2x2(k, output, hl->k_weight, NULL);
		tensor_mma_transposed_2x2(v, output, hl->v_weight, NULL);
		tensor_trace(q, "L%zu.Q", l);
		tensor_trace(k, "L%zu.K", l);
		tensor_trace(v, "L%zu.V", l);

		/* Q/K normalization (full-vector RMSNorm, applied before head split) */
		tensor_copy(output, q);
		rms_norm(q, output, hl->q_norm);
		tensor_copy(output, k);
		rms_norm(k, output, hl->k_norm);

		/* Apply RoPE to Q and K */
		for (size_t t_idx = 0; t_idx < T; t_idx++) {
			tensor_t qt, kt;
			tensor_at(q, t_idx, &qt);
			tensor_reshape_2d(&qt, H, HLEN);
			rope_apply(&qt, pos[t_idx], HLEN, model->rope_theta);

			tensor_at(k, t_idx, &kt);
			tensor_reshape_2d(&kt, H, HLEN);
			rope_apply(&kt, pos[t_idx], HLEN, model->rope_theta);
		}
		tensor_trace(NULL, "L%zu.rope", l);

		/* Multi-head attention */
		transformer(model, q, k, v, attn, l, mode);
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

		/* MoE FFN */
		moe_ffn(model, moe_out, output, l);
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

void olmoe_prefill(struct olmoe *model, int *tok, int *pos, size_t T, tensor_t *output)
{
	olmoe_forward(model, tok, pos, T, output, KV_PREFILL);
	model->cache->size += T;
}

void olmoe_decode(struct olmoe *model, int tok, int pos, tensor_t *output)
{
	if (model->cache->size >= model->cache->context)
		kvcache_rotate(model->cache);

	if (pos >= (int)model->cache->context)
		pos = model->cache->size;

	olmoe_forward(model, &tok, &pos, 1, output, KV_DECODE);
	model->cache->size++;
}

void olmoe_generate(void *ctx, const char *text, int num, pick_token_t f, void *cb_ctx)
{
	struct olmoe *model = ctx;
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
	olmoe_prefill(model, toks, poss, T, output);
	uint64_t prefill_end = profiler_now();

	model->pos += T;

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
		olmoe_decode(model, tok, model->pos, output);
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

void olmoe_close(void *ctx)
{
	struct olmoe *model = ctx;

	tensor_free_mapped(model->wte);
	tensor_free_mapped(model->lm_head);
	tensor_free_mapped(model->output_norm);

	for (size_t i = 0; i < model->layers; i++) {
		tensor_free_mapped(model->hl[i].attn_norm);
		tensor_free_mapped(model->hl[i].q_weight);
		tensor_free_mapped(model->hl[i].k_weight);
		tensor_free_mapped(model->hl[i].v_weight);
		tensor_free_mapped(model->hl[i].q_norm);
		tensor_free_mapped(model->hl[i].k_norm);
		tensor_free_mapped(model->hl[i].o_weight);
		tensor_free_mapped(model->hl[i].ffn_norm);
		tensor_free_mapped(model->hl[i].gate_inp);
		tensor_free_mapped(model->hl[i].gate_exps);
		tensor_free_mapped(model->hl[i].up_exps);
		tensor_free_mapped(model->hl[i].down_exps);
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
	tensor_free(model->state.gate_logits);
	tensor_free(model->state.expert_gate);
	tensor_free(model->state.expert_up);
	tensor_free(model->state.expert_out);
	tensor_free(model->state.moe_out);
	tensor_free(model->state.logits);

	kvcache_free(model->cache);

	free(model);
}

static const struct model olmoe_model = {
	.name = "olmoe",
	.load = olmoe_load,
	.generate = olmoe_generate,
	.close = olmoe_close,
};

__attribute__((constructor))
static void olmoe_register(void)
{
	register_model(&olmoe_model);
}

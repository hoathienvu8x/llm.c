#pragma once

#include <stddef.h>

#include "tensor.h"
#include "model.h"

struct gguf;
struct llama;

void *llama_load(struct gguf *g);
void llama_prefill(struct llama *model, int *tok, int *pos, size_t T, tensor_t *output);
void llama_decode(struct llama *model, int tok, int pos, tensor_t *output);
void llama_generate(void *ctx, const char *text, int num, pick_token_t f, void *cb_ctx);
void llama_close(void *ctx);

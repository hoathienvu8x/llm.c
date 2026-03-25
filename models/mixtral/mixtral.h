#pragma once

#include <stddef.h>

#include "tensor.h"
#include "model.h"

struct gguf;
struct mixtral;

void *mixtral_load(struct gguf *g);
void mixtral_prefill(struct mixtral *model, int *tok, int *pos, size_t T, tensor_t *output);
void mixtral_decode(struct mixtral *model, int tok, int pos, tensor_t *output);
void mixtral_generate(void *ctx, const char *text, int num, pick_token_t f, void *cb_ctx);
void mixtral_close(void *ctx);

#pragma once

#include <stddef.h>

#include "tensor.h"
#include "model.h"

struct gguf;
struct olmoe;

void *olmoe_load(struct gguf *g);
void olmoe_prefill(struct olmoe *model, int *tok, int *pos, size_t T, tensor_t *output);
void olmoe_decode(struct olmoe *model, int tok, int pos, tensor_t *output);
void olmoe_generate(void *ctx, const char *text, int num, pick_token_t f, void *cb_ctx);
void olmoe_close(void *ctx);

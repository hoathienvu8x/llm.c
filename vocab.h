#pragma once

#include <stddef.h>

enum vocab_token_type {
	VOCAB_TOKEN_NORMAL  = 1,
	VOCAB_TOKEN_CONTROL = 3,
	VOCAB_TOKEN_BYTE    = 6,
};

struct gguf;

const char *vocab_encode(struct gguf *g, size_t token);
int vocab_decode(struct gguf *g, const char *s, int *sz);
int vocab_tokenize(struct gguf *g, const char *text, int *out, int max);
void vocab_free(struct gguf *g);

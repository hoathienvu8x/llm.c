#pragma once

#include <stdint.h>

static inline uint32_t fnv1a(const char *s, int len)
{
	uint32_t h = 0x811c9dc5;
	for (int i = 0; i < len; i++) {
		h ^= (unsigned char)s[i];
		h *= 0x01000193;
	}
	return h;
}

static inline uint32_t fnv1a_str(const char *s)
{
	uint32_t h = 0x811c9dc5;
	while (*s) {
		h ^= (unsigned char)*s++;
		h *= 0x01000193;
	}
	return h;
}

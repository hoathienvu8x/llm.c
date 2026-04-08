#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include "gguf.h"
#include "vocab.h"
#include "hash.h"

#define VOCAB_KEY "tokenizer.ggml.tokens"
#define MERGES_KEY "tokenizer.ggml.merges"

struct vocab_entry {
	const char *str;
	int str_len;
	int token_idx;
};

struct merge_entry {
	const char *pair; /* "piece1\0piece2\0" */
	int p1_len, p2_len;
	int rank;
};

struct vocab_ht {
	struct vocab_entry *buckets;
	size_t mask;
	int max_token_len;
	char **decoded; /* decoded token strings, indexed by token_id */
	char **raw;     /* raw GGUF token strings, indexed by token_id */
	enum vocab_token_type *token_type;
	size_t n_tokens;
	int is_bpe;
	int has_bracket_tokens; /* vocab contains [FOO] style special tokens */
	float *scores;  /* SentencePiece token scores for Viterbi */

	/* BPE merge rules */
	struct merge_entry *merge_buckets;
	size_t merge_mask;
	size_t n_merges;

	/* BPE raw token lookup: raw GGUF string → token_id */
	struct vocab_entry *raw_buckets;
	size_t raw_mask;
};

static uint32_t fnv1a_pair(const char *s1, int l1, const char *s2, int l2)
{
	uint32_t h = 0x811c9dc5;
	for (int i = 0; i < l1; i++) {
		h ^= (unsigned char)s1[i];
		h *= 0x01000193;
	}
	h ^= 0xff;
	h *= 0x01000193;
	for (int i = 0; i < l2; i++) {
		h ^= (unsigned char)s2[i];
		h *= 0x01000193;
	}
	return h;
}

static void ht_insert(struct vocab_ht *ht, const char *str, int str_len,
		       int token_idx, int token_type)
{
	uint32_t h = fnv1a(str, str_len) & ht->mask;

	while (ht->buckets[h].str) {
		struct vocab_entry *e = &ht->buckets[h];
		if (e->str_len == str_len && memcmp(e->str, str, str_len) == 0) {
			/* Prefer normal tokens over byte/control tokens
			 * when multiple tokens decode to same string */
			int old_type = ht->token_type ?
				ht->token_type[e->token_idx] : VOCAB_TOKEN_NORMAL;
			if (token_type == VOCAB_TOKEN_NORMAL && old_type != VOCAB_TOKEN_NORMAL)
				e->token_idx = token_idx;
			return;
		}
		h = (h + 1) & ht->mask;
	}

	ht->buckets[h].str = str;
	ht->buckets[h].str_len = str_len;
	ht->buckets[h].token_idx = token_idx;
}

static int ht_lookup(struct vocab_ht *ht, const char *s, int len)
{
	uint32_t h = fnv1a(s, len) & ht->mask;

	while (ht->buckets[h].str) {
		struct vocab_entry *e = &ht->buckets[h];
		if (e->str_len == len && memcmp(e->str, s, len) == 0)
			return e->token_idx;
		h = (h + 1) & ht->mask;
	}

	return -1;
}

static void raw_ht_insert(struct vocab_ht *ht, const char *str, int str_len,
			   int token_idx)
{
	uint32_t h = fnv1a(str, str_len) & ht->raw_mask;

	while (ht->raw_buckets[h].str) {
		struct vocab_entry *e = &ht->raw_buckets[h];
		if (e->str_len == str_len && memcmp(e->str, str, str_len) == 0) {
			e->token_idx = token_idx;
			return;
		}
		h = (h + 1) & ht->raw_mask;
	}

	ht->raw_buckets[h].str = str;
	ht->raw_buckets[h].str_len = str_len;
	ht->raw_buckets[h].token_idx = token_idx;
}

static int raw_ht_lookup(struct vocab_ht *ht, const char *s, int len)
{
	uint32_t h = fnv1a(s, len) & ht->raw_mask;

	while (ht->raw_buckets[h].str) {
		struct vocab_entry *e = &ht->raw_buckets[h];
		if (e->str_len == len && memcmp(e->str, s, len) == 0)
			return e->token_idx;
		h = (h + 1) & ht->raw_mask;
	}

	return -1;
}

static void merge_insert(struct vocab_ht *ht, const char *s1, int l1,
			  const char *s2, int l2, int rank)
{
	uint32_t h = fnv1a_pair(s1, l1, s2, l2) & ht->merge_mask;

	while (ht->merge_buckets[h].pair) {
		h = (h + 1) & ht->merge_mask;
	}

	/* Store pair as "s1\0s2\0" */
	char *pair = malloc(l1 + l2 + 2);
	memcpy(pair, s1, l1);
	pair[l1] = '\0';
	memcpy(pair + l1 + 1, s2, l2);
	pair[l1 + 1 + l2] = '\0';

	ht->merge_buckets[h].pair = pair;
	ht->merge_buckets[h].p1_len = l1;
	ht->merge_buckets[h].p2_len = l2;
	ht->merge_buckets[h].rank = rank;
}

static int merge_lookup(struct vocab_ht *ht, const char *s1, int l1,
			const char *s2, int l2)
{
	uint32_t h = fnv1a_pair(s1, l1, s2, l2) & ht->merge_mask;

	while (ht->merge_buckets[h].pair) {
		struct merge_entry *e = &ht->merge_buckets[h];
		if (e->p1_len == l1 && e->p2_len == l2 &&
		    memcmp(e->pair, s1, l1) == 0 &&
		    memcmp(e->pair + l1 + 1, s2, l2) == 0)
			return e->rank;
		h = (h + 1) & ht->merge_mask;
	}

	return -1;
}

/*
 * GPT-2 byte-level BPE: byte values are mapped to Unicode code points
 * so that every token is a valid Unicode string. This function reverses
 * that mapping: Unicode code point -> original byte value.
 */
static int bpe_char_to_byte(uint32_t cp)
{
	if (cp >= 33 && cp <= 126) return cp;
	if (cp >= 161 && cp <= 172) return cp;
	if (cp >= 174 && cp <= 255) return cp;
	if (cp >= 256 && cp <= 323) {
		int idx = cp - 256;
		if (idx < 33) return idx;	/* bytes 0-32 */
		idx -= 33;
		if (idx < 34) return 127 + idx;	/* bytes 127-160 */
		return 173;			/* byte 173 */
	}
	return -1;
}

/* byte -> GPT-2 unicode code point */
static uint32_t byte_to_bpe_char(uint8_t b)
{
	if (b >= 33 && b <= 126) return b;
	if (b >= 161 && b <= 172) return b;
	if (b >= 174 && b <= 255) return b;
	/* Remaining 68 bytes map to U+0100..U+0143 */
	if (b < 33) return 256 + b;
	if (b >= 127 && b <= 160) return 256 + 33 + (b - 127);
	/* b == 173 */
	return 256 + 33 + 34;
}

/* Encode a single unicode code point as UTF-8, return length */
static int utf8_encode(uint32_t cp, char *out)
{
	if (cp < 0x80) {
		out[0] = cp;
		return 1;
	} else if (cp < 0x800) {
		out[0] = 0xC0 | (cp >> 6);
		out[1] = 0x80 | (cp & 0x3F);
		return 2;
	} else if (cp < 0x10000) {
		out[0] = 0xE0 | (cp >> 12);
		out[1] = 0x80 | ((cp >> 6) & 0x3F);
		out[2] = 0x80 | (cp & 0x3F);
		return 3;
	}
	out[0] = 0xF0 | (cp >> 18);
	out[1] = 0x80 | ((cp >> 12) & 0x3F);
	out[2] = 0x80 | ((cp >> 6) & 0x3F);
	out[3] = 0x80 | (cp & 0x3F);
	return 4;
}

/* Convert a raw byte string to GPT-2 unicode representation (UTF-8) */
static char *bytes_to_bpe_str(const uint8_t *bytes, int len)
{
	char *out = malloc(len * 4 + 1); /* worst case: 4 UTF-8 bytes per input byte */
	int pos = 0;
	for (int i = 0; i < len; i++) {
		uint32_t cp = byte_to_bpe_char(bytes[i]);
		pos += utf8_encode(cp, out + pos);
	}
	out[pos] = '\0';
	return out;
}

/* Decode a GPT-2 BPE token (UTF-8 encoded) back to raw bytes. */
static char *bpe_decode(const char *token)
{
	size_t max_len = strlen(token) + 1;
	char *out = malloc(max_len);
	size_t pos = 0;

	const uint8_t *p = (const uint8_t *)token;
	while (*p) {
		uint32_t cp;
		int nbytes;

		if (*p < 0x80) {
			cp = *p;
			nbytes = 1;
		} else if ((*p & 0xE0) == 0xC0) {
			cp = (uint32_t)(*p & 0x1F) << 6 | (p[1] & 0x3F);
			nbytes = 2;
		} else if ((*p & 0xF0) == 0xE0) {
			cp = (uint32_t)(*p & 0x0F) << 12 |
			     (uint32_t)(p[1] & 0x3F) << 6 | (p[2] & 0x3F);
			nbytes = 3;
		} else {
			cp = (uint32_t)(*p & 0x07) << 18 |
			     (uint32_t)(p[1] & 0x3F) << 12 |
			     (uint32_t)(p[2] & 0x3F) << 6 | (p[3] & 0x3F);
			nbytes = 4;
		}

		int b = bpe_char_to_byte(cp);
		if (b >= 0)
			out[pos++] = (char)b;

		p += nbytes;
	}
	out[pos] = '\0';

	return out;
}

/*
 * SentencePiece (Llama/Mixtral) token decoding:
 *   - ▁ (U+2581) maps to space
 *   - <0xNN> maps to raw byte value NN
 *   - Everything else is literal UTF-8
 */
static char *spm_decode(const char *token)
{
	size_t len = strlen(token);
	char *out = malloc(len + 1);
	size_t pos = 0;

	const uint8_t *p = (const uint8_t *)token;
	while (*p) {
		/* ▁ is U+2581, encoded as E2 96 81 in UTF-8 */
		if (p[0] == 0xe2 && p[1] == 0x96 && p[2] == 0x81) {
			out[pos++] = ' ';
			p += 3;
		} else if (p[0] == '<' && p[1] == '0' && p[2] == 'x'
			   && len >= 6 && p[5] == '>') {
			/* <0xNN> byte escape */
			unsigned val = 0;
			for (int i = 3; i < 5; i++) {
				val <<= 4;
				if (p[i] >= '0' && p[i] <= '9')
					val |= p[i] - '0';
				else if (p[i] >= 'A' && p[i] <= 'F')
					val |= p[i] - 'A' + 10;
				else if (p[i] >= 'a' && p[i] <= 'f')
					val |= p[i] - 'a' + 10;
			}
			out[pos++] = (char)val;
			p += 6;
		} else {
			out[pos++] = *p++;
		}
	}
	out[pos] = '\0';

	return out;
}

static struct vocab_ht *vocab_ht_build(struct gguf *g)
{
	size_t n = gguf_get_arr_n(g, VOCAB_KEY);
	size_t cap = 1;
	while (cap < 2 * n)
		cap <<= 1;

	const char *tok_model = gguf_get_str(g, "tokenizer.ggml.model");
	int is_spm = tok_model && strcmp(tok_model, "llama") == 0;
	int is_bpe = !is_spm;

	struct vocab_ht *ht = calloc(1, sizeof(*ht));
	ht->buckets = calloc(cap, sizeof(struct vocab_entry));
	ht->mask = cap - 1;
	ht->max_token_len = 0;
	ht->decoded = calloc(n, sizeof(char *));
	ht->raw = calloc(n, sizeof(char *));
	ht->n_tokens = n;
	ht->is_bpe = is_bpe;

	/* Load token types (enum vocab_token_type) */
	{
		size_t nt = gguf_get_arr_n(g, "tokenizer.ggml.token_type");
		ht->token_type = calloc(n, sizeof(ht->token_type[0]));
		for (size_t i = 0; i < nt && i < n; i++)
			ht->token_type[i] = gguf_get_arr_int32(g, "tokenizer.ggml.token_type", i);
	}

	/* Load token scores (for SentencePiece Viterbi) */
	if (is_spm) {
		size_t ns = gguf_get_arr_n(g, "tokenizer.ggml.scores");
		ht->scores = calloc(n, sizeof(float));
		for (size_t i = 0; i < ns && i < n; i++)
			ht->scores[i] = gguf_get_arr_float32(g, "tokenizer.ggml.scores", i);
	}

	/* Raw token lookup table (raw GGUF string → token_id) */
	ht->raw_buckets = calloc(cap, sizeof(struct vocab_entry));
	ht->raw_mask = cap - 1;

	for (size_t i = 0; i < n; i++) {
		const char *tok = gguf_get_arr_str(g, VOCAB_KEY, i);
		ht->raw[i] = strdup(tok);
		ht->decoded[i] = is_spm ? spm_decode(tok) : bpe_decode(tok);
		int slen = strlen(ht->decoded[i]);

		if (slen > ht->max_token_len)
			ht->max_token_len = slen;

		ht_insert(ht, ht->decoded[i], slen, i, ht->token_type[i]);

		raw_ht_insert(ht, ht->raw[i], strlen(ht->raw[i]), i);

		/* Detect [FOO] style special tokens */
		if (slen >= 3 && ht->decoded[i][0] == '[' &&
		    ht->decoded[i][slen-1] == ']')
			ht->has_bracket_tokens = 1;
	}

	/* Load BPE merge rules */
	if (is_bpe) {
		size_t nm = gguf_get_arr_n(g, MERGES_KEY);
		ht->n_merges = nm;

		size_t mcap = 1;
		while (mcap < 2 * nm)
			mcap <<= 1;
		ht->merge_buckets = calloc(mcap, sizeof(struct merge_entry));
		ht->merge_mask = mcap - 1;

		for (size_t i = 0; i < nm; i++) {
			const char *m = gguf_get_arr_str(g, MERGES_KEY, i);
			const char *sp = strchr(m, ' ');
			if (!sp)
				continue;
			int l1 = sp - m;
			int l2 = strlen(sp + 1);
			merge_insert(ht, m, l1, sp + 1, l2, (int)i);
		}
	}

	return ht;
}

static struct vocab_ht *ensure_ht(struct gguf *g)
{
	struct vocab_ht *ht = gguf_get_vocab(g);
	if (!ht) {
		ht = vocab_ht_build(g);
		gguf_set_vocab(g, ht);
	}
	return ht;
}

int vocab_decode(struct gguf *g, const char *s, int *sz)
{
	struct vocab_ht *ht = ensure_ht(g);

	if (*s == 0)
		return -1;

	/* Try special tokens first: <|...|> and [...] patterns */
	if (s[0] == '<' && s[1] == '|') {
		const char *end = strstr(s + 2, "|>");
		if (end) {
			int slen = (end + 2) - s;
			int idx = ht_lookup(ht, s, slen);
			if (idx >= 0) {
				*sz = slen;
				return idx;
			}
		}
	}
	if (s[0] == '[') {
		const char *end = strchr(s + 1, ']');
		if (end) {
			int slen = (end + 1) - s;
			int idx = ht_lookup(ht, s, slen);
			if (idx >= 0) {
				*sz = slen;
				return idx;
			}
		}
	}

	int remaining = strlen(s);

	/* Don't let greedy match cross into special token boundaries */
	const char *p = s + 1;
	while (*p) {
		if ((p[0] == '<' && p[1] == '|') || p[0] == '[') {
			int dist = p - s;
			if (dist < remaining)
				remaining = dist;
			break;
		}
		p++;
	}

	int limit = ht->max_token_len < remaining ? ht->max_token_len : remaining;
	int best_len = 0;
	int best_idx = -1;

	for (int len = 1; len <= limit; len++) {
		int idx = ht_lookup(ht, s, len);
		if (idx >= 0) {
			best_len = len;
			best_idx = idx;
		}
	}

	*sz = best_len;
	return best_idx;
}

/*
 * GPT-2 BPE tokenization: apply BPE merge rules to a "word" (a segment
 * of text between special tokens, pre-split by the GPT-2 regex).
 *
 * The word is first converted to GPT-2 unicode representation (each byte
 * becomes a unicode character), then BPE merges are applied iteratively.
 *
 * Returns the number of tokens produced, written to out[].
 */
static int bpe_word(struct vocab_ht *ht, const uint8_t *word, int word_len,
		    int *out, int max_out)
{
	if (word_len == 0)
		return 0;

	/* Convert each byte to its GPT-2 unicode string (UTF-8) */
	char **pieces = malloc(word_len * sizeof(char *));
	int *piece_lens = malloc(word_len * sizeof(int));
	int n_pieces = 0;

	for (int i = 0; i < word_len; i++) {
		char buf[8];
		uint32_t cp = byte_to_bpe_char(word[i]);
		int len = utf8_encode(cp, buf);
		pieces[n_pieces] = malloc(len + 1);
		memcpy(pieces[n_pieces], buf, len);
		pieces[n_pieces][len] = '\0';
		piece_lens[n_pieces] = len;
		n_pieces++;
	}

	/* Iteratively merge the highest-priority pair */
	while (n_pieces > 1) {
		int best_rank = -1;
		int best_pos = -1;

		for (int i = 0; i < n_pieces - 1; i++) {
			int rank = merge_lookup(ht, pieces[i], piece_lens[i],
						pieces[i+1], piece_lens[i+1]);
			if (rank >= 0 && (best_rank < 0 || rank < best_rank)) {
				best_rank = rank;
				best_pos = i;
			}
		}

		if (best_pos < 0)
			break;

		/* Merge pieces[best_pos] and pieces[best_pos+1] */
		int new_len = piece_lens[best_pos] + piece_lens[best_pos+1];
		char *merged = malloc(new_len + 1);
		memcpy(merged, pieces[best_pos], piece_lens[best_pos]);
		memcpy(merged + piece_lens[best_pos],
		       pieces[best_pos+1], piece_lens[best_pos+1]);
		merged[new_len] = '\0';

		free(pieces[best_pos]);
		free(pieces[best_pos+1]);

		pieces[best_pos] = merged;
		piece_lens[best_pos] = new_len;

		/* Shift remaining pieces down */
		for (int i = best_pos + 1; i < n_pieces - 1; i++) {
			pieces[i] = pieces[i+1];
			piece_lens[i] = piece_lens[i+1];
		}
		n_pieces--;
	}

	/* Look up each piece in the raw token vocabulary */
	int n_out = 0;
	for (int i = 0; i < n_pieces && n_out < max_out; i++) {
		int idx = raw_ht_lookup(ht, pieces[i], piece_lens[i]);
		if (idx >= 0)
			out[n_out++] = idx;
		free(pieces[i]);
	}

	free(pieces);
	free(piece_lens);
	return n_out;
}

/*
 * GPT-2 pretokenization: split text into "words" using a simplified
 * version of the GPT-2 regex pattern. Each word is then BPE-tokenized.
 *
 * Pattern: '(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+
 */
static int gpt2_next_word(const char *s, int *word_len)
{
	const uint8_t *p = (const uint8_t *)s;

	if (*p == 0) return 0;

	/* Contraction: 's 't 'd 'm 'll 've 're */
	if (*p == '\'') {
		if (p[1] == 's' || p[1] == 't' || p[1] == 'd' || p[1] == 'm') {
			*word_len = 2;
			return 1;
		}
		if ((p[1] == 'l' && p[2] == 'l') ||
		    (p[1] == 'v' && p[2] == 'e') ||
		    (p[1] == 'r' && p[2] == 'e')) {
			*word_len = 3;
			return 1;
		}
	}

	/* Optional space + letters */
	if ((*p == ' ' && isalpha(p[1])) || isalpha(*p)) {
		int len = (*p == ' ') ? 1 : 0;
		while (isalpha(p[len])) len++;
		*word_len = len;
		return 1;
	}

	/* Digits: max 3 per group (Llama 3 pattern: \p{N}{1,3}) */
	if (isdigit(*p)) {
		int len = 0;
		while (isdigit(p[len]) && len < 3) len++;
		*word_len = len;
		return 1;
	}

	/* Optional space + non-alphanumeric, non-whitespace */
	if ((*p == ' ' && p[1] && !isspace(p[1]) && !isalnum(p[1])) ||
	    (*p && !isspace(*p) && !isalnum(*p))) {
		int len = (*p == ' ') ? 1 : 0;
		while (p[len] && !isspace(p[len]) && !isalnum(p[len]))
			len++;
		*word_len = len;
		return 1;
	}

	/* Newlines: \s*[\r\n]+ */
	if (*p == '\n' || *p == '\r' ||
	    (isspace(*p) && (p[1] == '\n' || p[1] == '\r'))) {
		int len = 0;
		while (isspace(p[len]) && p[len] != '\n' && p[len] != '\r')
			len++;
		while (p[len] == '\n' || p[len] == '\r')
			len++;
		*word_len = len;
		return 1;
	}

	/* Whitespace not followed by non-whitespace: \s+(?!\S) */
	if (isspace(*p) && (isspace(p[1]) || p[1] == '\0')) {
		int len = 0;
		while (isspace(p[len]) && (isspace(p[len+1]) || p[len+1] == '\0'))
			len++;
		if (len > 0) {
			*word_len = len;
			return 1;
		}
	}

	/* Remaining whitespace: \s+ */
	if (isspace(*p)) {
		int len = 0;
		while (isspace(p[len])) len++;
		*word_len = len;
		return 1;
	}

	/* Fallback: single byte */
	*word_len = 1;
	return 1;
}

int vocab_tokenize(struct gguf *g, const char *text, int *out, int max)
{
	struct vocab_ht *ht = ensure_ht(g);
	int n = 0;

	while (*text && n < max) {
		/* Handle special tokens first */
		if (text[0] == '<' && text[1] == '|') {
			const char *end = strstr(text + 2, "|>");
			if (end) {
				int slen = (end + 2) - text;
				int idx = ht_lookup(ht, text, slen);
				if (idx >= 0) {
					out[n++] = idx;
					text += slen;
					continue;
				}
			}
		}
		if (text[0] == '[') {
			const char *end = strchr(text + 1, ']');
			if (end) {
				int slen = (end + 1) - text;
				int idx = ht_lookup(ht, text, slen);
				if (idx >= 0) {
					out[n++] = idx;
					text += slen;
					continue;
				}
			}
		}

		/* Find next special token boundary (start from text+1
		 * so we make progress if current position is an
		 * unrecognized special token pattern) */
		const char *boundary = NULL;
		const char *p = text + 1;
		while (*p) {
			if (p[0] == '<' && p[1] == '|') {
				const char *end = strstr(p + 2, "|>");
				if (end) {
					int slen = (end + 2) - p;
					if (ht_lookup(ht, p, slen) >= 0) {
						boundary = p;
						break;
					}
				}
			}
			if (p[0] == '[' && ht->has_bracket_tokens) {
				/* Only treat as boundary if [FOO] is a token */
				const char *cb = strchr(p + 1, ']');
				if (cb) {
					int slen = (cb + 1) - p;
					if (ht_lookup(ht, p, slen) >= 0) {
						boundary = p;
						break;
					}
				}
			}
			p++;
		}

		/* Process text up to boundary (or end) */
		int seg_len = boundary ? (int)(boundary - text) : (int)strlen(text);

		if (ht->is_bpe && ht->n_merges > 0) {
			/* BPE: split into words, apply BPE to each */
			const char *seg = text;
			int seg_remaining = seg_len;

			while (seg_remaining > 0 && n < max) {
				int word_len;
				if (!gpt2_next_word(seg, &word_len))
					break;
				if (word_len > seg_remaining)
					word_len = seg_remaining;

				n += bpe_word(ht, (const uint8_t *)seg, word_len,
					      &out[n], max - n);
				seg += word_len;
				seg_remaining -= word_len;
			}
		} else if (ht->scores) {
			/* SentencePiece tokenization using BPE-style
			 * score-based merging (matches llama.cpp's impl).
			 * 1. Normalize: add _ prefix, replace spaces with _
			 * 2. Split into UTF-8 code points
			 * 3. Greedily merge the highest-scored pair
			 */
			int slen = seg_len;

			/* Normalize: ▁ prefix + replace ' ' with ▁.
			 * SentencePiece adds ▁ at the start and after
			 * every special token. */
			int norm_cap = slen * 3 + 4;
			char *norm = malloc(norm_cap);
			int nlen = 0;
			norm[nlen++] = (char)0xe2;
			norm[nlen++] = (char)0x96;
			norm[nlen++] = (char)0x81;
			for (int i = 0; i < slen; i++) {
				if (text[i] == ' ') {
					norm[nlen++] = (char)0xe2;
					norm[nlen++] = (char)0x96;
					norm[nlen++] = (char)0x81;
				} else {
					norm[nlen++] = text[i];
				}
			}
			norm[nlen] = '\0';

			/* Split into UTF-8 code points as symbols */
			struct { int off, len, prev, next; } *syms;
			int nsyms = 0;
			syms = malloc(nlen * sizeof(*syms));
			for (int off = 0; off < nlen; ) {
				int cplen = 1;
				uint8_t c = (uint8_t)norm[off];
				if (c >= 0xc0) cplen = 2;
				if (c >= 0xe0) cplen = 3;
				if (c >= 0xf0) cplen = 4;
				if (off + cplen > nlen) cplen = nlen - off;
				syms[nsyms].off = off;
				syms[nsyms].len = cplen;
				syms[nsyms].prev = nsyms - 1;
				syms[nsyms].next = nsyms + 1;
				nsyms++;
				off += cplen;
			}
			if (nsyms > 0) syms[nsyms-1].next = -1;

			/* BPE-style merging: repeatedly merge the pair
			 * whose combined string has the highest score */
			for (;;) {
				float best = -1e30f;
				int best_i = -1;

				for (int i = 0; i < nsyms; i++) {
					if (syms[i].len == 0) continue;
					int j = syms[i].next;
					if (j < 0) continue;
					int clen = syms[i].len + syms[j].len;
					int idx = raw_ht_lookup(ht,
						norm + syms[i].off, clen);
					if (idx < 0) continue;
					float sc = ht->scores[idx];
					if (sc > best) {
						best = sc;
						best_i = i;
					}
				}

				if (best_i < 0) break;

				/* Merge sym[best_i] with sym[next] */
				int j = syms[best_i].next;
				syms[best_i].len += syms[j].len;
				syms[best_i].next = syms[j].next;
				syms[j].len = 0;
				if (syms[j].next >= 0)
					syms[syms[j].next].prev = best_i;
			}

			/* Emit tokens */
			for (int i = 0; i < nsyms && n < max; i++) {
				if (syms[i].len == 0) continue;
				int idx = raw_ht_lookup(ht,
					norm + syms[i].off, syms[i].len);
				if (idx >= 0) {
					out[n++] = idx;
				} else {
					/* Unknown: output as byte tokens.
					 * SentencePiece stores byte fallbacks as
					 * <0xNN> in the raw vocab. */
					for (int b = 0; b < syms[i].len && n < max; b++) {
						uint8_t byte = (uint8_t)norm[syms[i].off + b];
						char hex[7];
						snprintf(hex, sizeof(hex),
							 "<0x%02X>", byte);
						int bid = raw_ht_lookup(ht, hex, 6);
						if (bid >= 0)
							out[n++] = bid;
					}
				}
			}

			free(syms);
			free(norm);
		} else {
			/* Greedy fallback */
			const char *seg = text;
			int seg_remaining = seg_len;

			while (seg_remaining > 0 && n < max) {
				int limit = ht->max_token_len < seg_remaining ?
					    ht->max_token_len : seg_remaining;
				int best_len = 0, best_idx = -1;
				for (int len = 1; len <= limit; len++) {
					int idx = ht_lookup(ht, seg, len);
					if (idx >= 0) {
						best_len = len;
						best_idx = idx;
					}
				}
				if (best_idx < 0)
					break;
				out[n++] = best_idx;
				seg += best_len;
				seg_remaining -= best_len;
			}
		}

		text += seg_len;
	}

	return n;
}

const char *vocab_encode(struct gguf *g, size_t token)
{
	struct vocab_ht *ht = ensure_ht(g);
	assert(token < ht->n_tokens);
	return ht->decoded[token];
}

void vocab_free(struct gguf *g)
{
	struct vocab_ht *ht = gguf_get_vocab(g);
	if (!ht)
		return;
	for (size_t i = 0; i < ht->n_tokens; i++) {
		free(ht->decoded[i]);
		free(ht->raw[i]);
	}
	free(ht->decoded);
	free(ht->raw);
	free(ht->token_type);
	free(ht->scores);
	free(ht->buckets);
	free(ht->raw_buckets);
	if (ht->merge_buckets) {
		for (size_t i = 0; i <= ht->merge_mask; i++)
			free((void *)ht->merge_buckets[i].pair);
		free(ht->merge_buckets);
	}
	free(ht);
	gguf_set_vocab(g, NULL);
}

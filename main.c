#include "model.h"
#include "gguf.h"
#include "vocab.h"
#include "prompt.h"
#include "tensor_trace.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <signal.h>
#include <string.h>
#include <unistd.h>
#include <execinfo.h>
#include <time.h>

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#endif

struct config {
	float temperature;
	float top_p;
	float rep_penalty;
	int top_k;
	int max_tokens;
	int greedy;
	int raw;
	long seed;
};

static struct config cfg = {
	.temperature = 0.6f,
	.top_p       = 0.9f,
	.rep_penalty = 1.1f,
	.top_k       = 40,
	.max_tokens  = 200,
	.seed        = -1,
};

static void top_k(tensor_t *f, size_t *top_n, scalar_t *top_v, size_t k)
{
	assert(k <= f->totlen);

	for (size_t i = 0; i < k; i++) {
		top_n[i] = 0;
		top_v[i] = f->data[0];
	}

	for (size_t i = 1; i < f->totlen; i++) {
		scalar_t new_v = f->data[i];
		int new_p = -1;

		for (size_t j = 0; j < k; j++) {
			if (new_v > top_v[j])
				new_p = j;
		}

		if (new_p < 0)
			continue;

		for (size_t j = 0; j < k; j++) {
			if (j < new_p) {
				top_n[j] = top_n[j+1];
				top_v[j] = top_v[j+1];
			} else if (j == new_p) {
				top_n[j] = i;
				top_v[j] = new_v;
				break;
			}
		}
	}
}

static size_t recent_tokens[64];
static int recent_count;
static int eos_id = -1; /* set to suppress EOS output in chat mode */

static size_t on_token(void *ctx, tensor_t *logits)
{
	struct gguf *g = ctx;
	size_t token;
	int K = cfg.top_k;
	size_t top_n[K];
	scalar_t top_v[K];

	/* Repetition penalty on raw logits (skip in greedy mode) */
	if (!cfg.greedy) {
		for (int i = 0; i < recent_count; i++) {
			size_t t = recent_tokens[i];
			if (t < logits->totlen) {
				if (logits->data[t] > 0)
					logits->data[t] /= cfg.rep_penalty;
				else
					logits->data[t] *= cfg.rep_penalty;
			}
		}
	}

	top_k(logits, top_n, top_v, K);

	/* Temperature scaling + softmax */
	scalar_t max = top_v[K - 1];
	scalar_t sum = 0;
	for (int i = 0; i < K; i++) {
		top_v[i] = expf((top_v[i] - max) / cfg.temperature);
		sum += top_v[i];
	}
	for (int i = 0; i < K; i++)
		top_v[i] /= sum;

	/* Top-p (nucleus) filtering: zero out low-probability tail */
	scalar_t cumsum = 0;
	for (int i = K - 1; i >= 0; i--) {
		cumsum += top_v[i];
		if (cumsum > cfg.top_p) {
			for (int j = i - 1; j >= 0; j--)
				top_v[j] = 0;
			break;
		}
	}

	/* Re-normalize after top-p */
	sum = 0;
	for (int i = 0; i < K; i++)
		sum += top_v[i];
	for (int i = 0; i < K; i++)
		top_v[i] /= sum;

	/* Sample from distribution */
	if (cfg.greedy) {
		/* Pure argmax over all logits */
		token = 0;
		for (size_t i = 1; i < logits->totlen; i++)
			if (logits->data[i] > logits->data[token])
				token = i;
	} else {
		scalar_t rem = drand48();
		token = top_n[K - 1];
		for (int i = K - 1; i >= 0; i--) {
			if (rem < top_v[i]) {
				token = top_n[i];
				break;
			}
			rem -= top_v[i];
		}
	}
	/* Track recent tokens for repetition penalty */
	if (recent_count < 64)
		recent_tokens[recent_count++] = token;
	else {
		memmove(recent_tokens, recent_tokens + 1, 63 * sizeof(size_t));
		recent_tokens[63] = token;
	}

	if ((int)token != eos_id) {
		printf("%s", vocab_encode(g, token));
		fflush(stdout);
	}
	return token;
}

static void bt(int sig)
{
	void *buf[256];
	int num = backtrace(buf, ARRAY_SIZE(buf));
	char **sym = backtrace_symbols(buf, num);
	if (!sym)
		return;

	printf("\nUgh, I'm done, see the backtrace below.\n");
	printf("$ addr2line -e ./llmc -i <func>+<offset>\n\n");
	for (int i = 0; i < num; i++)
		printf("%s\n", sym[i]);
	free(sym);
}

static char *read_stdin(void)
{
	size_t cap = 4096, len = 0;
	char *buf = malloc(cap);
	assert(buf);

	size_t n;
	while ((n = fread(buf + len, 1, cap - len, stdin)) > 0) {
		len += n;
		if (len == cap) {
			cap *= 2;
			buf = realloc(buf, cap);
			assert(buf);
		}
	}
	buf[len] = '\0';
	return buf;
}

static char *join_args(int argc, char **argv)
{
	size_t sz = 0;
	for (int i = 0; i < argc; i++)
		sz += strlen(argv[i]) + 1;

	char *buf = malloc(sz);
	buf[0] = '\0';

	for (int i = 0; i < argc; i++) {
		if (i > 0)
			strcat(buf, " ");
		strcat(buf, argv[i]);
	}
	return buf;
}

static void generate(const struct model *m, void *ctx,
		     struct chat_template *tmpl, struct gguf *g,
		     const char *input)
{
	/* Strip trailing whitespace from user input (only for chat models) */
	char *trimmed = NULL;
	if (tmpl) {
		size_t len = strlen(input);
		while (len > 0 && (input[len-1] == '\n' || input[len-1] == '\r' ||
				   input[len-1] == ' ' || input[len-1] == '\t'))
			len--;
		if (len != strlen(input)) {
			trimmed = strndup(input, len);
			input = trimmed;
		}
	}

	char *prompt = chat_template_apply(tmpl, input);
	free(trimmed);
	m->generate(ctx, prompt, cfg.max_tokens, on_token, g);
	free(prompt);
}

static void chat_loop(const struct model *m, void *ctx,
		      struct chat_template *tmpl, struct gguf *g)
{
	char line[4096];
	int interactive = isatty(fileno(stdin));
	recent_count = 0;

	if (interactive) {
		printf("> ");
		fflush(stdout);
	}
	while (fgets(line, sizeof(line), stdin)) {
		size_t len = strlen(line);
		if (len > 0 && line[len - 1] == '\n')
			line[--len] = '\0';
		if (len == 0) {
			if (interactive) {
				printf("> ");
				fflush(stdout);
			}
			continue;
		}

		generate(m, ctx, tmpl, g, line);
		if (interactive) {
			printf("\n> ");
			fflush(stdout);
		} else {
			printf("\n");
		}
	}
	if (interactive)
		printf("\n");
}

static void seed_rng(void)
{
	if (cfg.seed >= 0) {
		srand48(cfg.seed);
		return;
	}

	struct timespec ts = {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	srand48(ts.tv_sec * 1000000000 + ts.tv_nsec);
}

static int parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++) {
		if (argv[i][0] != '-')
			break;
		if (strcmp(argv[i], "--greedy") == 0) {
			cfg.greedy = 1;
		} else if (strcmp(argv[i], "--raw") == 0) {
			cfg.raw = 1;
		} else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
			cfg.temperature = strtof(argv[++i], NULL);
		} else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
			cfg.top_p = strtof(argv[++i], NULL);
		} else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
			cfg.top_k = atoi(argv[++i]);
		} else if (strcmp(argv[i], "--rep-penalty") == 0 && i + 1 < argc) {
			cfg.rep_penalty = strtof(argv[++i], NULL);
		} else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
			cfg.max_tokens = atoi(argv[++i]);
		} else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
			cfg.seed = atol(argv[++i]);
		} else {
			fprintf(stderr, "unknown option: %s\n", argv[i]);
			return -1;
		}
	}
	return i;
}

int main(int argc, char *argv[])
{
	signal(SIGABRT, bt);
	tensor_trace_init();

	int argpos = parse_args(argc, argv);
	if (argpos < 0 || argpos >= argc) {
		fprintf(stderr, "Usage: %s [options] <model.gguf> [prompt...]\n"
			"  --greedy            argmax decoding\n"
			"  --temp <f>          temperature (default: 0.6)\n"
			"  --top-p <f>         nucleus sampling (default: 0.9)\n"
			"  --top-k <n>         top-k candidates (default: 40)\n"
			"  --rep-penalty <f>   repetition penalty (default: 1.1)\n"
			"  --max-tokens <n>    max tokens to generate (default: 200)\n"
			"  --seed <n>          random seed\n"
			"  --raw               read stdin as raw prompt (no chat template)\n",
			argv[0]);
		return EXIT_FAILURE;
	}

	fprintf(stderr, "loading model from %s\n", argv[argpos]);
	struct gguf *g = gguf_load(argv[argpos]);
	if (!g) {
		fprintf(stderr, "failed to load model from '%s'\n", argv[argpos]);
		return EXIT_FAILURE;
	}

	const char *model_name = gguf_get_str(g, "general.architecture");
	const struct model *m = find_model(model_name);
	if (!m) {
		fprintf(stderr, "unknown model '%s'\n", model_name);
		return EXIT_FAILURE;
	}

	seed_rng();

	void *ctx = m->load(g);
	if (!ctx) {
		fprintf(stderr, "failed to load model\n");
		return EXIT_FAILURE;
	}

	struct chat_template *tmpl = chat_template_load(g);

	if (argpos + 1 < argc) {
		char *inp = join_args(argc - argpos - 1, argv + argpos + 1);
		recent_count = 0;
		generate(m, ctx, tmpl, g, inp);
		free(inp);
	} else if (cfg.raw) {
		char *inp = read_stdin();
		recent_count = 0;
		generate(m, ctx, NULL, g, inp);
		free(inp);
	} else {
		eos_id = gguf_get_uint32(g, "tokenizer.ggml.eos_token_id");
		chat_loop(m, ctx, tmpl, g);
	}

	tensor_trace_shutdown();

	chat_template_free(tmpl);
	m->close(ctx);
	gguf_close(g);
}

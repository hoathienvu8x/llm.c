#include "model.h"
#include "gguf.h"
#include "vocab.h"
#include "prompt.h"

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

static size_t on_token(void *ctx, tensor_t *logits)
{
	struct gguf *g = ctx;
	size_t top_n[5];
	scalar_t top_v[5];

	top_k(logits, top_n, top_v, 5);

	scalar_t max = top_v[0];
	for (size_t i = 0; i < 5; i++)
		if (top_v[i] > max)
			max = top_v[i];

	scalar_t sum = 0;
	for (size_t i = 0; i < 5; i++) {
		top_v[i] = expf(top_v[i] - max);
		sum += top_v[i];
	}
	for (size_t i = 0; i < 5; i++)
		top_v[i] /= sum;

	scalar_t rem = drand48();
	size_t idx = 0;
	for (int i = 4; i >= 0; i--) {
		if (rem > top_v[i]) {
			rem -= top_v[i];
			continue;
		}
		idx = i;
		break;
	}

	size_t token = top_n[idx];
	printf("%s", vocab_encode(g, token));
	fflush(stdout);
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

	char *buf = malloc(sz + 1);
	assert(buf);
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
		     const char *input, int max_tokens)
{
	char *prompt = chat_template_apply(tmpl, input);
	m->generate(ctx, prompt, max_tokens, on_token, g);
	free(prompt);
}

static void chat_loop(const struct model *m, void *ctx,
		      struct chat_template *tmpl, struct gguf *g)
{
	char line[4096];

	printf("> ");
	fflush(stdout);
	while (fgets(line, sizeof(line), stdin)) {
		size_t len = strlen(line);
		if (len > 0 && line[len - 1] == '\n')
			line[--len] = '\0';
		if (len == 0) {
			printf("> ");
			fflush(stdout);
			continue;
		}

		generate(m, ctx, tmpl, g, line, 500);
		printf("\n> ");
		fflush(stdout);
	}
	printf("\n");
}

static void seed_rng(void)
{
	const char *env = getenv("SRAND48_SEED");
	if (env) {
		srand48(atoi(env));
		return;
	}

	struct timespec ts = {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	srand48(ts.tv_sec * 1000000000 + ts.tv_nsec);
}

int main(int argc, char *argv[])
{
	signal(SIGABRT, bt);

	if (argc < 2) {
		fprintf(stderr, "Usage: %s <model.gguf> [prompt...]\n", argv[0]);
		return EXIT_FAILURE;
	}

	fprintf(stderr, "loading model from %s\n", argv[1]);
	struct gguf *g = gguf_load(argv[1]);
	if (!g) {
		fprintf(stderr, "failed to load model from '%s'\n", argv[1]);
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

	if (argc >= 3) {
		char *inp = join_args(argc - 2, argv + 2);
		generate(m, ctx, tmpl, g, inp, 250);
		free(inp);
	} else if (!isatty(fileno(stdin))) {
		char *inp = read_stdin();
		generate(m, ctx, tmpl, g, inp, 250);
		free(inp);
	} else {
		chat_loop(m, ctx, tmpl, g);
	}

	chat_template_free(tmpl);
	m->close(ctx);
	gguf_close(g);
}

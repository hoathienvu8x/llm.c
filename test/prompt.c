#include "prompt.h"
#include "vocab.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>

static void test_template(const char *name, struct gguf *g)
{
	struct chat_template *t = chat_template_load(g);
	if (!t) {
		printf("%s: no template\n", name);
		return;
	}

	printf("%s:\n", name);
	printf("  bos=%d eos=%d\n", t->bos_id, t->eos_id);
	printf("  user: pre='%s' suf='%s'\n", t->user.prefix, t->user.suffix);
	printf("  asst: pre='%s' suf='%s' eos_after=%d\n",
	       t->assistant.prefix, t->assistant.suffix, t->eos_after_assistant);
	printf("  sys:  pre='%s' suf='%s'\n", t->system.prefix, t->system.suffix);
	printf("  gen_prompt: '%s'\n", t->gen_prompt ? t->gen_prompt : "(null)");

	char *formatted = chat_template_apply(t, "Hello, world!");
	printf("  applied: '%s'\n", formatted);
	free(formatted);

	chat_template_free(t);
}

static void assert_single_token(struct gguf *g, const char *text)
{
	int sz;
	int tok = vocab_decode(g, text, &sz);
	printf("  '%s' -> tok=%d sz=%d: ", text, tok, sz);
	assert(tok >= 0);
	assert(sz == (int)strlen(text));
	printf("ok\n");
}

static void assert_boundary(struct gguf *g, const char *text, const char *special)
{
	int sz;
	vocab_decode(g, text, &sz);
	const char *sp = strstr(text, special);
	assert(sp);
	printf("  '%s' boundary at '%s' sz=%d: ", text, special, sz);
	assert(sz <= (int)(sp - text));
	printf("ok\n");
}

static void test_special_tokens(const char *name, struct gguf *g)
{
	printf("%s special tokens:\n", name);

	if (strstr(name, "mistral") && !strstr(name, "mixtral")) {
		assert_single_token(g, "[INST]");
		assert_single_token(g, "[/INST]");
		assert_boundary(g, "hi [/INST]", "[/INST]");
		assert_boundary(g, "hello [INST] world", "[INST]");
	} else if (strstr(name, "llama")) {
		assert_single_token(g, "<|start_header_id|>");
		assert_single_token(g, "<|end_header_id|>");
		assert_single_token(g, "<|eot_id|>");
		assert_boundary(g, "?<|eot_id|>", "<|eot_id|>");
		assert_boundary(g, "hello<|start_header_id|>", "<|start_header_id|>");
	}
}

int main(int argc, char **argv)
{
	for (int i = 1; i < argc; i++) {
		struct gguf *g = gguf_load(argv[i]);
		if (!g) {
			fprintf(stderr, "failed to load %s\n", argv[i]);
			continue;
		}
		test_template(argv[i], g);
		test_special_tokens(argv[i], g);
		gguf_close(g);
	}
	printf("prompt: ok\n");
	return 0;
}

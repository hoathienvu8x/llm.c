#include "prompt.h"
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

int main(int argc, char **argv)
{
	for (int i = 1; i < argc; i++) {
		struct gguf *g = gguf_load(argv[i]);
		if (!g) {
			fprintf(stderr, "failed to load %s\n", argv[i]);
			continue;
		}
		test_template(argv[i], g);
		gguf_close(g);
	}
	printf("prompt: ok\n");
	return 0;
}

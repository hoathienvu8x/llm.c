#pragma once

#include "gguf.h"

struct chat_role {
	char *prefix;  /* text before message content */
	char *suffix;  /* text after message content */
};

struct chat_template {
	int bos_id;
	int eos_id;
	struct chat_role user;
	struct chat_role assistant;
	struct chat_role system;
	char *gen_prompt;        /* appended to trigger assistant generation */
	char *system_preamble;   /* full system block to prepend on first turn */
	int eos_after_assistant; /* append eos_token after assistant suffix */
	int system_sent;         /* system preamble already emitted */
};

struct chat_template *chat_template_load(struct gguf *g);
char *chat_template_apply(struct chat_template *t, const char *user_msg);
void chat_template_free(struct chat_template *t);

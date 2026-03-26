#include "prompt.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

static const char *find(const char *haystack, const char *needle)
{
	return strstr(haystack, needle);
}

static char *extract_string_literal(const char **pos)
{
	const char *p = *pos;
	char quote;

	while (*p == ' ' || *p == '\t' || *p == '\n')
		p++;

	if (*p != '\'' && *p != '"')
		return NULL;

	quote = *p++;
	const char *start = p;

	while (*p && *p != quote) {
		if (*p == '\\') p++;
		p++;
	}

	size_t len = p - start;
	char *s = malloc(len + 1);
	size_t out = 0;
	for (size_t i = 0; i < len; i++) {
		if (start[i] == '\\' && i + 1 < len) {
			i++;
			switch (start[i]) {
			case 'n': s[out++] = '\n'; break;
			case 't': s[out++] = '\t'; break;
			case '\\': s[out++] = '\\'; break;
			default: s[out++] = start[i]; break;
			}
		} else {
			s[out++] = start[i];
		}
	}
	s[out] = '\0';

	if (*p == quote) p++;
	*pos = p;
	return s;
}

static void parse_output_expr(const char *expr_start, const char *expr_end,
			      char **prefix, char **suffix, int *has_eos)
{
	*prefix = strdup("");
	*suffix = strdup("");
	*has_eos = 0;

	const char *content = find(expr_start, "message['content']");
	if (!content)
		content = find(expr_start, "message[\"content\"]");
	if (!content || content >= expr_end)
		return;

	const char *p = expr_start;
	while (*p == ' ' || *p == '{') p++;

	if (*p == '\'' || *p == '"') {
		free(*prefix);
		*prefix = extract_string_literal(&p);
	}

	p = content + strlen("message['content']");
	while (*p == ' ' || *p == '+') p++;

	if (p < expr_end && (*p == '\'' || *p == '"')) {
		free(*suffix);
		*suffix = extract_string_literal(&p);
	}

	if (find(content, "eos_token") && find(content, "eos_token") < expr_end)
		*has_eos = 1;
}

static int find_output_expr(const char *start, const char *end,
			    const char **expr_start, const char **expr_end)
{
	const char *p = start;

	while (p < end) {
		const char *open = find(p, "{{");
		if (!open || open >= end)
			return 0;

		const char *close = find(open, "}}");
		if (!close || close >= end)
			return 0;

		if (find(open, "raise_exception") &&
		    find(open, "raise_exception") < close) {
			p = close + 2;
			continue;
		}

		*expr_start = open;
		*expr_end = close + 2;
		return 1;
	}
	return 0;
}

static const char *find_block_end(const char *start)
{
	const char *p = start;
	int depth = 0;

	while (*p) {
		if (p[0] == '{' && p[1] == '%') {
			const char *tag = p + 2;
			while (*tag == ' ') tag++;

			if (strncmp(tag, "if ", 3) == 0 ||
			    strncmp(tag, "if(", 3) == 0) {
				depth++;
			} else if (depth == 0 &&
				   (strncmp(tag, "elif ", 5) == 0 ||
				    strncmp(tag, "else", 4) == 0 ||
				    strncmp(tag, "endif", 5) == 0)) {
				return p;
			} else if (strncmp(tag, "endif", 5) == 0) {
				depth--;
			}
			const char *close = find(p, "%}");
			if (close) p = close + 2;
			else break;
		} else {
			p++;
		}
	}
	return NULL;
}

static void parse_role_block(const char *tmpl, const char *role_name,
			     struct chat_role *role, int *has_eos)
{
	char pattern[64];
	const char *search_from = tmpl;

	*has_eos = 0;
	role->prefix = strdup("");
	role->suffix = strdup("");

	snprintf(pattern, sizeof(pattern), "message['role'] == '%s'", role_name);

	while (search_from) {
		const char *block = find(search_from, pattern);
		if (!block) {
			snprintf(pattern, sizeof(pattern),
				 "message[\"role\"] == \"%s\"", role_name);
			block = find(search_from, pattern);
		}
		if (!block)
			return;

		const char *block_end = find_block_end(block);
		if (!block_end)
			return;

		const char *expr_start, *expr_end;
		if (find_output_expr(block, block_end, &expr_start, &expr_end)) {
			free(role->prefix);
			free(role->suffix);
			parse_output_expr(expr_start, expr_end,
					  &role->prefix, &role->suffix, has_eos);
			return;
		}

		search_from = block_end;
	}
}

static char *find_gen_prompt(const char *tmpl)
{
	const char *p = find(tmpl, "add_generation_prompt");
	if (!p)
		return NULL;

	const char *expr = find(p, "{{");
	if (!expr)
		return NULL;

	expr += 2;
	while (*expr == ' ') expr++;

	return extract_string_literal(&expr);
}

struct chat_template *chat_template_load(struct gguf *g)
{
	const char *tmpl = gguf_get_str(g, "tokenizer.chat_template");
	if (!tmpl)
		return NULL;

	struct chat_template *t = calloc(1, sizeof(*t));
	t->bos_id = gguf_get_uint32(g, "tokenizer.ggml.bos_token_id");
	t->eos_id = gguf_get_uint32(g, "tokenizer.ggml.eos_token_id");

	int user_eos = 0, asst_eos = 0, sys_eos = 0;
	parse_role_block(tmpl, "user", &t->user, &user_eos);
	parse_role_block(tmpl, "assistant", &t->assistant, &asst_eos);
	parse_role_block(tmpl, "system", &t->system, &sys_eos);
	t->eos_after_assistant = asst_eos;
	t->gen_prompt = find_gen_prompt(tmpl);

	return t;
}

char *chat_template_apply(struct chat_template *t, const char *user_msg)
{
	if (!t)
		return strdup(user_msg);

	size_t len = strlen(t->user.prefix) + strlen(user_msg) +
		     strlen(t->user.suffix);
	if (t->gen_prompt)
		len += strlen(t->gen_prompt);
	if (!t->gen_prompt && t->assistant.prefix[0])
		len += strlen(t->assistant.prefix);

	char *out = malloc(len + 1);
	out[0] = '\0';

	strcat(out, t->user.prefix);
	strcat(out, user_msg);
	strcat(out, t->user.suffix);

	if (t->gen_prompt)
		strcat(out, t->gen_prompt);
	else if (t->assistant.prefix[0])
		strcat(out, t->assistant.prefix);

	return out;
}

void chat_template_free(struct chat_template *t)
{
	if (!t) return;
	free(t->user.prefix);
	free(t->user.suffix);
	free(t->assistant.prefix);
	free(t->assistant.suffix);
	free(t->system.prefix);
	free(t->system.suffix);
	free(t->gen_prompt);
	free(t);
}

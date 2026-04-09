#include "tools.h"
#include "json.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#define TOOLS_MAX 16

static struct tool tools[TOOLS_MAX];
static int tools_count = 0;

void tool_register(const struct tool *t)
{
	assert(tools_count < TOOLS_MAX);
	tools[tools_count++] = *t;
}

int tools_get_count(void)
{
	return tools_count;
}

const struct tool *tools_get(int i)
{
	return &tools[i];
}

char *tools_execute(const char *call_json)
{
	const char *p = call_json;
	while (*p == ' ' || *p == '[') p++;

	char *name = jsonr_str(p, "name");
	if (!name) {
		jsonw_t *j = jsonw_new();
		jsonw_obj(j);
		jsonw_str(j, "error", "no function name");
		jsonw_obj_end(j);
		return jsonw_done(j);
	}

	char *args = jsonr_obj(p, "arguments");
	if (!args)
		args = jsonr_obj(p, "parameters");
	if (!args)
		args = strdup("{}");

	char *result = NULL;
	for (int i = 0; i < tools_count; i++) {
		if (strcmp(tools[i].name, name) == 0) {
			result = tools[i].execute(args);
			break;
		}
	}

	if (!result) {
		jsonw_t *j = jsonw_new();
		jsonw_obj(j);
		jsonw_str(j, "error", name);
		jsonw_obj_end(j);
		result = jsonw_done(j);
	}

	free(name);
	free(args);
	return result;
}

void tools_format_params(jsonw_t *j, const struct tool *t)
{
	jsonw_key(j, "parameters");
	jsonw_obj(j);
	jsonw_str(j, "type", "object");

	jsonw_key(j, "properties");
	jsonw_obj(j);
	if (t->params) {
		for (int i = 0; t->params[i].name; i++) {
			jsonw_key(j, t->params[i].name);
			jsonw_obj(j);
			jsonw_str(j, "type", t->params[i].type);
			jsonw_str(j, "description", t->params[i].description);
			jsonw_obj_end(j);
		}
	}
	jsonw_obj_end(j);

	jsonw_key(j, "required");
	jsonw_arr(j);
	if (t->params) {
		for (int i = 0; t->params[i].name; i++) {
			if (t->params[i].required) {
				if (j->need_comma) fprintf(j->f, ",");
				fprintf(j->f, "\"%s\"", t->params[i].name);
				j->need_comma = 1;
			}
		}
	}
	jsonw_arr_end(j);

	jsonw_obj_end(j);
}

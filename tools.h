#pragma once

#include <stddef.h>
#include "json.h"

struct tool_param {
	const char *name;
	const char *type;
	const char *description;
	int required;
};

struct tool {
	const char *name;
	const char *description;
	const struct tool_param *params;
	char *(*execute)(const char *args_json);
};

void tool_register(const struct tool *t);
int tools_get_count(void);
const struct tool *tools_get(int i);

char *tools_execute(const char *call_json);
void tools_format_params(jsonw_t *j, const struct tool *t);

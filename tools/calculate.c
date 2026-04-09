#include "tools.h"
#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *execute(const char *args)
{
	char *expr = jsonr_str(args, "expression");
	if (!expr) {
		jsonw_t *j = jsonw_new();
		jsonw_obj(j);
		jsonw_str(j, "error", "no expression");
		jsonw_obj_end(j);
		return jsonw_done(j);
	}

	char cmd[256];
	snprintf(cmd, sizeof(cmd), "echo '%s' | bc -l 2>&1", expr);
	free(expr);

	FILE *fp = popen(cmd, "r");
	if (!fp) {
		jsonw_t *j = jsonw_new();
		jsonw_obj(j);
		jsonw_str(j, "error", "popen failed");
		jsonw_obj_end(j);
		return jsonw_done(j);
	}

	char buf[256] = {};
	fgets(buf, sizeof(buf), fp);
	pclose(fp);

	size_t len = strlen(buf);
	if (len > 0 && buf[len-1] == '\n') buf[len-1] = '\0';

	jsonw_t *j = jsonw_new();
	jsonw_obj(j);
	jsonw_str(j, "result", buf);
	jsonw_obj_end(j);
	return jsonw_done(j);
}

static const struct tool_param params[] = {
	{ "expression", "string", "The math expression to evaluate", 1 },
	{}
};

static const struct tool tool_calculate = {
	.name = "calculate",
	.description = "Evaluate a mathematical expression",
	.params = params,
	.execute = execute,
};

__attribute__((constructor))
static void register_calculate(void)
{
	tool_register(&tool_calculate);
}

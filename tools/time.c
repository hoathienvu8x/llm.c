#include "tools.h"
#include "json.h"
#include <time.h>

static char *execute(const char *args)
{
	time_t now = time(NULL);
	struct tm *tm = localtime(&now);
	char buf[64];
	strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", tm);

	jsonw_t *j = jsonw_new();
	jsonw_obj(j);
	jsonw_str(j, "time", buf);
	jsonw_obj_end(j);
	return jsonw_done(j);
}

static const struct tool tool_time = {
	.name = "get_current_time",
	.description = "Get the current date and time",
	.params = NULL,
	.execute = execute,
};

__attribute__((constructor))
static void register_time(void)
{
	tool_register(&tool_time);
}

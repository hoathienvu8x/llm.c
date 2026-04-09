#include "json.h"

#include <stdlib.h>
#include <string.h>

jsonw_t *jsonw_new(void)
{
	jsonw_t *j = calloc(1, sizeof(*j));
	j->f = open_memstream(&j->buf, &j->size);
	return j;
}

void jsonw_obj(jsonw_t *j)
{
	if (j->need_comma)
		fprintf(j->f, ",");
	fprintf(j->f, "{");
	j->need_comma = 0;
}

void jsonw_arr(jsonw_t *j)
{
	if (j->need_comma)
		fprintf(j->f, ",");
	fprintf(j->f, "[");
	j->need_comma = 0;
}

void jsonw_obj_end(jsonw_t *j)
{
	fprintf(j->f, "}");
	j->need_comma = 1;
}

void jsonw_arr_end(jsonw_t *j)
{
	fprintf(j->f, "]");
	j->need_comma = 1;
}

void jsonw_key(jsonw_t *j, const char *key)
{
	if (j->need_comma)
		fprintf(j->f, ",");
	fprintf(j->f, "\"%s\":", key);
	j->need_comma = 0;
}

void jsonw_str(jsonw_t *j, const char *key, const char *val)
{
	if (j->need_comma)
		fprintf(j->f, ",");
	fprintf(j->f, "\"%s\":\"%s\"", key, val);
	j->need_comma = 1;
}

void jsonw_num(jsonw_t *j, const char *key, double val)
{
	if (j->need_comma)
		fprintf(j->f, ",");
	fprintf(j->f, "\"%s\":%g", key, val);
	j->need_comma = 1;
}

void jsonw_bool(jsonw_t *j, const char *key, int val)
{
	if (j->need_comma)
		fprintf(j->f, ",");
	fprintf(j->f, "\"%s\":%s", key, val ? "true" : "false");
	j->need_comma = 1;
}

char *jsonw_done(jsonw_t *j)
{
	fclose(j->f);
	char *buf = j->buf;
	free(j);
	return buf;
}

jsonw_t *trace_begin(const char *event)
{
	jsonw_t *j = jsonw_new();
	jsonw_obj(j);
	jsonw_str(j, "event", event);
	return j;
}

void trace_end(jsonw_t *j)
{
	jsonw_obj_end(j);
	char *s = jsonw_done(j);
	fprintf(stderr, "%s\n", s);
	free(s);
}

static const char *find_key(const char *json, const char *key)
{
	char pattern[64];
	snprintf(pattern, sizeof(pattern), "\"%s\"", key);

	const char *p = strstr(json, pattern);
	if (!p)
		return NULL;

	p += strlen(pattern);
	while (*p == ' ' || *p == ':')
		p++;
	return p;
}

char *jsonr_str(const char *json, const char *key)
{
	const char *p = find_key(json, key);
	if (!p || *p != '"')
		return NULL;
	p++;

	const char *end = p;
	while (*end && *end != '"') {
		if (*end == '\\')
			end++;
		end++;
	}

	return strndup(p, end - p);
}

char *jsonr_obj(const char *json, const char *key)
{
	const char *p = find_key(json, key);
	if (!p || *p != '{')
		return NULL;

	int depth = 0;
	const char *start = p;
	while (*p) {
		if (*p == '{') {
			depth++;
		} else if (*p == '}') {
			if (--depth == 0) {
				p++;
				break;
			}
		} else if (*p == '"') {
			p++;
			while (*p && *p != '"') {
				if (*p == '\\')
					p++;
				p++;
			}
		}
		p++;
	}

	return strndup(start, p - start);
}

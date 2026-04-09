#pragma once

#include <stdio.h>

typedef struct {
	FILE *f;
	char *buf;
	size_t size;
	int need_comma;
} jsonw_t;

jsonw_t *jsonw_new(void);
void jsonw_obj(jsonw_t *j);
void jsonw_arr(jsonw_t *j);
void jsonw_obj_end(jsonw_t *j);
void jsonw_arr_end(jsonw_t *j);
void jsonw_key(jsonw_t *j, const char *key);
void jsonw_str(jsonw_t *j, const char *key, const char *val);
void jsonw_num(jsonw_t *j, const char *key, double val);
void jsonw_bool(jsonw_t *j, const char *key, int val);
char *jsonw_done(jsonw_t *j);

jsonw_t *trace_begin(const char *event);
void trace_end(jsonw_t *j);

char *jsonr_str(const char *json, const char *key);
char *jsonr_obj(const char *json, const char *key);

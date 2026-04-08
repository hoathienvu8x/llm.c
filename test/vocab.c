#include "vocab.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/*
 * Vocab tokenizer test: reads expected tokenizations from a file
 * and compares against vocab_tokenize() output.
 *
 * Expected file format (one test per line):
 *   TEXT\tTOK1 TOK2 TOK3 ...
 * Where TEXT uses \n \t \\ escapes.
 */

static char *unescape(const char *s)
{
	size_t len = strlen(s);
	char *out = malloc(len + 1);
	size_t j = 0;

	for (size_t i = 0; i < len; i++) {
		if (s[i] == '\\' && i + 1 < len) {
			switch (s[i+1]) {
			case 'n': out[j++] = '\n'; i++; break;
			case 't': out[j++] = '\t'; i++; break;
			case '\\': out[j++] = '\\'; i++; break;
			default: out[j++] = s[i]; break;
			}
		} else {
			out[j++] = s[i];
		}
	}
	out[j] = '\0';
	return out;
}

static int run_tests(struct gguf *g, const char *expected_file)
{
	FILE *f = fopen(expected_file, "r");
	if (!f) {
		fprintf(stderr, "cannot open %s\n", expected_file);
		return 1;
	}

	char line[16384];
	int test_num = 0;
	int failures = 0;

	while (fgets(line, sizeof(line), f)) {
		/* Skip comments and empty lines */
		if (line[0] == '#' || line[0] == '\n')
			continue;

		/* Strip newline */
		size_t len = strlen(line);
		if (len > 0 && line[len-1] == '\n')
			line[--len] = '\0';

		/* Split on tab */
		char *tab = strchr(line, '\t');
		if (!tab) continue;
		*tab = '\0';

		char *text = unescape(line);
		const char *expected_toks = tab + 1;

		/* Parse expected token IDs */
		int expected[4096];
		int n_expected = 0;
		{
			const char *p = expected_toks;
			while (*p) {
				while (*p == ' ') p++;
				if (*p == '\0') break;
				expected[n_expected++] = atoi(p);
				while (*p && *p != ' ') p++;
			}
		}

		/* Tokenize with our code */
		int got[4096];
		int n_got = vocab_tokenize(g, text, got, 4096);

		/* Compare */
		test_num++;
		int ok = (n_got == n_expected);
		if (ok) {
			for (int i = 0; i < n_got; i++) {
				if (got[i] != expected[i]) {
					ok = 0;
					break;
				}
			}
		}

		if (!ok) {
			failures++;
			fprintf(stderr, "FAIL test %d: '%s'\n", test_num, line);
			fprintf(stderr, "  expected (%d):", n_expected);
			for (int i = 0; i < n_expected; i++)
				fprintf(stderr, " %d", expected[i]);
			fprintf(stderr, "\n  got      (%d):", n_got);
			for (int i = 0; i < n_got; i++)
				fprintf(stderr, " %d", got[i]);
			fprintf(stderr, "\n");

			/* Show first difference */
			int max = n_got > n_expected ? n_got : n_expected;
			for (int i = 0; i < max; i++) {
				int e = i < n_expected ? expected[i] : -1;
				int g2 = i < n_got ? got[i] : -1;
				if (e != g2) {
					fprintf(stderr, "  first diff at [%d]: expected=%d got=%d",
						i, e, g2);
					if (g2 >= 0)
						fprintf(stderr, " ('%s')", vocab_encode(g, g2));
					fprintf(stderr, "\n");
					break;
				}
			}
		} else {
			printf("  ok %d: %s (%d tokens)\n", test_num, line, n_got);
		}

		free(text);
	}

	fclose(f);

	if (failures > 0) {
		fprintf(stderr, "%d/%d tests FAILED\n", failures, test_num);
		return 1;
	}

	printf("  all %d vocab tests passed\n", test_num);
	return 0;
}

int main(int argc, char **argv)
{
	if (argc < 3) {
		fprintf(stderr, "Usage: %s <model.gguf> <expected.txt>\n", argv[0]);
		return 1;
	}

	struct gguf *g = gguf_load(argv[1]);
	if (!g) {
		fprintf(stderr, "failed to load %s\n", argv[1]);
		return 1;
	}

	int ret = run_tests(g, argv[2]);

	vocab_free(g);
	gguf_close(g);
	return ret;
}

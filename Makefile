LIBS=-lm -lpthread

#BLAS_CFLAGS=$(shell pkg-config --cflags cblas)
#BLAS_LDFLAGS=$(shell pkg-config --libs cblas)

# OpenBLAS
#BLAS_CFLAGS=-I/usr/include/openblas -DUSE_CBLAS
#BLAS_LDFLAGS=-lopenblas

# Intel MKL
#BLAS_CFLAGS=$(shell pkg-config --cflags mkl-dynamic-lp64-iomp) -DUSE_CBLAS
#BLAS_LDFLAGS=$(shell pkg-config --libs mkl-dynamic-lp64-iomp)

# ROCm
#BLAS_CFLAGS=-I/opt/rocm/include -DUSE_CBLAS
#BLAS_LDFLAGS=-L/opt/rocm/lib -lrocblas -Dcblas_sgemm=rocblas_sgemm

#SLEEF_CFLAGS=-I$(HOME)/src/sleef/install/include -D USE_SLEEF
#SLEEF_LDFLAGS=-L$(HOME)/src/sleef/install/lib
#LIBS+=-lsleef

O=3

MODEL_DIRS=$(wildcard models/*)
MODEL_CFLAGS=$(addprefix -I,$(MODEL_DIRS))
MODEL_SRCS=$(wildcard models/*/*.c)
TOOL_SRCS=$(wildcard tools/*.c)

FEATURES=-DFLASH_ATTENTION -DFUSED_GEMV -DQ8_DOT

CFLAGS=$(SLEEF_CFLAGS) $(BLAS_CFLAGS) -I. $(MODEL_CFLAGS) -O$(O) -march=native -rdynamic $(FEATURES)
LDFLAGS=$(SLEEF_LDFLAGS) $(BLAS_LDFLAGS)
CC=clang

COMMON_SRCS=tensor.c quant.c matmul.c thread_pool.c

M=124M
#M=355M
#M=774M
#M=1558M

FG=/home/sdf/src/FlameGraph

SEED=--seed 1337

all: download check

download:
	test -f gpt2_$(M).gguf || models/gpt2/download_gguf.sh
	test -f olmoe-1b-7b-q4_k_m.gguf || models/olmoe/download_gguf.sh
	test -f mixtral-8x7b-Q4_K_M.gguf || models/mixtral/download_gguf.sh
	test -f llama-3.2-3b-instruct-Q4_K_M.gguf || models/llama/download_gguf.sh

build:
	$(CC) $(LDFLAGS) $(CFLAGS) -g main.c $(MODEL_SRCS) $(TOOL_SRCS) model.c nn.c kvcache.c gguf.c vocab.c prompt.c tools.c json.c $(COMMON_SRCS) tensor_trace.c $(LIBS) -o llmc

run: build
	./llmc mistral-7b-instruct-v0.3-Q4_K_M.gguf

check: build
	$(CC) $(LDFLAGS) $(CFLAGS) -g test/tensor.c $(COMMON_SRCS) $(LIBS) && ./a.out
	$(CC) $(LDFLAGS) $(CFLAGS) -g test/simd.c $(LIBS) && ./a.out
	$(CC) $(LDFLAGS) $(CFLAGS) -g test/nn.c $(COMMON_SRCS) nn.c $(LIBS) && ./a.out
	$(CC) $(LDFLAGS) $(CFLAGS) -g test/rope.c $(COMMON_SRCS) nn.c $(LIBS) && ./a.out
	$(CC) $(LDFLAGS) $(CFLAGS) -g test/gguf.c gguf.c vocab.c $(COMMON_SRCS) $(LIBS) && ./a.out gpt2_$(M).gguf
	$(CC) $(LDFLAGS) $(CFLAGS) -g test/quant.c $(COMMON_SRCS) $(LIBS) && ./a.out
	$(CC) $(LDFLAGS) $(CFLAGS) -g test/quant_dot.c $(COMMON_SRCS) $(LIBS) && ./a.out
	$(CC) $(LDFLAGS) $(CFLAGS) -g test/matmul.c $(COMMON_SRCS) $(LIBS) && ./a.out
	$(CC) $(LDFLAGS) $(CFLAGS) -g test/prompt.c gguf.c prompt.c vocab.c $(COMMON_SRCS) $(LIBS) && ./a.out mixtral-8x7b-Q4_K_M.gguf mistral-7b-instruct-v0.3-Q4_K_M.gguf olmoe-1b-7b-q4_k_m.gguf gpt2_$(M).gguf llama-3.2-3b-instruct-Q4_K_M.gguf
	$(CC) $(LDFLAGS) $(CFLAGS) -g test/vocab.c gguf.c vocab.c $(COMMON_SRCS) $(LIBS) && ./a.out llama-3.2-3b-instruct-Q4_K_M.gguf test/expected_vocab_llama3.txt && ./a.out mistral-7b-instruct-v0.3-Q4_K_M.gguf test/expected_vocab_mistral.txt
	@for expected in models/*/test/expected_*.txt; do \
		dir=$${expected%/test/*}; \
		variant=$$(basename $$expected .txt | sed 's/^expected_//'); \
		gguf=$$(ls *$$variant.gguf 2>/dev/null | head -1); \
		prefill=$$(ls $$dir/test/prefill*.txt | head -1); \
		got=$$dir/test/got_$$variant.txt; \
		echo "Testing $$gguf..."; \
		./llmc $(SEED) $$gguf < $$prefill > $$got && \
		diff $$expected $$got || exit 1; \
	done
	test/multiturn.sh ./llmc mistral-7b-instruct-v0.3-Q4_K_M.gguf

flamegraph:
	$(MAKE) build O=0
	perf record -F 99 -g -- ./llmc gpt2_$(M).gguf In the morning I was able to
	perf script | $(FG)/stackcollapse-perf.pl > out.perf-folded
	$(FG)/flamegraph.pl out.perf-folded > perf.svg

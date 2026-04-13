#pragma once

#include <string.h>
#include <stdint.h>

typedef float scalar_t;

#include "vector_cpu.h"
#include "vector_avx2.h"
#include "vector_avx512.h"

#if defined(__AVX512F__)

/* https://github.com/flame/blis/blob/master/kernels/skx/3/bli_dgemm_skx_asm_16x12_l2.c */
#define MMA_MR 16
#define MMA_NR 12

typedef avx512_vector_t vector_t;
#define VECTOR_BATCH AVX512_BATCH

#define vector_load                    avx512_vector_load
#define vector_set                     avx512_vector_set
#define vector_store                   avx512_vector_store
#define vector_add                     avx512_vector_add
#define vector_sub                     avx512_vector_sub
#define vector_mul                     avx512_vector_mul
#define vector_div                     avx512_vector_div
#define vector_exp                     avx512_vector_exp
#define vector_tanh                    avx512_vector_tanh
#define vector_i8_to_f32               avx512_vector_i8_to_f32
#define vector_u4_lo_to_f32            avx512_vector_u4_lo_to_f32
#define vector_u4_hi_to_f32            avx512_vector_u4_hi_to_f32
#define vector_u4_lo_to_f32_unsigned   avx512_vector_u4_lo_to_f32_unsigned
#define vector_u4_hi_to_f32_unsigned   avx512_vector_u4_hi_to_f32_unsigned
#define vector_fma                     avx512_vector_fma
#define vector_reduce_sum              avx512_vector_reduce_sum
#define vector_reduce_max              avx512_vector_reduce_max

#elif defined(__AVX2__)

/* https://github.com/flame/blis/blob/master/kernels/haswell/3/bli_gemmtrsm_l_haswell_asm_d6x8.c */
#define MMA_MR 6
#define MMA_NR 8

typedef avx2_vector_t vector_t;
#define VECTOR_BATCH AVX2_BATCH

#define vector_load                    avx2_vector_load
#define vector_set                     avx2_vector_set
#define vector_store                   avx2_vector_store
#define vector_add                     avx2_vector_add
#define vector_sub                     avx2_vector_sub
#define vector_mul                     avx2_vector_mul
#define vector_div                     avx2_vector_div
#define vector_exp                     avx2_vector_exp
#define vector_tanh                    avx2_vector_tanh
#define vector_i8_to_f32               avx2_vector_i8_to_f32
#define vector_u4_lo_to_f32            avx2_vector_u4_lo_to_f32
#define vector_u4_hi_to_f32            avx2_vector_u4_hi_to_f32
#define vector_u4_lo_to_f32_unsigned   avx2_vector_u4_lo_to_f32_unsigned
#define vector_u4_hi_to_f32_unsigned   avx2_vector_u4_hi_to_f32_unsigned
#define vector_fma                     avx2_vector_fma
#define vector_reduce_sum              avx2_vector_reduce_sum
#define vector_reduce_max              avx2_vector_reduce_max

#else

/* https://github.com/flame/blis/blob/master/kernels/sandybridge/3/bli_gemm_sandybridge_asm_d8x4.c */
#define MMA_MR 8
#define MMA_NR 4

typedef cpu_vector_t vector_t;
#define VECTOR_BATCH CPU_BATCH

#define vector_load                    cpu_vector_load
#define vector_set                     cpu_vector_set
#define vector_store                   cpu_vector_store
#define vector_add                     cpu_vector_add
#define vector_sub                     cpu_vector_sub
#define vector_mul                     cpu_vector_mul
#define vector_div                     cpu_vector_div
#define vector_exp                     cpu_vector_exp
#define vector_tanh                    cpu_vector_tanh
#define vector_i8_to_f32               cpu_vector_i8_to_f32
#define vector_u4_lo_to_f32            cpu_vector_u4_lo_to_f32
#define vector_u4_hi_to_f32            cpu_vector_u4_hi_to_f32
#define vector_u4_lo_to_f32_unsigned   cpu_vector_u4_lo_to_f32_unsigned
#define vector_u4_hi_to_f32_unsigned   cpu_vector_u4_hi_to_f32_unsigned
#define vector_fma                     cpu_vector_fma
#define vector_reduce_sum              cpu_vector_reduce_sum
#define vector_reduce_max              cpu_vector_reduce_max

#endif

#define VECTOR_ALIGN (sizeof(scalar_t) * VECTOR_BATCH)

#define for_each_vec(var, len) \
	for (size_t var = 0; var < vector_batches(len); var += VECTOR_BATCH)

static inline size_t vector_batches(size_t nelems)
{
    return nelems - (nelems % VECTOR_BATCH);
}

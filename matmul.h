#pragma once

#include "tensor.h"

/* C = A @ B + bias */
void tensor_mma_2x2_tp(tensor_t *ret, const tensor_t *lhs,
			const tensor_t *rhs, const tensor_t *add);

/* C = A @ B.T + bias */
void tensor_mma_transposed_2x2_tp(tensor_t *ret, const tensor_t *lhs,
				   const tensor_t *rhs, const tensor_t *add);

#ifdef USE_CBLAS
void tensor_mma_2x2_cblas(tensor_t *ret, const tensor_t *lhs,
			   const tensor_t *rhs, const tensor_t *add);

void tensor_mma_transposed_2x2_cblas(tensor_t *ret, const tensor_t *lhs,
				     const tensor_t *rhs, const tensor_t *add);
#endif

/* Naive scalar loops (for testing) */
void tensor_mma_2x2_naive(tensor_t *ret, const tensor_t *lhs,
			   const tensor_t *rhs, const tensor_t *add);
void tensor_mma_transposed_2x2_naive(tensor_t *ret, const tensor_t *lhs,
				     const tensor_t *rhs, const tensor_t *add);

/* Default dispatch: cblas when USE_CBLAS, tiled otherwise */
void tensor_mma_2x2(tensor_t *ret, const tensor_t *lhs,
		     const tensor_t *rhs, const tensor_t *add);
void tensor_mma_transposed_2x2(tensor_t *ret, const tensor_t *lhs,
				const tensor_t *rhs, const tensor_t *add);


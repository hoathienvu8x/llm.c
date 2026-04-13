#pragma once

#include <assert.h>
#include <string.h>
#include "quant.h"

static float cpu_dot_q4_K_q8_K(const block_q4_K *x, const block_q8_K *y, size_t n)
{
	assert(n % QK_K == 0);
	size_t nb = n / QK_K;
	float acc = 0, acc_m = 0;

	static const uint32_t km1 = 0x3f3f3f3f;
	static const uint32_t km2 = 0x0f0f0f0f;
	static const uint32_t km3 = 0x03030303;

	for (size_t i = 0; i < nb; i++) {
		float d = y[i].d * f16_to_f32(x[i].d);
		float dmin = -y[i].d * f16_to_f32(x[i].dmin);

		/* unpack scales */
		uint32_t utmp[4];
		memcpy(utmp, x[i].scales, 12);
		utmp[3] = ((utmp[2] >> 4) & km2) | (((utmp[1] >> 6) & km3) << 4);
		uint32_t uaux = utmp[1] & km1;
		utmp[1] = (utmp[2] & km2) | (((utmp[0] >> 6) & km3) << 4);
		utmp[2] = uaux;
		utmp[0] &= km1;

		uint8_t *raw = (uint8_t *)utmp;
		int16_t sc[8], mn[8];
		for (int k = 0; k < 8; k++) {
			sc[k] = raw[k];
			mn[k] = raw[8 + k];
		}

		/* accumulate min contribution */
		int32_t min_sum = 0;
		for (int k = 0; k < 8; k++)
			min_sum += mn[k] * (y[i].bsums[2*k] + y[i].bsums[2*k+1]);
		acc_m += dmin * min_sum;

		const uint8_t *q4 = x[i].qs;
		const int8_t *q8 = y[i].qs;
		int32_t sumi = 0;

		for (int j = 0; j < QK_K / 64; j++) {
			int32_t s0 = 0, s1 = 0;

			for (int k = 0; k < 32; k++) {
				s0 += (int)(q4[k] & 0xF) * q8[k];
				s1 += (int)(q4[k] >> 4) * q8[32 + k];
			}
			sumi += sc[2*j] * s0 + sc[2*j+1] * s1;
			q4 += 32;
			q8 += 64;
		}
		acc += d * sumi;
	}
	return acc + acc_m;
}

static float cpu_dot_q6_K_q8_K(const block_q6_K *x, const block_q8_K *y, size_t n)
{
	assert(n % QK_K == 0);
	size_t nb = n / QK_K;
	float acc = 0;

	for (size_t i = 0; i < nb; i++) {
		float d = y[i].d * f16_to_f32(x[i].d);
		const uint8_t *ql = x[i].ql;
		const uint8_t *qh = x[i].qh;
		const int8_t *q8 = y[i].qs;
		const int8_t *sc = x[i].scales;
		int32_t sumi = 0;

		for (int j = 0; j < QK_K / 128; j++) {
			/* 4 groups of 32 values, each split into 2x16 with separate scales */
			int is = j * 8;

			/* group 0: ql[0..31] low nibble + qh bits 0-1 */
			int32_t s0a = 0, s0b = 0;
			for (int k = 0; k < 16; k++) {
				s0a += ((int)((ql[k] & 0xF) | (((qh[k] >> 0) & 3) << 4)) - 32) * q8[k];
				s0b += ((int)((ql[k+16] & 0xF) | (((qh[k+16] >> 0) & 3) << 4)) - 32) * q8[k+16];
			}
			sumi += sc[is+0] * s0a + sc[is+1] * s0b;

			/* group 1: ql[32..63] low nibble + qh bits 2-3 */
			int32_t s1a = 0, s1b = 0;
			for (int k = 0; k < 16; k++) {
				s1a += ((int)((ql[k+32] & 0xF) | (((qh[k] >> 2) & 3) << 4)) - 32) * q8[32+k];
				s1b += ((int)((ql[k+48] & 0xF) | (((qh[k+16] >> 2) & 3) << 4)) - 32) * q8[48+k];
			}
			sumi += sc[is+2] * s1a + sc[is+3] * s1b;

			/* group 2: ql[0..31] high nibble + qh bits 4-5 */
			int32_t s2a = 0, s2b = 0;
			for (int k = 0; k < 16; k++) {
				s2a += ((int)((ql[k] >> 4) | (((qh[k] >> 4) & 3) << 4)) - 32) * q8[64+k];
				s2b += ((int)((ql[k+16] >> 4) | (((qh[k+16] >> 4) & 3) << 4)) - 32) * q8[80+k];
			}
			sumi += sc[is+4] * s2a + sc[is+5] * s2b;

			/* group 3: ql[32..63] high nibble + qh bits 6-7 */
			int32_t s3a = 0, s3b = 0;
			for (int k = 0; k < 16; k++) {
				s3a += ((int)((ql[k+32] >> 4) | (((qh[k] >> 6) & 3) << 4)) - 32) * q8[96+k];
				s3b += ((int)((ql[k+48] >> 4) | (((qh[k+16] >> 6) & 3) << 4)) - 32) * q8[112+k];
			}
			sumi += sc[is+6] * s3a + sc[is+7] * s3b;

			ql += 64;
			qh += 32;
			q8 += 128;
		}
		acc += d * sumi;
	}
	return acc;
}

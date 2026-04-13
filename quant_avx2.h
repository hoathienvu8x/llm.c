#pragma once

#include <assert.h>
#include <immintrin.h>
#include <string.h>
#include "quant.h"

static float avx2_dot_q4_K_q8_K(const block_q4_K *x, const block_q8_K *y, size_t n)
{
	assert(n % QK_K == 0);
	size_t nb = n / QK_K;
	float acc = 0, acc_m = 0;

	static const uint32_t km1 = 0x3f3f3f3f;
	static const uint32_t km2 = 0x0f0f0f0f;
	static const uint32_t km3 = 0x03030303;

	static const uint8_t sh[256] = {
		 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
		 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
		 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
		 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
		 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
		10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
		12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
		14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
	};

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

		__m256i ms = _mm256_cvtepu8_epi16(
			_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));

		/* accumulate min contribution */
		__m256i q8s = _mm256_loadu_si256((const __m256i *)y[i].bsums);
		__m128i q8h = _mm_hadd_epi16(
			_mm256_extracti128_si256(q8s, 0),
			_mm256_extracti128_si256(q8s, 1));
		__m128i prod = _mm_madd_epi16(
			_mm256_extracti128_si256(ms, 1), q8h);
		__m128 fp = _mm_cvtepi32_ps(prod);
		fp = _mm_add_ps(fp, _mm_movehl_ps(fp, fp));
		fp = _mm_add_ss(fp, _mm_movehdup_ps(fp));
		acc_m += dmin * _mm_cvtss_f32(fp);

		/* broadcast low 8 scales to both lanes */
		__m128i sc128 = _mm256_extracti128_si256(ms, 0);
		__m256i scales = _mm256_set_m128i(sc128, sc128);

		const uint8_t *q4 = x[i].qs;
		const int8_t *q8 = y[i].qs;
		__m256i sumi = _mm256_setzero_si256();
		__m256i mask_lo = _mm256_set1_epi8(0x0F);

		for (int j = 0; j < QK_K / 64; j++) {
			__m256i sl = _mm256_shuffle_epi8(scales,
				_mm256_loadu_si256((const __m256i *)&sh[(2*j) * 32]));
			__m256i sv = _mm256_shuffle_epi8(scales,
				_mm256_loadu_si256((const __m256i *)&sh[(2*j+1) * 32]));

			__m256i q4b = _mm256_loadu_si256((const __m256i *)q4);
			q4 += 32;
			__m256i q8l = _mm256_loadu_si256((const __m256i *)q8);
			q8 += 32;
			__m256i q8v = _mm256_loadu_si256((const __m256i *)q8);
			q8 += 32;

			__m256i q4lo = _mm256_and_si256(q4b, mask_lo);
			__m256i q4hi = _mm256_and_si256(
				_mm256_srli_epi16(q4b, 4), mask_lo);

			__m256i p16l = _mm256_madd_epi16(sl,
				_mm256_maddubs_epi16(q4lo, q8l));
			__m256i p16h = _mm256_madd_epi16(sv,
				_mm256_maddubs_epi16(q4hi, q8v));

			sumi = _mm256_add_epi32(sumi,
				_mm256_add_epi32(p16l, p16h));
		}

		/* horizontal sum */
		__m128i hi = _mm256_extracti128_si256(sumi, 1);
		__m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(sumi), hi);
		__m128i s2 = _mm_add_epi32(s4, _mm_srli_si128(s4, 8));
		__m128i s1 = _mm_add_epi32(s2, _mm_srli_si128(s2, 4));
		acc += d * _mm_cvtsi128_si32(s1);
	}
	return acc + acc_m;
}

static float avx2_dot_q6_K_q8_K(const block_q6_K *x, const block_q8_K *y, size_t n)
{
	assert(n % QK_K == 0);
	size_t nb = n / QK_K;
	float acc = 0;

	__m256i m4 = _mm256_set1_epi8(0xF);
	__m256i m2 = _mm256_set1_epi8(3);
	__m256i m32s = _mm256_set1_epi8(32);

	static const uint8_t sc_sh[128] = {
		 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
		 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
		 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
		 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
		 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
		10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,
		12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13,
		14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15,
	};

	for (size_t i = 0; i < nb; i++) {
		float d = y[i].d * f16_to_f32(x[i].d);
		const uint8_t *ql = x[i].ql;
		const uint8_t *qh = x[i].qh;
		const int8_t *q8 = y[i].qs;

		__m128i scales = _mm_loadu_si128((const __m128i *)x[i].scales);
		__m256i sumi = _mm256_setzero_si256();
		int is = 0;

		for (int j = 0; j < QK_K / 128; j++) {
			/* shuffle scales to 16-bit */
			__m256i sc0 = _mm256_cvtepi8_epi16(
				_mm_shuffle_epi8(scales, _mm_loadu_si128((const __m128i *)&sc_sh[is * 16])));
			__m256i sc1 = _mm256_cvtepi8_epi16(
				_mm_shuffle_epi8(scales, _mm_loadu_si128((const __m128i *)&sc_sh[(is+1) * 16])));
			__m256i sc2 = _mm256_cvtepi8_epi16(
				_mm_shuffle_epi8(scales, _mm_loadu_si128((const __m128i *)&sc_sh[(is+2) * 16])));
			__m256i sc3 = _mm256_cvtepi8_epi16(
				_mm_shuffle_epi8(scales, _mm_loadu_si128((const __m128i *)&sc_sh[(is+3) * 16])));
			is += 4;

			__m256i q4b1 = _mm256_loadu_si256((const __m256i *)ql);
			ql += 32;
			__m256i q4b2 = _mm256_loadu_si256((const __m256i *)ql);
			ql += 32;
			__m256i q4bH = _mm256_loadu_si256((const __m256i *)qh);
			qh += 32;

			/* reconstruct 6-bit values: low4 | (high2 << 4) */
			__m256i t, q4h0, q4h1, q4h2, q4h3;
			t = _mm256_and_si256(q4bH, m2);
			q4h0 = _mm256_slli_epi16(t, 4);
			t = _mm256_and_si256(_mm256_srli_epi16(q4bH, 2), m2);
			q4h1 = _mm256_slli_epi16(t, 4);
			t = _mm256_and_si256(_mm256_srli_epi16(q4bH, 4), m2);
			q4h2 = _mm256_slli_epi16(t, 4);
			t = _mm256_and_si256(_mm256_srli_epi16(q4bH, 6), m2);
			q4h3 = _mm256_slli_epi16(t, 4);

			__m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4b1, m4), q4h0);
			__m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4b2, m4), q4h1);
			__m256i q4_2 = _mm256_or_si256(
				_mm256_and_si256(_mm256_srli_epi16(q4b1, 4), m4), q4h2);
			__m256i q4_3 = _mm256_or_si256(
				_mm256_and_si256(_mm256_srli_epi16(q4b2, 4), m4), q4h3);

			__m256i q8_0 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
			__m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
			__m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
			__m256i q8_3 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;

			/* dot - offset: maddubs(q6,q8) - maddubs(32,q8) */
			__m256i p0 = _mm256_sub_epi16(
				_mm256_maddubs_epi16(q4_0, q8_0),
				_mm256_maddubs_epi16(m32s, q8_0));
			__m256i p1 = _mm256_sub_epi16(
				_mm256_maddubs_epi16(q4_1, q8_1),
				_mm256_maddubs_epi16(m32s, q8_1));
			__m256i p2 = _mm256_sub_epi16(
				_mm256_maddubs_epi16(q4_2, q8_2),
				_mm256_maddubs_epi16(m32s, q8_2));
			__m256i p3 = _mm256_sub_epi16(
				_mm256_maddubs_epi16(q4_3, q8_3),
				_mm256_maddubs_epi16(m32s, q8_3));

			/* apply scales and accumulate */
			p0 = _mm256_madd_epi16(sc0, p0);
			p1 = _mm256_madd_epi16(sc1, p1);
			p2 = _mm256_madd_epi16(sc2, p2);
			p3 = _mm256_madd_epi16(sc3, p3);

			sumi = _mm256_add_epi32(sumi,
				_mm256_add_epi32(p0, p1));
			sumi = _mm256_add_epi32(sumi,
				_mm256_add_epi32(p2, p3));
		}

		/* horizontal sum */
		__m128i hi = _mm256_extracti128_si256(sumi, 1);
		__m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(sumi), hi);
		__m128i s2 = _mm_add_epi32(s4, _mm_srli_si128(s4, 8));
		__m128i s1 = _mm_add_epi32(s2, _mm_srli_si128(s2, 4));
		acc += d * _mm_cvtsi128_si32(s1);
	}
	return acc;
}

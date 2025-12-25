/* crc32_pclmulqdq_tpl.h -- Compute the CRC32 using a parallelized folding
 * approach with the PCLMULQDQ and VPCMULQDQ instructions.
 *
 * A white paper describing this algorithm can be found at:
 *     doc/crc-pclmulqdq.pdf
 *
 * Copyright (C) 2020 Wangyang Guo (wangyang.guo@intel.com) (VPCLMULQDQ support)
 * Copyright (C) 2013 Intel Corporation. All rights reserved.
 * Copyright (C) 2016 Marian Beermann (support for initial value)
 * Authors:
 *     Wajdi Feghali   <wajdi.k.feghali@intel.com>
 *     Jim Guilford    <james.guilford@intel.com>
 *     Vinodh Gopal    <vinodh.gopal@intel.com>
 *     Erdinc Ozturk   <erdinc.ozturk@intel.com>
 *     Jim Kukunas     <james.t.kukunas@linux.intel.com>
 *
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#if (COPY == 0)
#  define FOLD_16 fold_16
#  define CRC32_COPY_SMALL crc32_copy_small
#  define CRC32_COPY_IMPL crc32_copy_impl
#else
#  define FOLD_16 fold_16_copy
#  define CRC32_COPY_SMALL crc32_copy_small_copy
#  define CRC32_COPY_IMPL crc32_copy_impl_copy
#endif

#ifdef X86_VPCLMULQDQ
static size_t FOLD_16(__m128i *xmm_crc0, __m128i *xmm_crc1, __m128i *xmm_crc2, __m128i *xmm_crc3, uint8_t *dst,
    const uint8_t *src, size_t len, __m128i crc) {
    __m512i zmm_initial = _mm512_zextsi128_si512(crc);
    __m512i zmm_t0, zmm_t1, zmm_t2, zmm_t3;
    __m512i zmm_crc0, zmm_crc1, zmm_crc2, zmm_crc3;
    __m512i z0, z1, z2, z3;
    size_t len_tmp = len;
    const __m512i zmm_fold4 = _mm512_set4_epi32(
        0x00000001, 0x54442bd4, 0x00000001, 0xc6e41596);
    const __m512i zmm_fold16 = _mm512_set4_epi32(
        0x00000001, 0x1542778a, 0x00000001, 0x322d1430);

    // zmm register init
    zmm_crc0 = _mm512_setzero_si512();
    zmm_t0 = _mm512_loadu_si512((__m512i *)src);
    zmm_crc1 = _mm512_loadu_si512((__m512i *)src + 1);
    zmm_crc2 = _mm512_loadu_si512((__m512i *)src + 2);
    zmm_crc3 = _mm512_loadu_si512((__m512i *)src + 3);

    if (COPY) {
        _mm512_storeu_si512((__m512i *)dst, zmm_t0);
        _mm512_storeu_si512((__m512i *)dst + 1, zmm_crc1);
        _mm512_storeu_si512((__m512i *)dst + 2, zmm_crc2);
        _mm512_storeu_si512((__m512i *)dst + 3, zmm_crc3);
        dst += 256;
    }

    // XOR initial CRC
    zmm_t0 = _mm512_xor_si512(zmm_t0, zmm_initial);

    // already have intermediate CRC in xmm registers fold4 with 4 xmm_crc to get zmm_crc0
    zmm_crc0 = _mm512_inserti32x4(zmm_crc0, *xmm_crc0, 0);
    zmm_crc0 = _mm512_inserti32x4(zmm_crc0, *xmm_crc1, 1);
    zmm_crc0 = _mm512_inserti32x4(zmm_crc0, *xmm_crc2, 2);
    zmm_crc0 = _mm512_inserti32x4(zmm_crc0, *xmm_crc3, 3);
    z0 = _mm512_clmulepi64_epi128(zmm_crc0, zmm_fold4, 0x01);
    zmm_crc0 = _mm512_clmulepi64_epi128(zmm_crc0, zmm_fold4, 0x10);
    zmm_crc0 = _mm512_ternarylogic_epi32(zmm_crc0, z0, zmm_t0, 0x96);

    len -= 256;
    src += 256;

    // fold-16 loops
    while (len >= 256) {
        zmm_t0 = _mm512_loadu_si512((__m512i *)src);
        zmm_t1 = _mm512_loadu_si512((__m512i *)src + 1);
        zmm_t2 = _mm512_loadu_si512((__m512i *)src + 2);
        zmm_t3 = _mm512_loadu_si512((__m512i *)src + 3);

        z0 = _mm512_clmulepi64_epi128(zmm_crc0, zmm_fold16, 0x01);
        z1 = _mm512_clmulepi64_epi128(zmm_crc1, zmm_fold16, 0x01);
        z2 = _mm512_clmulepi64_epi128(zmm_crc2, zmm_fold16, 0x01);
        z3 = _mm512_clmulepi64_epi128(zmm_crc3, zmm_fold16, 0x01);

        zmm_crc0 = _mm512_clmulepi64_epi128(zmm_crc0, zmm_fold16, 0x10);
        zmm_crc1 = _mm512_clmulepi64_epi128(zmm_crc1, zmm_fold16, 0x10);
        zmm_crc2 = _mm512_clmulepi64_epi128(zmm_crc2, zmm_fold16, 0x10);
        zmm_crc3 = _mm512_clmulepi64_epi128(zmm_crc3, zmm_fold16, 0x10);

        zmm_crc0 = _mm512_ternarylogic_epi32(zmm_crc0, z0, zmm_t0, 0x96);
        zmm_crc1 = _mm512_ternarylogic_epi32(zmm_crc1, z1, zmm_t1, 0x96);
        zmm_crc2 = _mm512_ternarylogic_epi32(zmm_crc2, z2, zmm_t2, 0x96);
        zmm_crc3 = _mm512_ternarylogic_epi32(zmm_crc3, z3, zmm_t3, 0x96);

        if (COPY) {
            _mm512_storeu_si512((__m512i *)dst, zmm_t0);
            _mm512_storeu_si512((__m512i *)dst + 1, zmm_t1);
            _mm512_storeu_si512((__m512i *)dst + 2, zmm_t2);
            _mm512_storeu_si512((__m512i *)dst + 3, zmm_t3);
            dst += 256;
        }
        len -= 256;
        src += 256;
    }
    // zmm_crc[0,1,2,3] -> zmm_crc0
    z0 = _mm512_clmulepi64_epi128(zmm_crc0, zmm_fold4, 0x01);
    zmm_crc0 = _mm512_clmulepi64_epi128(zmm_crc0, zmm_fold4, 0x10);
    zmm_crc0 = _mm512_ternarylogic_epi32(zmm_crc0, z0, zmm_crc1, 0x96);

    z0 = _mm512_clmulepi64_epi128(zmm_crc0, zmm_fold4, 0x01);
    zmm_crc0 = _mm512_clmulepi64_epi128(zmm_crc0, zmm_fold4, 0x10);
    zmm_crc0 = _mm512_ternarylogic_epi32(zmm_crc0, z0, zmm_crc2, 0x96);

    z0 = _mm512_clmulepi64_epi128(zmm_crc0, zmm_fold4, 0x01);
    zmm_crc0 = _mm512_clmulepi64_epi128(zmm_crc0, zmm_fold4, 0x10);
    zmm_crc0 = _mm512_ternarylogic_epi32(zmm_crc0, z0, zmm_crc3, 0x96);

    // zmm_crc0 -> xmm_crc[0, 1, 2, 3]
    *xmm_crc0 = _mm512_extracti32x4_epi32(zmm_crc0, 0);
    *xmm_crc1 = _mm512_extracti32x4_epi32(zmm_crc0, 1);
    *xmm_crc2 = _mm512_extracti32x4_epi32(zmm_crc0, 2);
    *xmm_crc3 = _mm512_extracti32x4_epi32(zmm_crc0, 3);

    return (len_tmp - len);  // return n bytes processed
}
#endif

static inline uint32_t CRC32_COPY_SMALL(uint32_t crc, uint8_t *dst, const uint8_t *buf, size_t len) {
    uint32_t c = (~crc) & 0xffffffff;

    while (len) {
        len--;
        if (COPY) {
            *dst++ = *buf;
        }
        CRC_DO1;
    }

    return c ^ 0xffffffff;
}

static inline uint32_t CRC32_COPY_IMPL(uint32_t crc, uint8_t *dst, const uint8_t *src, size_t len) {
    size_t copy_len = len;
    if (len >= 16) {
        /* Calculate 16-byte alignment offset */
        unsigned algn_diff = ((uintptr_t)16 - ((uintptr_t)src & 0xF)) & 0xF;

        /* If total length is less than (alignment bytes + 16), use the faster small method.
         * Handles both initially small buffers and cases where alignment would leave < 16 bytes */
        copy_len = len < algn_diff + 16 ? len : algn_diff;
    }

    if (copy_len > 0) {
        crc = CRC32_COPY_SMALL(crc, dst, src, copy_len);
        src += copy_len;
        len -= copy_len;
        if (COPY) {
            dst += copy_len;
        }
    }

    if (len == 0)
        return crc;

    __m128i xmm_t0, xmm_t1, xmm_t2, xmm_t3;
    __m128i xmm_crc_part = _mm_setzero_si128();
    __m128i xmm_crc0 = _mm_cvtsi32_si128(0x9db42487);
    __m128i xmm_crc1 = _mm_setzero_si128();
    __m128i xmm_crc2 = _mm_setzero_si128();
    __m128i xmm_crc3 = _mm_setzero_si128();
    __m128i xmm_initial = _mm_cvtsi32_si128(crc);

#ifdef X86_VPCLMULQDQ
    if (len >= 256) {
        size_t n = FOLD_16(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3, dst, src, len, xmm_initial);
        len -= n;
        src += n;
        if (COPY) {
            dst += n;
        }
        crc = 0;
    }
#endif

    /* Implement Chorba algorithm from https://arxiv.org/abs/2412.16398
     * We interleave the PCLMUL-base folds with 8x scaled generator
     * polynomial copies; we read 8x QWORDS and then XOR them into
     * the stream at the following offsets: 6, 9, 10, 16, 20, 22,
     * 24, 25, 27, 28, 30, 31, 32 - this is detailed in the paper
     * as "generator_64_bits_unrolled_8" */
#ifndef __AVX512VL__
    if (!COPY) {
#endif
    while (len >= 512 + 64 + 16*8) {
        __m128i chorba8 = _mm_load_si128((__m128i *)src);
        __m128i chorba7 = _mm_load_si128((__m128i *)src + 1);
        __m128i chorba6 = _mm_load_si128((__m128i *)src + 2);
        __m128i chorba5 = _mm_load_si128((__m128i *)src + 3);
        __m128i chorba4 = _mm_load_si128((__m128i *)src + 4);
        __m128i chorba3 = _mm_load_si128((__m128i *)src + 5);
        __m128i chorba2 = _mm_load_si128((__m128i *)src + 6);
        __m128i chorba1 = _mm_load_si128((__m128i *)src + 7);
        if (COPY) {
            _mm_storeu_si128((__m128i *)dst, chorba8);
            _mm_storeu_si128((__m128i *)dst + 1, chorba7);
            _mm_storeu_si128((__m128i *)dst + 2, chorba6);
            _mm_storeu_si128((__m128i *)dst + 3, chorba5);
            _mm_storeu_si128((__m128i *)dst + 4, chorba4);
            _mm_storeu_si128((__m128i *)dst + 5, chorba3);
            _mm_storeu_si128((__m128i *)dst + 6, chorba2);
            _mm_storeu_si128((__m128i *)dst + 7, chorba1);
            dst += 16*8;
        }
        XOR_INITIAL128(chorba8);

        chorba2 = _mm_xor_si128(chorba2, chorba8);
        chorba1 = _mm_xor_si128(chorba1, chorba7);
        src += 16*8;
        len -= 16*8;

        xmm_t0 = _mm_load_si128((__m128i *)src);
        xmm_t1 = _mm_load_si128((__m128i *)src + 1);
        xmm_t2 = _mm_load_si128((__m128i *)src + 2);
        xmm_t3 = _mm_load_si128((__m128i *)src + 3);

        fold_12(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);
        if (COPY) {
            _mm_storeu_si128((__m128i *)dst, xmm_t0);
            _mm_storeu_si128((__m128i *)dst + 1, xmm_t1);
            _mm_storeu_si128((__m128i *)dst + 2, xmm_t2);
            _mm_storeu_si128((__m128i *)dst + 3, xmm_t3);
            dst += 64;
        }
        xmm_t0 = _mm_xor_si128(xmm_t0, chorba6);
        xmm_t1 = _mm_xor_si128(_mm_xor_si128(xmm_t1, chorba5), chorba8);
        xmm_t2 = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(xmm_t2, chorba4), chorba8), chorba7);
        xmm_t3 = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(xmm_t3, chorba3), chorba7), chorba6);
        xmm_crc0 = _mm_xor_si128(xmm_t0, xmm_crc0);
        xmm_crc1 = _mm_xor_si128(xmm_t1, xmm_crc1);
        xmm_crc2 = _mm_xor_si128(xmm_t2, xmm_crc2);
        xmm_crc3 = _mm_xor_si128(xmm_t3, xmm_crc3);

        xmm_t0 = _mm_load_si128((__m128i *)src + 4);
        xmm_t1 = _mm_load_si128((__m128i *)src + 5);
        xmm_t2 = _mm_load_si128((__m128i *)src + 6);
        xmm_t3 = _mm_load_si128((__m128i *)src + 7);

        fold_4(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);
        if (COPY) {
            _mm_storeu_si128((__m128i *)dst, xmm_t0);
            _mm_storeu_si128((__m128i *)dst + 1, xmm_t1);
            _mm_storeu_si128((__m128i *)dst + 2, xmm_t2);
            _mm_storeu_si128((__m128i *)dst + 3, xmm_t3);
            dst += 64;
        }

        xmm_t0 = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(xmm_t0, chorba2), chorba6), chorba5);
        xmm_t1 = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(xmm_t1, chorba1), chorba4), chorba5);
        xmm_t2 = _mm_xor_si128(_mm_xor_si128(xmm_t2, chorba3), chorba4);
        xmm_t3 = _mm_xor_si128(_mm_xor_si128(xmm_t3, chorba2), chorba3);
        xmm_crc0 = _mm_xor_si128(xmm_t0, xmm_crc0);
        xmm_crc1 = _mm_xor_si128(xmm_t1, xmm_crc1);
        xmm_crc2 = _mm_xor_si128(xmm_t2, xmm_crc2);
        xmm_crc3 = _mm_xor_si128(xmm_t3, xmm_crc3);

        xmm_t0 = _mm_load_si128((__m128i *)src + 8);
        xmm_t1 = _mm_load_si128((__m128i *)src + 9);
        xmm_t2 = _mm_load_si128((__m128i *)src + 10);
        xmm_t3 = _mm_load_si128((__m128i *)src + 11);

        fold_4(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);
        if (COPY) {
            _mm_storeu_si128((__m128i *)dst, xmm_t0);
            _mm_storeu_si128((__m128i *)dst + 1, xmm_t1);
            _mm_storeu_si128((__m128i *)dst + 2, xmm_t2);
            _mm_storeu_si128((__m128i *)dst + 3, xmm_t3);
            dst += 64;
        }

        xmm_t0 = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(xmm_t0, chorba1), chorba2), chorba8);
        xmm_t1 = _mm_xor_si128(_mm_xor_si128(xmm_t1, chorba1), chorba7);
        xmm_t2 = _mm_xor_si128(xmm_t2, chorba6);
        xmm_t3 = _mm_xor_si128(xmm_t3, chorba5);
        xmm_crc0 = _mm_xor_si128(xmm_t0, xmm_crc0);
        xmm_crc1 = _mm_xor_si128(xmm_t1, xmm_crc1);
        xmm_crc2 = _mm_xor_si128(xmm_t2, xmm_crc2);
        xmm_crc3 = _mm_xor_si128(xmm_t3, xmm_crc3);

        xmm_t0 = _mm_load_si128((__m128i *)src + 12);
        xmm_t1 = _mm_load_si128((__m128i *)src + 13);
        xmm_t2 = _mm_load_si128((__m128i *)src + 14);
        xmm_t3 = _mm_load_si128((__m128i *)src + 15);

        fold_4(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);
        if (COPY) {
            _mm_storeu_si128((__m128i *)dst, xmm_t0);
            _mm_storeu_si128((__m128i *)dst + 1, xmm_t1);
            _mm_storeu_si128((__m128i *)dst + 2, xmm_t2);
            _mm_storeu_si128((__m128i *)dst + 3, xmm_t3);
            dst += 64;
        }

        xmm_t0 = _mm_xor_si128(_mm_xor_si128(xmm_t0, chorba4), chorba8);
        xmm_t1 = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(xmm_t1, chorba3), chorba8), chorba7);
        xmm_t2 = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(_mm_xor_si128(xmm_t2, chorba2), chorba8), chorba7), chorba6);
        xmm_t3 = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(_mm_xor_si128(xmm_t3, chorba1), chorba7), chorba6), chorba5);
        xmm_crc0 = _mm_xor_si128(xmm_t0, xmm_crc0);
        xmm_crc1 = _mm_xor_si128(xmm_t1, xmm_crc1);
        xmm_crc2 = _mm_xor_si128(xmm_t2, xmm_crc2);
        xmm_crc3 = _mm_xor_si128(xmm_t3, xmm_crc3);

        xmm_t0 = _mm_load_si128((__m128i *)src + 16);
        xmm_t1 = _mm_load_si128((__m128i *)src + 17);
        xmm_t2 = _mm_load_si128((__m128i *)src + 18);
        xmm_t3 = _mm_load_si128((__m128i *)src + 19);

        fold_4(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);
        if (COPY) {
            _mm_storeu_si128((__m128i *)dst, xmm_t0);
            _mm_storeu_si128((__m128i *)dst + 1, xmm_t1);
            _mm_storeu_si128((__m128i *)dst + 2, xmm_t2);
            _mm_storeu_si128((__m128i *)dst + 3, xmm_t3);
            dst += 64;
        }

        xmm_t0 = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(_mm_xor_si128(xmm_t0, chorba4), chorba8), chorba6), chorba5);
        xmm_t1 = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(_mm_xor_si128(_mm_xor_si128(xmm_t1, chorba3), chorba4), chorba8), chorba7), chorba5);
        xmm_t2 = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(_mm_xor_si128(_mm_xor_si128(xmm_t2, chorba2), chorba3), chorba4), chorba7), chorba6);
        xmm_t3 = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(_mm_xor_si128(_mm_xor_si128(_mm_xor_si128(xmm_t3, chorba1), chorba2), chorba3), chorba8), chorba6), chorba5);
        xmm_crc0 = _mm_xor_si128(xmm_t0, xmm_crc0);
        xmm_crc1 = _mm_xor_si128(xmm_t1, xmm_crc1);
        xmm_crc2 = _mm_xor_si128(xmm_t2, xmm_crc2);
        xmm_crc3 = _mm_xor_si128(xmm_t3, xmm_crc3);

        xmm_t0 = _mm_load_si128((__m128i *)src + 20);
        xmm_t1 = _mm_load_si128((__m128i *)src + 21);
        xmm_t2 = _mm_load_si128((__m128i *)src + 22);
        xmm_t3 = _mm_load_si128((__m128i *)src + 23);

        fold_4(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);
        if (COPY) {
            _mm_storeu_si128((__m128i *)dst, xmm_t0);
            _mm_storeu_si128((__m128i *)dst + 1, xmm_t1);
            _mm_storeu_si128((__m128i *)dst + 2, xmm_t2);
            _mm_storeu_si128((__m128i *)dst + 3, xmm_t3);
            dst += 64;
        }

        xmm_t0 = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(_mm_xor_si128(_mm_xor_si128(_mm_xor_si128(xmm_t0, chorba1), chorba2), chorba4), chorba8), chorba7), chorba5);
        xmm_t1 = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(_mm_xor_si128(_mm_xor_si128(xmm_t1, chorba1), chorba3), chorba4), chorba7), chorba6);
        xmm_t2 = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(_mm_xor_si128(_mm_xor_si128(xmm_t2, chorba2), chorba3), chorba8), chorba6), chorba5);
        xmm_t3 = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(_mm_xor_si128(_mm_xor_si128(_mm_xor_si128(xmm_t3, chorba1), chorba2), chorba4), chorba8), chorba7), chorba5);
        xmm_crc0 = _mm_xor_si128(xmm_t0, xmm_crc0);
        xmm_crc1 = _mm_xor_si128(xmm_t1, xmm_crc1);
        xmm_crc2 = _mm_xor_si128(xmm_t2, xmm_crc2);
        xmm_crc3 = _mm_xor_si128(xmm_t3, xmm_crc3);

        xmm_t0 = _mm_load_si128((__m128i *)src + 24);
        xmm_t1 = _mm_load_si128((__m128i *)src + 25);
        xmm_t2 = _mm_load_si128((__m128i *)src + 26);
        xmm_t3 = _mm_load_si128((__m128i *)src + 27);

        fold_4(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);
        if (COPY) {
            _mm_storeu_si128((__m128i *)dst, xmm_t0);
            _mm_storeu_si128((__m128i *)dst + 1, xmm_t1);
            _mm_storeu_si128((__m128i *)dst + 2, xmm_t2);
            _mm_storeu_si128((__m128i *)dst + 3, xmm_t3);
            dst += 64;
        }
        xmm_t0 = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(_mm_xor_si128(_mm_xor_si128(_mm_xor_si128(xmm_t0, chorba1), chorba3), chorba4), chorba8), chorba7), chorba6);
        xmm_t1 = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(_mm_xor_si128(_mm_xor_si128(xmm_t1, chorba2), chorba3), chorba7), chorba6), chorba5);
        xmm_t2 = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(_mm_xor_si128(_mm_xor_si128(xmm_t2, chorba1), chorba2), chorba4), chorba6), chorba5);
        xmm_t3 = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(_mm_xor_si128(xmm_t3, chorba1), chorba3), chorba4), chorba5);
        xmm_crc0 = _mm_xor_si128(xmm_t0, xmm_crc0);
        xmm_crc1 = _mm_xor_si128(xmm_t1, xmm_crc1);
        xmm_crc2 = _mm_xor_si128(xmm_t2, xmm_crc2);
        xmm_crc3 = _mm_xor_si128(xmm_t3, xmm_crc3);

        xmm_t0 = _mm_load_si128((__m128i *)src + 28);
        xmm_t1 = _mm_load_si128((__m128i *)src + 29);
        xmm_t2 = _mm_load_si128((__m128i *)src + 30);
        xmm_t3 = _mm_load_si128((__m128i *)src + 31);

        fold_4(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);
        if (COPY) {
            _mm_storeu_si128((__m128i *)dst, xmm_t0);
            _mm_storeu_si128((__m128i *)dst + 1, xmm_t1);
            _mm_storeu_si128((__m128i *)dst + 2, xmm_t2);
            _mm_storeu_si128((__m128i *)dst + 3, xmm_t3);
            dst += 64;
        }
        xmm_t0 = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(xmm_t0, chorba2), chorba3), chorba4);
        xmm_t1 = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(xmm_t1, chorba1), chorba2), chorba3);
        xmm_t2 = _mm_xor_si128(_mm_xor_si128(xmm_t2, chorba1), chorba2);
        xmm_t3 = _mm_xor_si128(xmm_t3, chorba1);
        xmm_crc0 = _mm_xor_si128(xmm_t0, xmm_crc0);
        xmm_crc1 = _mm_xor_si128(xmm_t1, xmm_crc1);
        xmm_crc2 = _mm_xor_si128(xmm_t2, xmm_crc2);
        xmm_crc3 = _mm_xor_si128(xmm_t3, xmm_crc3);

        len -= 512;
        src += 512;
    }
#ifndef __AVX512VL__
    }
#endif

    while (len >= 64) {
        len -= 64;
        xmm_t0 = _mm_load_si128((__m128i *)src);
        xmm_t1 = _mm_load_si128((__m128i *)src + 1);
        xmm_t2 = _mm_load_si128((__m128i *)src + 2);
        xmm_t3 = _mm_load_si128((__m128i *)src + 3);
        src += 64;

        fold_4(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);
        if (COPY) {
            _mm_storeu_si128((__m128i *)dst, xmm_t0);
            _mm_storeu_si128((__m128i *)dst + 1, xmm_t1);
            _mm_storeu_si128((__m128i *)dst + 2, xmm_t2);
            _mm_storeu_si128((__m128i *)dst + 3, xmm_t3);
            dst += 64;
        }
        XOR_INITIAL128(xmm_t0);

        xmm_crc0 = _mm_xor_si128(xmm_crc0, xmm_t0);
        xmm_crc1 = _mm_xor_si128(xmm_crc1, xmm_t1);
        xmm_crc2 = _mm_xor_si128(xmm_crc2, xmm_t2);
        xmm_crc3 = _mm_xor_si128(xmm_crc3, xmm_t3);
    }

    /*
     * len = num bytes left - 64
     */
    if (len >= 48) {
        len -= 48;

        xmm_t0 = _mm_load_si128((__m128i *)src);
        xmm_t1 = _mm_load_si128((__m128i *)src + 1);
        xmm_t2 = _mm_load_si128((__m128i *)src + 2);
        src += 48;
        if (COPY) {
            _mm_storeu_si128((__m128i *)dst, xmm_t0);
            _mm_storeu_si128((__m128i *)dst + 1, xmm_t1);
            _mm_storeu_si128((__m128i *)dst + 2, xmm_t2);
            dst += 48;
        }
        XOR_INITIAL128(xmm_t0);
        fold_3(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);

        xmm_crc1 = _mm_xor_si128(xmm_crc1, xmm_t0);
        xmm_crc2 = _mm_xor_si128(xmm_crc2, xmm_t1);
        xmm_crc3 = _mm_xor_si128(xmm_crc3, xmm_t2);
    } else if (len >= 32) {
        len -= 32;

        xmm_t0 = _mm_load_si128((__m128i *)src);
        xmm_t1 = _mm_load_si128((__m128i *)src + 1);
        src += 32;
        if (COPY) {
            _mm_storeu_si128((__m128i *)dst, xmm_t0);
            _mm_storeu_si128((__m128i *)dst + 1, xmm_t1);
            dst += 32;
        }
        XOR_INITIAL128(xmm_t0);
        fold_2(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);

        xmm_crc2 = _mm_xor_si128(xmm_crc2, xmm_t0);
        xmm_crc3 = _mm_xor_si128(xmm_crc3, xmm_t1);
    } else if (len >= 16) {
        len -= 16;
        xmm_t0 = _mm_load_si128((__m128i *)src);
        src += 16;
        if (COPY) {
            _mm_storeu_si128((__m128i *)dst, xmm_t0);
            dst += 16;
        }
        XOR_INITIAL128(xmm_t0);
        fold_1(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);

        xmm_crc3 = _mm_xor_si128(xmm_crc3, xmm_t0);
    }

    if (len) {
        memcpy(&xmm_crc_part, src, len);
        if (COPY) {
            uint8_t ALIGNED_(16) partial_buf[16] = { 0 };
            _mm_storeu_si128((__m128i *)partial_buf, xmm_crc_part);
            memcpy(dst, partial_buf, len);
        }
        partial_fold(len, &xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3, &xmm_crc_part);
    }

    return fold_final(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);
}

#undef FOLD_16
#undef CRC32_COPY_SMALL
#undef CRC32_COPY_IMPL

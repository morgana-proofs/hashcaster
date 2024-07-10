use std::{arch::x86_64::{__m128i, _mm_clmulepi64_si128, _mm_shuffle_epi32, _mm_slli_epi64, _mm_srli_epi64, _mm_unpacklo_epi64, _mm_xor_si128}, mem::transmute};

// Using polyval impl from rust-crypto as a reference.
// I also use this as opportunity to learn about x86 instructions.

#[inline(always)]
pub fn mul_128(a: u128, b:u128) -> u128 {
    unsafe{transmute(mul(transmute(a), transmute(b)))}
}

#[inline(always)]
unsafe fn mul(a0: __m128i, b0: __m128i) ->  __m128i {
    let a1 = _mm_shuffle_epi32(a0, 0x0E); // [a3; a2; a1; a0], 11 11 11 10 -> [a3; a3; a3; a2]
    let a2 = _mm_xor_si128(a0, a1); // [0; a2+a3; a1+a3; a0+a2]
    
    let b1 = _mm_shuffle_epi32(b0, 0x0E);
    let b2 = _mm_xor_si128(b0, b1);

    let t0 = _mm_clmulepi64_si128(b0, a0, 0x00); // Karatsuba in 0
    let t1 = _mm_clmulepi64_si128(b0, a0, 0x11); // Karatsuba in 1
    let t2 = _mm_clmulepi64_si128(b2, a2, 0x00); // Weirdly, Karatsuba in \infty.
    let t2 = _mm_xor_si128(t2, _mm_xor_si128(t0, t1)); // Subtract t0, t1 back

    // Now t0 + 2^64 t1 + 2^128 t2 is our product

    let v0 = t0;
    let v1 = _mm_xor_si128(_mm_shuffle_epi32(t0, 0x0E), t2);
    let v2 = _mm_xor_si128(t1, _mm_shuffle_epi32(t2, 0x0E));
    let v3 = _mm_shuffle_epi32(t1, 0x0E);


    // Polynomial reduction
    let v2 = xor5(
        v2,
        v0,
        _mm_srli_epi64(v0, 1),
        _mm_srli_epi64(v0, 2),
        _mm_srli_epi64(v0, 7),
    );

    let v1 = xor4(
        v1,
        _mm_slli_epi64(v0, 63),
        _mm_slli_epi64(v0, 62),
        _mm_slli_epi64(v0, 57),
    );

    let v3 = xor5(
        v3,
        v1,
        _mm_srli_epi64(v1, 1),
        _mm_srli_epi64(v1, 2),
        _mm_srli_epi64(v1, 7),
    );

    let v2 = xor4(
        v2,
        _mm_slli_epi64(v1, 63),
        _mm_slli_epi64(v1, 62),
        _mm_slli_epi64(v1, 57),
    );

    _mm_unpacklo_epi64(v2, v3)
}

#[inline(always)]
unsafe fn xor4(e1: __m128i, e2: __m128i, e3: __m128i, e4: __m128i) -> __m128i {
    _mm_xor_si128(_mm_xor_si128(e1, e2), _mm_xor_si128(e3, e4))
}

#[inline(always)]
unsafe fn xor5(e1: __m128i, e2: __m128i, e3: __m128i, e4: __m128i, e5: __m128i) -> __m128i {
    _mm_xor_si128(
        e1,
        _mm_xor_si128(_mm_xor_si128(e2, e3), _mm_xor_si128(e4, e5)),
    )
}
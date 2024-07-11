use std::{mem::transmute, arch::aarch64::{uint8x16_t, veorq_u8, vextq_u8, vmull_p64, vgetq_lane_u64, vreinterpretq_u64_u8, vreinterpretq_u8_p128, vld1q_s8, vandq_u8, vdupq_n_u8, vshlq_u8, vaddv_u8, vget_low_u8, vget_high_u8, vld1q_u8}, time::Instant};

use rand::{rngs::OsRng, RngCore};


pub fn mul_128(x: u128, y:u128) -> u128{
    unsafe{
    let (h, m, l) = karatsuba1(transmute(x), transmute(y));
    let (h, l) = karatsuba2(h, m, l);
    transmute(mont_reduce(h, l))
}
}

/// Karatsuba decomposition for `x*y`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn karatsuba1(x: uint8x16_t, y: uint8x16_t) -> (uint8x16_t, uint8x16_t, uint8x16_t) {
    // First Karatsuba step: decompose x and y.
    //
    // (x1*y0 + x0*y1) = (x1+x0) * (y1+x0) + (x1*y1) + (x0*y0)
    //        M                                 H         L
    //
    // m = x.hi^x.lo * y.hi^y.lo
    let m = pmull(
        veorq_u8(x, vextq_u8(x, x, 8)), // x.hi^x.lo
        veorq_u8(y, vextq_u8(y, y, 8)), // y.hi^y.lo
    );
    let h = pmull2(x, y); // h = x.hi * y.hi
    let l = pmull(x, y); // l = x.lo * y.lo
    (h, m, l)
}

/// Multiplies the low bits in `a` and `b`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn pmull(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    transmute(vmull_p64(
        vgetq_lane_u64(vreinterpretq_u64_u8(a), 0),
        vgetq_lane_u64(vreinterpretq_u64_u8(b), 0),
    ))
}

/// Multiplies the high bits in `a` and `b`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn pmull2(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    transmute(vmull_p64(
        vgetq_lane_u64(vreinterpretq_u64_u8(a), 1),
        vgetq_lane_u64(vreinterpretq_u64_u8(b), 1),
    ))
}

/// Karatsuba combine.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn karatsuba2(h: uint8x16_t, m: uint8x16_t, l: uint8x16_t) -> (uint8x16_t, uint8x16_t) {
    // Second Karatsuba step: combine into a 2n-bit product.
    //
    // m0 ^= l0 ^ h0 // = m0^(l0^h0)
    // m1 ^= l1 ^ h1 // = m1^(l1^h1)
    // l1 ^= m0      // = l1^(m0^l0^h0)
    // h0 ^= l0 ^ m1 // = h0^(l0^m1^l1^h1)
    // h1 ^= l1      // = h1^(l1^m0^l0^h0)
    let t = {
        //   {m0, m1} ^ {l1, h0}
        // = {m0^l1, m1^h0}
        let t0 = veorq_u8(m, vextq_u8(l, h, 8));

        //   {h0, h1} ^ {l0, l1}
        // = {h0^l0, h1^l1}
        let t1 = veorq_u8(h, l);

        //   {m0^l1, m1^h0} ^ {h0^l0, h1^l1}
        // = {m0^l1^h0^l0, m1^h0^h1^l1}
        veorq_u8(t0, t1)
    };

    // {m0^l1^h0^l0, l0}
    let x01 = vextq_u8(
        vextq_u8(l, l, 8), // {l1, l0}
        t,
        8,
    );

    // {h1, m1^h0^h1^l1}
    let x23 = vextq_u8(
        t,
        vextq_u8(h, h, 8), // {h1, h0}
        8,
    );

    (x23, x01)
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn mont_reduce(x23: uint8x16_t, x01: uint8x16_t) -> uint8x16_t {
    // Perform the Montgomery reduction over the 256-bit X.
    //    [A1:A0] = X0 • poly
    //    [B1:B0] = [X0 ⊕ A1 : X1 ⊕ A0]
    //    [C1:C0] = B0 • poly
    //    [D1:D0] = [B0 ⊕ C1 : B1 ⊕ C0]
    // Output: [D1 ⊕ X3 : D0 ⊕ X2]
    let poly = vreinterpretq_u8_p128(1 << 127 | 1 << 126 | 1 << 121 | 1 << 63 | 1 << 62 | 1 << 57);
    let a = pmull(x01, poly);
    let b = veorq_u8(x01, vextq_u8(a, a, 8));
    let c = pmull2(b, poly);
    veorq_u8(x23, veorq_u8(c, b))
}

#[unroll::unroll_for_loops]
fn mm_movemask_epi8(bytes: [u8; 16]) -> u16 {
    let mut mask = 0u16;
    for (i, &byte) in bytes.iter().enumerate() {
        mask |= ((byte & 0x80) as u16 >> 7) << i;
    }
    mask
}
#[unroll::unroll_for_loops]
pub fn cpu_v_movemask_epi8(x: [u8; 16]) -> i32 {
    let mut ret = 0;
    for i in 0..16 {
        ret <<= 1;
        ret += (x[15-i] >> 7) as i32;
    }
    ret
}


pub(crate) fn v_movemask_epi8(input: [u8; 16]) -> i32 {
    let uc_shift: [i8; 16] = [-7, -6, -5, -4, -3, -2, -1, 0, -7, -6, -5, -4, -3, -2, -1, 0];
    let vshift = unsafe { vld1q_s8(uc_shift.as_ptr()) };

    let vmask = unsafe { vandq_u8(transmute(input), vdupq_n_u8(0x80)) };
    let shifted_mask = unsafe { vshlq_u8(vmask, vshift) };

    let lower_sum = unsafe { vaddv_u8(vget_low_u8(shifted_mask)) } as u32;
    let higher_sum = unsafe { vaddv_u8(vget_high_u8(shifted_mask)) } as u32;

    unsafe { transmute(lower_sum + (higher_sum << 8)) } 
}

#[test]
fn bench_movemask() {
    unsafe{
        let rng = &mut OsRng;
        use crate::utils::u128_rand;
        let s = u128_rand(rng);
        let x = rng.next_u32() as i32;

        let n = 100_000_000usize;
        let mut u = x;
        let mut v = s;

        let mut ret = 0;

        let label0 = Instant::now();

        for i in 0..n {
            v += s;
            u *= x;
            u += 1;
        }
        ret += u;

        let label1 = Instant::now();

        for i in 0..n {
            v += s;
            u *= v_movemask_epi8(transmute(v));
            u += 1;
        }

        ret += u;

        let label2 = Instant::now();

        for i in 0..n {
            v += s;
            u *= cpu_v_movemask_epi8(transmute(v));
            u += 1;
        }
        
        ret += u;

        let label3 = Instant::now();

        println!(
            "Native: {} ms\nReference: {} ms\nOffset (must be greater than zero or this thing is lying) {} ms", 
            ((label3 - label2)).as_millis(),
            ((label2 - label1)).as_millis(),
            (label1 - label0).as_millis(),
        );

        println!("{}", ret);
    }
}

#[test]
fn test_for_mm_movemask_aarch64() {
    let bytes: [u8; 16] = [
        0x80, 0x01, 0x80, 0x01, 0x80, 0x01, 0x80, 0x01,
        0x80, 0x01, 0x80, 0x01, 0x80, 0x01, 0x80, 0x01,
    ];
    let input_vector = unsafe { vld1q_u8(bytes.as_ptr()) };
    let start = Instant::now();
    let result = v_movemask_epi8(input_vector);
    let end = Instant::now();
    println!("Olen2a {} nanos", (end - start).as_nanos());

    let start = Instant::now();
    let mask = mm_movemask_epi8(bytes);
    let end = Instant::now();
    println!("Olena {} nanos", (end - start).as_nanos());

    let start = Instant::now();
    let lev_mask = cpu_v_movemask_epi8(bytes);
    let end = Instant::now();
    println!("Lev {} nanos", (end - start).as_nanos());

}
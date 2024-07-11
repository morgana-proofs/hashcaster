use std::{mem::{transmute, MaybeUninit}, time::Instant};

use num_traits::{One, Pow, Zero};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};
use crate::{field::{pi, F128}, parallelize::parallelize, utils::u128_to_bits};
use itertools::Itertools;

pub fn eq_poly(pt: &[F128]) -> Vec<F128> {
    let l = pt.len();
    let mut ret = Vec::with_capacity(1 << l);
    ret.push(F128::one());
    for i in 0..l {
//        let pt_idx = l - i - 1;
        let half = 1 << i;
        for j in 0..half {
            ret.push(pt[i] * ret[j]);
        }
        for j in 0..half{
            let tmp = ret[half + j];
            ret[j] += tmp;
        }
    }
    ret
}

pub fn eq_ev(x: &[F128], y: &[F128]) -> F128 {
    x.iter().zip_eq(y.iter()).fold(F128::one(), |acc, (x, y)| acc * (F128::one() + x + y))
}

pub fn evaluate(poly: &[F128], pt: &[F128]) -> F128 {
    assert!(poly.len() == 1 << pt.len());
    poly.iter().zip_eq(eq_poly(pt)).fold(F128::zero(), |acc, (x, y)| acc + *x * y)
}

pub fn bits_to_trits(mut x: usize) -> usize {
    let mut multiplier = 1;
    let mut ret = 0;
    while x > 0 {
        ret += multiplier * (x % 2);
        x >>= 1;
        multiplier *= 3;
    }
    ret
}


fn compute_trit_mappings(c: usize)  -> (Vec<u16>, Vec<u16>) {
    let pow3 = 3usize.pow((c+1) as u32);
    
    let mut trits = vec![0u8; c + 1];
    let mut bits_mapping = vec![0u16; 1 << (c + 1)];
    let mut trits_mapping = vec![0u16; pow3];
    
    let mut i = 0;
    loop {
        let mut bin_value = 0;
        let mut j = c;
        let mut flag = true;
        let mut bad_offset = 1u16;
        loop {
            if flag {
                bad_offset *= 3;
            }
            bin_value *= 2;
            if trits[j] == 2 {
                flag = false;
            } else {
                bin_value += trits[j] as usize;
            }

            if j == 0 {break}
            j -= 1;
        }
        if flag {
            bits_mapping[bin_value] = i as u16;
        } else {
            trits_mapping[i] = pow3 as u16 / bad_offset;
        }

        i += 1;
        if i == pow3 {
            break;
        }
        // add 1 to trits
        // this would go out of bounds for (2, 2, 2, 2, ..., 2) but this never happens because we leave
        // the main cycle before this
        let mut j = 0;
        loop {
            if trits[j] < 2 {
                trits[j] += 1;
                break;
            } else {
                trits[j] = 0;
                j += 1;
            }
        }
    }

    (bits_mapping, trits_mapping)
}


pub fn alt_extend(table: &[F128], dims: usize, c: usize) -> Vec<F128> {
    assert!(table.len() == 1 << dims);
    assert!(c < dims);
    let mut target = vec![MaybeUninit::uninit(); 3usize.pow((c + 1) as u32) << (dims - c - 1)];
    unsafe {
        _alt_extend(table, &mut target);
        transmute(target)
    }
}

unsafe fn _alt_extend(table: &[F128], target: &mut[MaybeUninit<F128>]) {
    if target.len() == table.len() {
        target.copy_from_slice(transmute(table));
        return
    }
    let a = table.len() / 2;
    let b = target.len() / 3;
    let (t0, t1) = table.split_at(a);
    let (s0, tmp) = target.split_at_mut(b);
    let (s1, s2) = tmp.split_at_mut(b);

    _alt_extend(t0, s0);
    _alt_extend(t1, s1);

    s0.par_iter().zip(s1.par_iter()).zip(s2.par_iter_mut())
        .map(|((x, y), z)| *z = MaybeUninit::new(x.assume_init() + y.assume_init())).count();
}

pub fn alt_extend_2(table: &[F128], dims: usize, c: usize) -> Vec<F128> {
    assert!(table.len() == 1 << dims);
    assert!(c < dims);
    let pow3 = 3usize.pow((c + 1) as u32);
    assert!(pow3 < (u16::MAX) as usize, "This is too large anyway ;)");
    let pow2 = 2usize.pow((dims - c - 1) as u32);
    let mut ret = vec![MaybeUninit::<F128>::uninit(); pow3 << (dims - c - 1)];
    for i in 0..pow3 {
        let mut head = i;
        let mut offset_3 = 1;
        let mut counter = 0;
        let mut bitmask = 0;
        let mut trit;     
        loop {
            trit = head % 3;
            head = head / 3;
            if trit == 2 {
                // Compute 3 slices: [i * pow2, (i+1) * pow]
                // [(i - offset) * pow, (i - offset + 1) * pow]
                // [(i - 2*offset) * pow, (i - 2*offset + 1) * pow]
                let (t1, t2) = ret.split_at_mut(i * pow2);
                let a = &t1[(i - offset_3) * pow2 .. (i - offset_3 + 1) * pow2];
                let b = &t1[(i - 2 * offset_3) * pow2 .. (i - 2 * offset_3 + 1) * pow2];
                let c = &mut t2[.. pow2];
                c.par_iter_mut().zip(a.par_iter().zip(b.par_iter()))
                    .map(|(c, (a, b))| {
                        unsafe{*c = MaybeUninit::new(a.assume_init() + b.assume_init())}
                    })
                    .count();
                break;
            }
            bitmask += trit << counter;
            offset_3 *= 3;
            counter += 1;

            if head == 0 {
                ret[i * pow2 .. i * pow2 + pow2].copy_from_slice(
                    unsafe{transmute(&table[bitmask * pow2 .. bitmask * pow2 + pow2])}
                );
                break;
            }

        }

    }
    unsafe {transmute(ret)}

}

/// Makes table 3^{c+1} * 2^{dims - c - 1}
pub fn extend_table(table: &[F128], dims: usize, c: usize) -> Vec<F128> {
    // TODO: suggest adding parallelization only on 2^k layer, as this algo for extension doesn't work well in parallel
    
    // let label0 = Instant::now();

    assert!(table.len() == 1 << dims);
    assert!(c < dims);
    let pow3 = 3usize.pow((c + 1) as u32);
    assert!(pow3 < (u16::MAX) as usize, "This is too large anyway ;)");
    let pow2 = 2usize.pow((dims - c - 1) as u32);

    let (bits_mapping, trits_mapping) = compute_trit_mappings(c);

    // let label1 = Instant::now();

    let mut ret = vec![MaybeUninit::<F128>::uninit(); pow3 * pow2];
    for i in 0..pow2 {
        let t = i * pow3;
        let mut s = i << (c + 1);
        for j in 0.. (1 << (c+1)) {
            let k = bits_mapping[j] as usize + t;
            ret[k] = MaybeUninit::new(table[s]);
            s += 1;
        }
    }

    // let label2 = Instant::now();

    let mut task_size = pow3;
    let mut d = 0;
    while (task_size < pow3 * pow2) && (task_size < 500) {
        task_size <<= 1;
        d += 1;
    }

    parallelize(|chunk, i_offset| {
        for q in 0 .. 1 << d {
            for j in 0..pow3 {
                let offset = trits_mapping[j];
                //let i = (q as usize) + (i_offset >> d);
                // actual index: j + i * pow3
                let idx = (q as usize) * pow3 + j as usize;
                unsafe {
                chunk[idx] = 
                    transmute::<u128, MaybeUninit<F128>>(
                    chunk[idx - offset as usize].assume_init().raw ^
                    (chunk[idx - 2 * offset as usize].assume_init().raw & (0u128.wrapping_sub((offset % 2) as u128)))
                    );
                }
            };
        }    
    },
    &mut ret,
    task_size,
    );

    // let label3 = Instant::now();

    // println!(
    //     "Extend table timings:\nCompute trit mappings: {} mcs\nInitialize table: {} mcs\nExtend table: {} mcs",
    //     (label1 - label0).as_micros(),
    //     (label2 - label1).as_micros(),
    //     (label3 - label2).as_micros(),
    // );

    unsafe{ transmute(ret) }
}

// /// Splits v by bits, and computes 128 vectors sum_j bit(i, v[j])w[j]
// /// Equivalently, treats first vector as 128 x l matrix, and second as l x 128 matrix,
// /// and computes 128x128 product.
// fn by_coord_product_naive(v: &[F128], w: &[F128]) -> Vec<F128> {
//     assert!(v.len() == w.len());
//     let mut ret = vec![F128::zero(); 128];
//     for j in 0..v.len() {
//         let bits = u128_to_bits(v[j].raw());
//         for k in 0..128 {
//             if bits[k] {ret[k] += w[j]}
//         }
//     }
//     ret
// }

// #[unroll::unroll_for_loops]
// unsafe fn by_coord_product_nobits(v: &[F128], w: &[F128]) -> Vec<F128> {
//     assert!(v.len() == w.len());
//     let mut ret = vec![F128::zero(); 128];
//     for j in 0..v.len() {
//         let bytes: [u8; 16] = transmute::<F128, [u8; 16]>(v[j]);
//         for k in 0..16 {
//             for s in 0..8 {
//                 if (bytes[k] >> s) % 2 != 0 {ret[k] += w[j]}
//             }
//         }
//     }
//     ret
// }

// #[unroll::unroll_for_loops]
// unsafe fn by_coord_product_nobranch(v: &[F128], w: &[F128]) -> Vec<F128> {
//     assert!(v.len() == w.len());
//     let mut ret : Vec<u128> = vec![0; 128];
//     let mut ret = transmute::<_, Vec<__m128i>>(ret);
//     for j in 0..v.len() {
//         let bytes: [u8; 16] = transmute::<F128, [u8; 16]>(v[j]);
//         for k in 0..16 {
//             let byte = bytes[k];
//             for s in 0..8 {
//                 let control = 0u128.wrapping_sub(((byte >> s) % 2) as u128);
//                 ret[k] = _mm_xor_si128(ret[k], _mm_and_si128(transmute::<F128, __m128i>(w[j]), transmute(control)));
//             }
//         }
//     }
//     transmute(ret)
// }

// fn _by_coord_product_4_ru (v: &[F128], w: &[F128]) -> Vec<F128> {
//     let mut target = vec![F128::zero(); 128];
//     by_coord_product_4_ru(v, w, &mut target.iter_mut().collect_vec());
//     target
// }

// //#[unroll::unroll_for_loops]
// fn by_coord_product_4_ru(v: &[F128], w: &[F128], target: &mut [&mut F128]) {
// unsafe{
//     let target = transmute::<_, &mut [&mut u128]>(target);
    
//     let mut j = 0;
//     loop {
//         let chunk = transmute::<&[F128], &[[u8; 16]]>(&v[j .. j + 16]);

//         let mut xortable = [MaybeUninit::<u128>::uninit(); 16];        
//         xortable[0b0000] = transmute(0u128);
//         xortable[0b0001] = transmute(w[j]);
//         xortable[0b0010] = transmute(w[j + 1]);
//         xortable[0b0100] = transmute(w[j + 2]);
//         xortable[0b1000] = transmute(w[j + 3]);
//         xortable[0b0011] = transmute(xortable[0b0001].assume_init() ^ xortable[0b0010].assume_init());
//         xortable[0b0101] = transmute(xortable[0b0100].assume_init() ^ xortable[0b0001].assume_init());
//         xortable[0b0110] = transmute(xortable[0b0100].assume_init() ^ xortable[0b0010].assume_init());
//         xortable[0b0111] = transmute(xortable[0b0101].assume_init() ^ xortable[0b0010].assume_init());
//         xortable[0b1001] = transmute(xortable[0b1000].assume_init() ^ xortable[0b0001].assume_init());
//         xortable[0b1010] = transmute(xortable[0b1000].assume_init() ^ xortable[0b0010].assume_init());
//         xortable[0b1011] = transmute(xortable[0b1000].assume_init() ^ xortable[0b0011].assume_init());
//         xortable[0b1100] = transmute(xortable[0b1000].assume_init() ^ xortable[0b0100].assume_init());
//         xortable[0b1101] = transmute(xortable[0b1000].assume_init() ^ xortable[0b0101].assume_init());
//         xortable[0b1110] = transmute(xortable[0b1000].assume_init() ^ xortable[0b0110].assume_init());
//         xortable[0b1111] = transmute(xortable[0b1000].assume_init() ^ xortable[0b0111].assume_init());
//         let xortable : [u128; 16] = transmute(xortable);

//         let mut xortable2 = [MaybeUninit::<u128>::uninit(); 16];        
//         xortable2[0b0000] = transmute(0u128);
//         xortable2[0b0001] = transmute(w[j + 4]);
//         xortable2[0b0010] = transmute(w[j + 5]);
//         xortable2[0b0100] = transmute(w[j + 6]);
//         xortable2[0b1000] = transmute(w[j + 7]);
//         xortable2[0b0011] = transmute(xortable2[0b0001].assume_init() ^ xortable2[0b0010].assume_init());
//         xortable2[0b0101] = transmute(xortable2[0b0100].assume_init() ^ xortable2[0b0001].assume_init());
//         xortable2[0b0110] = transmute(xortable2[0b0100].assume_init() ^ xortable2[0b0010].assume_init());
//         xortable2[0b0111] = transmute(xortable2[0b0101].assume_init() ^ xortable2[0b0010].assume_init());
//         xortable2[0b1001] = transmute(xortable2[0b1000].assume_init() ^ xortable2[0b0001].assume_init());
//         xortable2[0b1010] = transmute(xortable2[0b1000].assume_init() ^ xortable2[0b0010].assume_init());
//         xortable2[0b1011] = transmute(xortable2[0b1000].assume_init() ^ xortable2[0b0011].assume_init());
//         xortable2[0b1100] = transmute(xortable2[0b1000].assume_init() ^ xortable2[0b0100].assume_init());
//         xortable2[0b1101] = transmute(xortable2[0b1000].assume_init() ^ xortable2[0b0101].assume_init());
//         xortable2[0b1110] = transmute(xortable2[0b1000].assume_init() ^ xortable2[0b0110].assume_init());
//         xortable2[0b1111] = transmute(xortable2[0b1000].assume_init() ^ xortable2[0b0111].assume_init());
//         let xortable2 : [u128; 16] = transmute(xortable2);

//         let mut xortable3 = [MaybeUninit::<u128>::uninit(); 16];        
//         xortable3[0b0000] = transmute(0u128);
//         xortable3[0b0001] = transmute(w[j + 8]);
//         xortable3[0b0010] = transmute(w[j + 9]);
//         xortable3[0b0100] = transmute(w[j + 10]);
//         xortable3[0b1000] = transmute(w[j + 11]);
//         xortable3[0b0011] = transmute(xortable3[0b0001].assume_init() ^ xortable3[0b0010].assume_init());
//         xortable3[0b0101] = transmute(xortable3[0b0100].assume_init() ^ xortable3[0b0001].assume_init());
//         xortable3[0b0110] = transmute(xortable3[0b0100].assume_init() ^ xortable3[0b0010].assume_init());
//         xortable3[0b0111] = transmute(xortable3[0b0101].assume_init() ^ xortable3[0b0010].assume_init());
//         xortable3[0b1001] = transmute(xortable3[0b1000].assume_init() ^ xortable3[0b0001].assume_init());
//         xortable3[0b1010] = transmute(xortable3[0b1000].assume_init() ^ xortable3[0b0010].assume_init());
//         xortable3[0b1011] = transmute(xortable3[0b1000].assume_init() ^ xortable3[0b0011].assume_init());
//         xortable3[0b1100] = transmute(xortable3[0b1000].assume_init() ^ xortable3[0b0100].assume_init());
//         xortable3[0b1101] = transmute(xortable3[0b1000].assume_init() ^ xortable3[0b0101].assume_init());
//         xortable3[0b1110] = transmute(xortable3[0b1000].assume_init() ^ xortable3[0b0110].assume_init());
//         xortable3[0b1111] = transmute(xortable3[0b1000].assume_init() ^ xortable3[0b0111].assume_init());
//         let xortable3 : [u128; 16] = transmute(xortable3);

//         let mut xortable4 = [MaybeUninit::<u128>::uninit(); 16];        
//         xortable4[0b0000] = transmute(0u128);
//         xortable4[0b0001] = transmute(w[j + 12]);
//         xortable4[0b0010] = transmute(w[j + 13]);
//         xortable4[0b0100] = transmute(w[j + 14]);
//         xortable4[0b1000] = transmute(w[j + 15]);
//         xortable4[0b0011] = transmute(xortable4[0b0001].assume_init() ^ xortable4[0b0010].assume_init());
//         xortable4[0b0101] = transmute(xortable4[0b0100].assume_init() ^ xortable4[0b0001].assume_init());
//         xortable4[0b0110] = transmute(xortable4[0b0100].assume_init() ^ xortable4[0b0010].assume_init());
//         xortable4[0b0111] = transmute(xortable4[0b0101].assume_init() ^ xortable4[0b0010].assume_init());
//         xortable4[0b1001] = transmute(xortable4[0b1000].assume_init() ^ xortable4[0b0001].assume_init());
//         xortable4[0b1010] = transmute(xortable4[0b1000].assume_init() ^ xortable4[0b0010].assume_init());
//         xortable4[0b1011] = transmute(xortable4[0b1000].assume_init() ^ xortable4[0b0011].assume_init());
//         xortable4[0b1100] = transmute(xortable4[0b1000].assume_init() ^ xortable4[0b0100].assume_init());
//         xortable4[0b1101] = transmute(xortable4[0b1000].assume_init() ^ xortable4[0b0101].assume_init());
//         xortable4[0b1110] = transmute(xortable4[0b1000].assume_init() ^ xortable4[0b0110].assume_init());
//         xortable4[0b1111] = transmute(xortable4[0b1000].assume_init() ^ xortable4[0b0111].assume_init());
//         let xortable4 : [u128; 16] = transmute(xortable4);

//         for k in 0..16 {
//             let mut x = transmute::<_, __m128i>(
//                 [chunk[15][k], chunk[14][k], chunk[13][k], chunk[12][k], chunk[11][k], chunk[10][k], chunk[9][k], chunk[8][k],
//                      chunk[7][k], chunk[6][k], chunk[5][k], chunk[4][k], chunk[3][k], chunk[2][k], chunk[1][k], chunk[0][k],]
//             );
//             for s in 0..8 {
//                 let addr = _mm_movemask_epi8(x);
//                 *target[k] ^= xortable[(addr & 15) as usize];
//                 *target[k] ^= xortable2[((addr >> 4) & 15) as usize];
//                 *target[k] ^= xortable3[((addr >> 8) & 15) as usize];
//                 *target[k] ^= xortable4[(addr >> 12) as usize];
                
//                 x = _mm_slli_epi64(x,1);
//             }
//         };
        
//         j += 16;
//         if j >= v.len() {
//             break;
//         }
//     }
// }
// }

/// This function takes a F128-valued polynomial P, and computes restrictions of
/// its F2-valued coordinates P_i = pi_i(P) - i.e. it returns 128 polynomials
/// P_i(r0, ..., r_{j-1}, x_j, ..., x_{n-1})
pub fn restrict(mut poly: Vec<F128>, coords: &[F128], dims: usize) -> Vec<Vec<F128>> {
    // TODO: this (and matrixmult) should be optimized using hardware instructions
    assert!(poly.len() == 1 << dims);
    assert!(coords.len() <= dims);

    let chunk_size = 1 << coords.len();
    let num_chunks = 1 << (dims - coords.len());
    let eq = eq_poly(coords);
    let poly : &mut[F128] = &mut poly;

    let mut ret = vec![vec![F128::zero(); num_chunks]; 128];

    for i in 0..num_chunks { // This external cycle can be potentially parallelized
        let chunk = &mut poly[i * chunk_size .. i * chunk_size + chunk_size];
        // this is matrix 128 x chunk_size, which we multiply by eq_poly, treated as chunk_size x 128 matrix
        for j in 0..chunk_size {
            let bits = u128_to_bits(chunk[j].raw());
            for k in 0..128 {
                if bits[k] {ret[k][i] += eq[j]}
            }
        }
    }

    ret
}

pub struct AndcheckProver {
    pt: Vec<F128>,
    p: Option<Vec<F128>>,
    q: Option<Vec<F128>>,

    p_q_ext: Option<Vec<F128>>, // Table of evaluations on 3^{c+1-round} x 2^{n-c-1}

    p_coords: Option<Vec<Vec<F128>>>,
    q_coords: Option<Vec<Vec<F128>>>,

    c: usize, // PHASE SWITCH, round < c => PHASE 1.
    evaluation_claim: F128,
    challenges: Vec<F128>,
}

pub struct RoundResponse {
    pub values: Vec<F128>,
}

/// This struct holds evaluations of p and q in inverse Frobenius orbit of a challenge point.
pub struct FinalClaim {
    pub p_evs: Vec<F128>,
    pub q_evs: Vec<F128>,
}

impl FinalClaim {
    /// The function that computes evaluation of (P & Q) in a challenge point 
    /// through evaluations of P, Q in inverse Frobenius orbit.
    pub fn apply_algebraic_combinator(&self) -> F128 {
        let mut ret = F128::zero();
        let p_twists : Vec<_> = self.p_evs.iter().enumerate().map(|(i, x)|x.frob(i as i32)).collect();
        let q_twists : Vec<_> = self.q_evs.iter().enumerate().map(|(i, x)|x.frob(i as i32)).collect();
        for i in 0..128 {
            ret += F128::basis(i) * pi(i, &p_twists) * pi(i, &q_twists);
        }
        ret
    } 
}


impl AndcheckProver {
    pub fn new(pt: Vec<F128>, p: Vec<F128>, q: Vec<F128>, evaluation_claim: F128, phase_switch: usize, check_correct: bool) -> Self {
        assert!(1 << pt.len() == p.len());
        assert!(1 << pt.len() == q.len());
        assert!(phase_switch < pt.len());
        if check_correct {
            assert!(
                p.iter().zip_eq(q.iter()).zip_eq(eq_poly(&pt).iter()).fold(F128::zero(), |acc, ((&p, &q), &e)| {acc + (p & q) * e})
                ==
                evaluation_claim
            )
        }

        // Represent values in (0, 1, \infty)^{c+1} (0, 1)^{n-c-1}
        
        let start = Instant::now();
        let p_ext = extend_table(&p, pt.len(), phase_switch);
        let q_ext = extend_table(&q, pt.len(), phase_switch);

        let after_ext = Instant::now();

        let p_q_ext = p_ext.iter().zip_eq(q_ext.iter()).map(|(a, b)| *a & *b).collect();

        let end = Instant::now();

        println!("AndcheckProver::new timings\nExtensions: {} ms\nP&Q evals: {} ms\n", (after_ext - start).as_millis(), (end-after_ext).as_millis());

        Self{
            pt,
            p: Some(p),
            q: Some(q),
            p_q_ext: Some(p_q_ext),
            p_coords: None,
            q_coords: None,
            evaluation_claim,
            c: phase_switch,
            challenges: vec![]
        }
    }

    pub fn num_vars(&self) -> usize {
        self.pt.len()
    }

    pub fn curr_round(&self) -> usize {
        self.challenges.len()
    }

    pub fn round(&mut self, round_challenge: F128) -> RoundResponse {
        let round = self.curr_round();
        let num_vars = self.num_vars();
        let c = self.c;
        assert!(round < num_vars, "Protocol has already finished.");
        let curr_phase_1 = round <= c;

        let pt = &self.pt;

        let pt_l = &pt[..round];
        let pt_g = &pt[(round + 1)..];
        let pt_r = pt[round];

        let ret;

        if curr_phase_1 {
            // PHASE 1:
            let p_q_ext = self.p_q_ext.as_mut().unwrap();

            let eq_evs = eq_poly(&pt_g); // eq(x, pt_{>})
            let mut poly_deg_2 = vec![F128::zero(); 3]; //Evaluations in 0, 1 and \infty
            let phase1_dims = c - round;
            let pow3 = 3usize.pow(phase1_dims as u32);
            
            for i in 0..(1 << num_vars - c - 1) {
                for j in 0..(1 << phase1_dims) {
                    let index = (i << phase1_dims) + j;
                    let trindex = i * pow3 + bits_to_trits(j);
                    let multiplier = eq_evs[index];
                    poly_deg_2.iter_mut()
                        .zip(p_q_ext[3*trindex .. 3*trindex + 3].iter())
                        .map(|(a, b)| *a += *b * multiplier).count();
                }
            }

            // Cast poly to coefficient form
            // For f(x) = a + bx + cx^2
            // f(0) = a
            // f(\infty) = c
            // f(1) = a+b+c
            // => b = f(1) + f(0) + f(\infty)

            let tmp = poly_deg_2[0];
            poly_deg_2[1] += tmp;
            let tmp = poly_deg_2[2];
            poly_deg_2[1] += tmp;

            let eq_y_multiplier = eq_ev(&self.challenges, &pt_l);

            poly_deg_2.iter_mut().map(|c| *c *= eq_y_multiplier).count();

            // eq(t, pt_r) = t pt_r + (1 - t) (1 - pt_r) = (1+pt_r) + t
            let eq_t = vec![pt_r + F128::one(), F128::one()];

            let poly_final = vec![
                eq_t[0] * poly_deg_2[0],
                eq_t[0] * poly_deg_2[1] + eq_t[1] * poly_deg_2[0],
                eq_t[0] * poly_deg_2[2] + eq_t[1] * poly_deg_2[1],
                eq_t[1] * poly_deg_2[2],
            ];

            let r2 = round_challenge * round_challenge;
            let r3 = round_challenge * r2;

            assert!(poly_final[1] + poly_final[2] + poly_final[3] == self.evaluation_claim);

            self.evaluation_claim = poly_final[0] + poly_final[1] * round_challenge + poly_final[2] * r2 + poly_final[3] * r3;
            self.challenges.push(round_challenge);
            self.p_q_ext = Some((0..(p_q_ext.len()/3)).map(|i| {
                let chunk = &p_q_ext[3 * i .. 3 * i + 3];
                chunk[0] + (chunk[0] + chunk[1] + chunk[2]) * round_challenge + chunk[2] * r2
            }).collect());

            ret = RoundResponse{values: poly_final};
        } else {
            let eq_evs = eq_poly(&pt_g);
            let half = eq_evs.len();

            let p_coords = self.p_coords.as_mut().unwrap();
            let q_coords = self.q_coords.as_mut().unwrap();


            let mut poly_deg_2 = vec![F128::zero(); 3]; // Actually, value in 1 is not necessary.

            for i in 0..half {
                // This data layout sucks :(
                poly_deg_2[0] += eq_evs[i] * ((0..128).map(|j| {
                    F128::basis(j) * p_coords[j][2 * i] * q_coords[j][2 * i]
                }).fold(F128::zero(), |a, b| a + b));
                poly_deg_2[1] += eq_evs[i] * ((0..128).map(|j| {
                    F128::basis(j) * p_coords[j][2 * i + 1] * q_coords[j][2 * i + 1]
                }).fold(F128::zero(), |a, b| a + b));
                poly_deg_2[2] += eq_evs[i] * ((0..128).map(|j| {
                    F128::basis(j) * (p_coords[j][2 * i] + p_coords[j][2 * i + 1]) * (q_coords[j][2 * i] + q_coords[j][2 * i + 1])
                }).fold(F128::zero(), |a, b| a + b));
            }

            let eq_y_multiplier = eq_ev(&self.challenges, &pt_l);
            poly_deg_2.iter_mut().map(|c| *c *= eq_y_multiplier).count();

            // Cast poly to coefficient form
            // For f(x) = a + bx + cx^2
            // f(0) = a
            // f(\infty) = c
            // f(1) = a+b+c
            // => b = f(1) + f(0) + f(\infty)

            let tmp = poly_deg_2[0];
            poly_deg_2[1] += tmp;
            let tmp = poly_deg_2[2];
            poly_deg_2[1] += tmp;

            let eq_t = vec![pt_r + F128::one(), F128::one()];

            let poly_final = vec![
                eq_t[0] * poly_deg_2[0],
                eq_t[0] * poly_deg_2[1] + eq_t[1] * poly_deg_2[0],
                eq_t[0] * poly_deg_2[2] + eq_t[1] * poly_deg_2[1],
                eq_t[1] * poly_deg_2[2],
            ];

            let r2 = round_challenge * round_challenge;
            let r3 = round_challenge * r2;

            assert!(poly_final[1] + poly_final[2] + poly_final[3] == self.evaluation_claim);

            self.evaluation_claim = poly_final[0] + poly_final[1] * round_challenge + poly_final[2] * r2 + poly_final[3] * r3;
            self.challenges.push(round_challenge);

            // External iteration can be parallelized for early-ish rounds.
            p_coords.iter_mut().map(|arr| {
                for j in 0..half {
                    arr[j] = arr[2 * j] + (arr[2 * j + 1] + arr[2 * j]) * round_challenge
                };
                arr.truncate(half);
            }).count();


            q_coords.iter_mut().map(|arr| {
                for j in 0..half {
                    arr[j] = arr[2 * j] + (arr[2 * j + 1] + arr[2 * j]) * round_challenge
                };
                arr.truncate(half);
            }).count();

            ret = RoundResponse{values: poly_final};
        };

        // SWITCH PHASES
        // we switch phases at the end of the function to ensure that we do the switch even if c = num_vars-1
        // because our finish() function expects to find restricted P, Q anyway

        if self.curr_round() == c + 1 { // Note that we are in the next round now.
            let _ = self.p_q_ext.take(); // it is useless now
            let p = self.p.take().unwrap(); // and these now will turn into p_i-s and q_is
            let q = self.q.take().unwrap();
            self.p_coords = Some(restrict(p, &self.challenges, num_vars));
            self.q_coords = Some(restrict(q, &self.challenges, num_vars));
            // TODO: we can avoid recomputing eq-s throughout the protocol in multiple places, including restrict
        }

        ret
    }


    pub fn finish(&self) -> FinalClaim {
        assert!(self.curr_round() == self.num_vars(), "Protocol is not finished.");


        let mut inverse_orbit = vec![];
        let mut pt = self.challenges.clone();
        for _ in 0..128 {
            pt.iter_mut().map(|x| *x *= *x).count();
            inverse_orbit.push(pt.clone());
        }
        inverse_orbit.reverse();

        let mut p_i_evs = self.p_coords.as_ref().unwrap().iter().map(|a| {
            assert!(a.len() == 1);
            a[0]
        }).collect_vec();

        let mut q_i_evs = self.q_coords.as_ref().unwrap().iter().map(|a| {
            assert!(a.len() == 1);
            a[0]
        }).collect_vec();

        // We have got P_i(r).
        // P_i(Fr^j(r)) = Fr^j(P_i(r))


        let mut p_evs = vec![];
        let mut q_evs = vec![];

        // We square first and then compute evals so after inversion we get reverse Frobenius orbit
        // So we have smth like r^2, r^{2^2}, ..., r^{2^128}=r --> reverse
        // r, r^{2^{127}}, r^{2^{126}}, ... 
        for _ in 0..128 {
            p_i_evs.iter_mut().map(|x| *x *= *x).count();
            q_i_evs.iter_mut().map(|x| *x *= *x).count();
            p_evs.push(
                (0..128).map(|i| {
                    F128::basis(i) * p_i_evs[i]
                }).fold(F128::zero(), |a, b| a + b)
            );
            q_evs.push(
                (0..128).map(|i| {
                    F128::basis(i) * q_i_evs[i]
                }).fold(F128::zero(), |a, b| a + b)
            );
        }

        p_evs.reverse();
        q_evs.reverse();

        FinalClaim { p_evs, q_evs }
    }
}



#[cfg(test)]
mod tests {
    use std::{iter::repeat_with, time::Instant};

    use itertools::Itertools;
    use num_traits::Zero;
    use rand::rngs::OsRng;

    use super::*;

    #[test]
    fn test_eq_ev() {
        let rng = &mut OsRng;
        let num_vars = 5;

        let x : Vec<_> = repeat_with(|| F128::rand(rng)).take(num_vars).collect();
        let y : Vec<_> = repeat_with(|| F128::rand(rng)).take(num_vars).collect();
        

        assert!(eq_ev(&x, &y) == evaluate(&eq_poly(&x), &y));
    }

    #[test]
    /// WARNING: THIS TEST WILL DO NOTHING AFTER WE SWITCH TO MAYBEUNINITS.
    /// NOW IT CHECKS INTEGRITY
    fn extend_table_collisions() {
        let rng = &mut OsRng;
        for i in 0..7 {
            let table = repeat_with(|| F128::rand(rng)).take(1 << i).collect_vec();
            for c in 0..i {
                let ret = extend_table(&table, i, c);
                assert!(ret.len() == 3usize.pow((c+1) as u32)*2usize.pow((i-c-1) as u32));
            }
        }
    }

    #[test]

    fn trits_test() {
        let c = 2;
        println!("{:?}", compute_trit_mappings(c));
    }

    #[test]
    fn twists_as_expected() {
        let rng = &mut OsRng;
        let num_vars = 5;
        let pt : Vec<_> = repeat_with(|| F128::rand(rng)).take(num_vars).collect();
        let p : Vec<_> = repeat_with(|| F128::rand(rng)).take(1 << num_vars).collect();

        for i in 0..128 {
            let inv_twisted_pt = pt.iter().map(|x| x.frob(- (i as i32))).collect_vec();
            let ev = evaluate(&p, &inv_twisted_pt);
            let twisted_p = p.iter().map(|x|x.frob(i as i32)).collect_vec();
            assert!(ev.frob(i as i32) == evaluate(&twisted_p, &pt));
        }
    }

    #[test]

    fn restrict_as_expected() {
        let rng = &mut OsRng;
        let num_vars = 8;
        let num_vars_to_restrict = 5;
        let pt : Vec<_> = repeat_with(|| F128::rand(rng)).take(num_vars).collect();
        let poly : Vec<_> = repeat_with(|| F128::rand(rng)).take(1 << num_vars).collect();        
        
        let mut poly_unzip = vec![];
        for i in 0..128 {
            poly_unzip.push(
                poly.iter().map(|x|{
                    F128::new((x.raw() >> i) % 2 == 1)
                }).collect_vec()
            )
        }

        let answer = restrict(poly, &pt[..num_vars_to_restrict], num_vars);

        for i in 0..128 {
            assert!(evaluate(&answer[i], &pt[num_vars_to_restrict..]) == evaluate(&poly_unzip[i], &pt));
        }
    }

    #[test]
    fn extends_vs() {
        let rng = &mut OsRng;
        let mut a = F128::rand(rng);
        let mut b = F128::rand(rng);
        let mut v = vec![];
        let mut w = vec![];
        for i in 0..(1<<19) {
            v.push(a);
            w.push(b);
            a *= a;
            b *= b;
            a += F128::from_raw(3294703249);
            b += F128::from_raw(892347934);
        }

        let l0 = Instant::now();
        let a = extend_table(&v,19, 5);
        let l1 = Instant::now();
        let b = alt_extend(&v, 19, 5);
        let l2 = Instant::now();
        let c = alt_extend_2(&v, 19, 5);
        let l3 = Instant::now();

        println!("extend_table: {} ms", (l1 - l0).as_millis());
        println!("alt_extend: {} ms", (l2 - l1).as_millis());
        println!("alt_extend_2: {} ms", (l3 - l2).as_millis());

        assert_eq!(b, c);
    }

    // #[test]
    // fn bitprod() {
    //     let rng = &mut OsRng;
    //     let mut a = F128::rand(rng);
    //     let mut b = F128::rand(rng);
    //     let mut v = vec![];
    //     let mut w = vec![];
    //     for i in 0..(1 << 19) {
    //         v.push(a);
    //         w.push(b);
    //         a *= a;
    //         b *= b;
    //     }

    //     let start = Instant::now();
    //     let p = by_coord_product_naive(&v, &w);
    //     let end = Instant::now();

    //     println!("Naive prod took {} ms", (end - start).as_millis());

    //     let start = Instant::now();
    //     let q = unsafe{by_coord_product_nobits(&v, &w)};
    //     let end = Instant::now();

    //     println!("Nobits prod took {} ms", (end - start).as_millis());

    //     let start = Instant::now();
    //     let r = unsafe{by_coord_product_nobranch(&v, &w)};
    //     let end = Instant::now();

    //     println!("Nobranch prod took {} ms", (end - start).as_millis());

    //     let start = Instant::now();
    //     let s = _by_coord_product_4_ru(&v, &w);
    //     let end = Instant::now();

    //     println!("Nobranch 4 russians prod took {} ms", (end - start).as_millis());

    //     assert_eq!(p, q);
    //     assert_eq!(q, r);
    //     assert_eq!(r, s)

    // }

    #[test]
    fn verify_prover() {
        let rng = &mut OsRng;
        let num_vars = 18;

        let pt : Vec<_> = repeat_with(|| F128::rand(rng)).take(num_vars).collect();
        let p : Vec<_> = repeat_with(|| F128::rand(rng)).take(1 << num_vars).collect();
        let q : Vec<_> = repeat_with(|| F128::rand(rng)).take(1 << num_vars).collect();

        let p_zip_q : Vec<_> = p.iter().zip_eq(q.iter()).map(|(x, y)| *x & *y).collect();
        //let evaluation_claim = evaluate(&p_zip_q, &pt);
        let evaluation_claim = p_zip_q.iter().zip(eq_poly(&pt).iter()).fold(F128::zero(), |acc, (x, y)|acc + *x * *y);

        let start = Instant::now();

        let mut prover = AndcheckProver::new(pt, p, q, evaluation_claim, 5,false);

        for i in 0..num_vars {
            println!("Entering round {}, phase {}", i, if i <= 5 {1} else {2});
            let start = Instant::now();
            let round_challenge = F128::rand(rng);
            prover.round(round_challenge);
            let end = Instant::now();
            println!("Round {} elapsed time {} ms", i, (end - start).as_millis());
        }

        assert!(
            prover.finish().apply_algebraic_combinator() * eq_ev(&prover.pt, &prover.challenges)
            ==
            prover.evaluation_claim
        );

        let end = Instant::now();

        println!("Time elapsed: {}", (end - start).as_millis());
    }
}
use std::{mem::{transmute, MaybeUninit}, time::Instant};

use num_traits::{One, Pow, Zero};
use rand::{rngs::OsRng, RngCore};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};
use crate::{backend::autodetect::{v_movemask_epi8, v_slli_epi64}, field::{pi, F128}, parallelize::parallelize, utils::u128_to_bits};
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


pub fn extend_table2(table: &[F128], dims: usize, c: usize) -> Vec<F128> {
    assert!(table.len() == 1 << dims);
    assert!(c < dims);
    let mut target = vec![MaybeUninit::uninit(); 3usize.pow((c + 1) as u32) << (dims - c - 1)];
    unsafe {
        _extend_table2(table, &mut target);
        transmute(target)
    }
}

unsafe fn _extend_table2(table: &[F128], target: &mut[MaybeUninit<F128>]) {
    if target.len() == table.len() {
        target.copy_from_slice(transmute(table));
        return
    }
    let a = table.len() / 2;
    let b = target.len() / 3;
    let (t0, t1) = table.split_at(a);
    let (s0, tmp) = target.split_at_mut(b);
    let (s1, s2) = tmp.split_at_mut(b);

    _extend_table2(t0, s0);
    _extend_table2(t1, s1);

    s0.par_iter().zip(s1.par_iter()).zip(s2.par_iter_mut())
        .map(|((x, y), z)| *z = MaybeUninit::new(x.assume_init() + y.assume_init())).count();
}



/// Makes table 3^{c+1} * 2^{dims - c - 1}
pub fn extend_table(table: &[F128], dims: usize, c: usize) -> Vec<F128> {
    // TODO: suggest adding parallelization only on 2^k layer, as this algo for extension doesn't work well in parallel
    
    assert!(table.len() == 1 << dims);
    assert!(c < dims);
    let pow3 = 3usize.pow((c + 1) as u32);
    assert!(pow3 < (u16::MAX) as usize, "This is too large anyway ;)");
    let pow2 = 2usize.pow((dims - c - 1) as u32);

    let (bits_mapping, trits_mapping) = compute_trit_mappings(c);

    let mut ret = vec![MaybeUninit::uninit(); pow3 * pow2];
    for i in 0..(1 << dims) {
        let hi = i >> (c + 1);
        let lo = i ^ (hi << (c + 1));
        let j = bits_mapping[lo] as usize + hi * pow3;
        ret[j] = MaybeUninit::new(table[i]);
    }

    let mut task_size = pow3;
    let mut d = 0;
    while (task_size < pow3 * pow2) && (task_size < 500) {
        task_size <<= 1;
        d += 1;
    }

    parallelize(|chunk, i_offset| {
        for q in 0 .. 1 << d {
            for j in 0..pow3 {
                let offset = trits_mapping[j] as usize;
                //let i = (q as usize) + (i_offset >> d);
                // actual index: j + i * pow3
                let idx = (q as usize) * pow3 + j as usize;
                
                if offset != 0 {
                    unsafe{
                        chunk[idx] = MaybeUninit::new(chunk[idx - offset].assume_init() + chunk[idx - 2*offset].assume_init());// set it 0 and 1, add values
                    }
                }
            };
        }    
    },
    &mut ret,
    task_size,
    );
    unsafe{transmute::<Vec<MaybeUninit<F128>>, Vec<F128>>(ret)}
}

#[unroll::unroll_for_loops]
const fn drop_top_bit(x: usize) -> (usize, usize) {
    let mut s = 0;
    for i in 0..8 {
        let bit = (x >> i) % 2;
        s = i * bit + s * (1 - bit);
    }
    (x - (1 << s), s)
}

#[unroll::unroll_for_loops]
pub fn restrict2(poly: Vec<F128>, coords: &[F128], dims: usize) -> Vec<Vec<F128>> {
    assert!(poly.len() == 1 << dims);
    assert!(coords.len() <= dims);

    let chunk_size = (1 << coords.len());
    let num_chunks = 1 << (dims - coords.len());

    let eq = eq_poly(coords);

    assert!(eq.len() % 16 == 0, "Technical condition for now.");

    let mut eq_sums = Vec::with_capacity(256 * eq.len() / 8);

    for i in 0..eq.len()/8 {
        eq_sums.push(F128::zero());
        for j in 1..256 {
            let (sum_idx, eq_idx) = drop_top_bit(j);
            let tmp = eq[i * 8 + eq_idx] + eq_sums[i * 256 + sum_idx];
            eq_sums.push(tmp);
        }
    }

    let poly : &[F128] = & poly;

    let mut ret = vec![vec![F128::zero(); num_chunks]; 128];
    let ret_ptrs : [usize; 128] = ret.iter_mut().map(|v| unsafe{
        transmute::<*mut F128, usize>((*v).as_mut_ptr()) // This is extremely ugly.  
    }).collect_vec().try_into().unwrap();

    (0..num_chunks).into_par_iter().map(move |i| {
        for j in 0 .. eq.len() / 16 { // Step by 16 
            let v0 = &eq_sums[j * 512 .. j * 512 + 256];
            let v1 = &eq_sums[j * 512 + 256 .. j * 512 + 512];
            let bytearr = unsafe{ transmute::<&[F128], &[[u8; 16]]>(
                &poly[i * chunk_size + j * 16 .. i * chunk_size + (j + 1) * 16]
            ) };

            // Iteration over bytes
            for s in 0..16 {
                let mut t = [
                    bytearr[0][s], bytearr[1][s], bytearr[2][s], bytearr[3][s],
                    bytearr[4][s], bytearr[5][s], bytearr[6][s], bytearr[7][s],
                    bytearr[8][s], bytearr[9][s], bytearr[10][s], bytearr[11][s],
                    bytearr[12][s], bytearr[13][s], bytearr[14][s], bytearr[15][s],
                ];
 
                for u in 0..8 {
                    let bits = v_movemask_epi8(t) as u16;

                    unsafe{
                        let ret_ptrs = transmute::<[usize; 128], [*mut F128; 128]>(ret_ptrs);
                        * ret_ptrs[s*8 + 7 - u].offset(i as isize) += v0[(bits & 255) as usize];
                        * ret_ptrs[s*8 + 7 - u].offset(i as isize) += v1[((bits >> 8) & 255) as usize];
                    }
                    t = v_slli_epi64::<1>(t);
                }
            }

        }
    }
    ).count();

    ret
}

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
        let p_q_ext = p_ext.iter().zip_eq(q_ext.iter()).map(|(a, b)| *a & *b).collect();

        let end = Instant::now();

        println!("AndcheckProver::new time {} ms", (end-start).as_millis());

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
            self.p_coords = Some(restrict2(p, &self.challenges, num_vars));
            self.q_coords = Some(restrict2(q, &self.challenges, num_vars));
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
    fn verify_prover() {
        let rng = &mut OsRng;
        let num_vars = 20;

        let pt : Vec<_> = repeat_with(|| F128::rand(rng)).take(num_vars).collect();
        let p : Vec<_> = repeat_with(|| F128::rand(rng)).take(1 << num_vars).collect();
        let q : Vec<_> = repeat_with(|| F128::rand(rng)).take(1 << num_vars).collect();

        let p_zip_q : Vec<_> = p.iter().zip_eq(q.iter()).map(|(x, y)| *x & *y).collect();
        //let evaluation_claim = evaluate(&p_zip_q, &pt);
        let evaluation_claim = p_zip_q.iter().zip(eq_poly(&pt).iter()).fold(F128::zero(), |acc, (x, y)|acc + *x * *y);

        let phase_switch = 5;

        let start = Instant::now();

        let mut prover = AndcheckProver::new(pt, p, q, evaluation_claim, phase_switch,false);

        for i in 0..num_vars {
            println!("Entering round {}, phase {}", i, if i <= phase_switch {1} else {2});
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

    #[test]
    fn movemask_where_bits_go() {
        let a : [u8; 16] = [0xFF, 0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0x00];
        println!("{:#032b}", v_movemask_epi8(a));
    }
}
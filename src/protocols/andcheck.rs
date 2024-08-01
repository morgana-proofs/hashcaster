use std::{mem::{transmute, MaybeUninit}, sync::atomic::{AtomicU64, Ordering}, time::{Duration, Instant}};

use num_traits::{One, Zero};
use rayon::{iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator}, slice::{ParallelSlice, ParallelSliceMut}};
use crate::{field::{pi, F128}, protocols::utils::{compute_trit_mappings, eq_ev, eq_poly, eq_poly_sequence, extend_2_tables_legacy, extend_n_tables, restrict_legacy}};
use itertools::Itertools;


pub struct AndcheckProver {
    pt: Vec<F128>,
    p: Option<Vec<F128>>,
    q: Option<Vec<F128>>,

    p_q_ext: Option<Vec<F128>>, // Table of evaluations on 3^{c+1-round} x 2^{n-c-1}

    p_coords: Option<Vec<Vec<F128>>>,
    q_coords: Option<Vec<Vec<F128>>>,

    c: usize, // PHASE SWITCH, round < c => PHASE 1.
    pub evaluation_claim: F128,
    pub challenges: Vec<F128>,

    bits_to_trits_map: Vec<u16>,

    eq_sequence: Vec<Vec<F128>>, // Precomputed eqs of all slices pt[i..].
}

pub struct RoundResponse {
    pub values: Vec<F128>,
}

/// This struct holds evaluations of p and q in inverse Frobenius orbit of a challenge point.
pub struct AndcheckFinalClaim {
    pub p_evs: Vec<F128>,
    pub q_evs: Vec<F128>,
}


impl AndcheckFinalClaim {
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
        #[cfg(not(feature = "parallel"))]
        println!("I'm single-threaded.");
        #[cfg(feature = "parallel")]
        println!("I'm multi-threaded.");

        // Represent values in (0, 1, \infty)^{c+1} (0, 1)^{n-c-1}
        
        let (bit_mapping, trit_mapping) = compute_trit_mappings(phase_switch);

        let start = Instant::now();
        // let p_ext = extend_table(&p, pt.len(), phase_switch, &trit_mapping);
        // let q_ext = extend_table(&q, pt.len(), phase_switch, &trit_mapping);

        // let label = Instant::now();
        
        // let p_q_ext = p_ext.par_iter().zip(q_ext.par_iter()).map(|(a, b)| *a & *b).collect();
        // let p_q_ext = extend_2_tables_legacy(&p, &q, pt.len(), phase_switch, &trit_mapping);
        
        let p_q_ext = extend_n_tables(&[&p, &q], phase_switch, &trit_mapping, |[a, b]| {a & b});

        let eq_sequence = eq_poly_sequence(&pt[1..]); // We do not need whole eq, only partials. 

        let end = Instant::now();

        // println!("AndcheckProver::new time {} ms",
        //     (end-start).as_millis(),
        // );

        Self{
            pt,
            p: Some(p),
            q: Some(q),
            p_q_ext: Some(p_q_ext),
            p_coords: None,
            q_coords: None,
            evaluation_claim,
            c: phase_switch,
            challenges: vec![],
            bits_to_trits_map: bit_mapping,
            eq_sequence,
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

            let eq_evs = &self.eq_sequence[pt.len() - round - 1]; // eq(x, pt_{>})
            let phase1_dims = c - round;
            let pow3 = 3usize.pow(phase1_dims as u32);

            #[cfg(not(feature = "parallel"))]
            let mut poly_deg_2 =
            (0 .. (1 << num_vars - c - 1)).into_iter().map(|i| {
                let mut pd2_part = [F128::zero(), F128::zero(), F128::zero()];
                for j in 0..(1 << phase1_dims) {
                    let index = (i << phase1_dims) + j;
                    let offset = 3 * (i * pow3 + self.bits_to_trits_map[j] as usize);
                    let multiplier = eq_evs[index];
                    pd2_part[0] += p_q_ext[offset] * multiplier;
                    pd2_part[1] += p_q_ext[offset + 1] * multiplier;
                    pd2_part[2] += p_q_ext[offset + 2] * multiplier;
                }
                pd2_part
            }).fold([F128::zero(), F128::zero(), F128::zero()], |a, b|{
                [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
            });
   
            #[cfg(feature = "parallel")]
            let mut poly_deg_2 =
            (0 .. (1 << num_vars - c - 1)).into_par_iter().map(|i| {
                let mut pd2_part = [F128::zero(), F128::zero(), F128::zero()];
                for j in 0..(1 << phase1_dims) {
                    let index = (i << phase1_dims) + j;
                    let offset = 3 * (i * pow3 + self.bits_to_trits_map[j] as usize);
                    let multiplier = eq_evs[index];
                    pd2_part[0] += p_q_ext[offset] * multiplier;
                    pd2_part[1] += p_q_ext[offset + 1] * multiplier;
                    pd2_part[2] += p_q_ext[offset + 2] * multiplier;
                }
                pd2_part
            }).reduce(||{[F128::zero(), F128::zero(), F128::zero()]}, |a, b|{
                [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
            });


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

            #[cfg(feature = "parallel")]
            let p_q_ext_chunks = p_q_ext.par_chunks(3);

            #[cfg(not(feature = "parallel"))]
            let p_q_ext_chunks = p_q_ext.chunks(3);

            self.p_q_ext = Some(
                p_q_ext_chunks.map(|chunk| {
                    chunk[0] + (chunk[0] + chunk[1] + chunk[2]) * round_challenge + chunk[2] * r2
                }).collect()
            );

            ret = RoundResponse{values: poly_final};
        } else {
            let eq_evs = &self.eq_sequence[pt.len() - round - 1];
            let half = eq_evs.len();

            let p_coords = self.p_coords.as_mut().unwrap();
            let q_coords = self.q_coords.as_mut().unwrap();


            let poly_deg_2 : [AtomicU64; 6] = [0.into(), 0.into(), 0.into(), 0.into(), 0.into(), 0.into()];

            // For some reason, version without atomics performs almost the same *and it seems even a bit worse*
            // TODO: benchmark properly :)
            // But for phase 1, usage of atomic degrades severely degrades perf ¯\_(ツ)_/¯

            // let mut poly_deg_2 = 

            // (0..half).into_par_iter().map(|i| {                
            //     let mut pd2_part = [MaybeUninit::uninit(), MaybeUninit::uninit(), MaybeUninit::uninit()];

            //     pd2_part[0] = MaybeUninit::new((eq_evs[i] * ((0..128).map(|j| {
            //         F128::basis(j) * p_coords[j][2 * i] * q_coords[j][2 * i]
            //     }).fold(F128::zero(), |a, b| a + b))));

            //     pd2_part[1] = MaybeUninit::new(eq_evs[i] * ((0..128).map(|j| {
            //         F128::basis(j) * p_coords[j][2 * i + 1] * q_coords[j][2 * i + 1]
            //     }).fold(F128::zero(), |a, b| a + b)));

            //     pd2_part[2] = MaybeUninit::new(eq_evs[i] * ((0..128).map(|j| {
            //         F128::basis(j) * (p_coords[j][2 * i] + p_coords[j][2 * i + 1])
            //         * (q_coords[j][2 * i] + q_coords[j][2 * i + 1])
            //     }).fold(F128::zero(), |a, b| a + b)));

            //     unsafe{ transmute::<[MaybeUninit<F128>; 3], [F128; 3]>(pd2_part)}
            // }).reduce(||{[F128::zero(), F128::zero(), F128::zero()]}, |a, b|{
            //     [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
            // });

            unsafe{
                #[cfg(not(feature = "parallel"))]
                let iter = (0..half).into_iter();

                #[cfg(feature = "parallel")]
                let iter = (0..half).into_par_iter();

                iter.map(|i| {
                    let a = transmute::<F128, [u64; 2]>(eq_evs[i] * ((0..128).map(|j| {
                        F128::basis(j) * p_coords[j][2 * i] * q_coords[j][2 * i]
                    }).fold(F128::zero(), |a, b| a + b)));

                    poly_deg_2[0].fetch_xor(a[0], Ordering::Relaxed);
                    poly_deg_2[1].fetch_xor(a[1], Ordering::Relaxed);

                    let a = transmute::<F128, [u64; 2]>(eq_evs[i] * ((0..128).map(|j| {
                        F128::basis(j) * p_coords[j][2 * i + 1] * q_coords[j][2 * i + 1]
                    }).fold(F128::zero(), |a, b| a + b)));

                    poly_deg_2[2].fetch_xor(a[0], Ordering::Relaxed);
                    poly_deg_2[3].fetch_xor(a[1], Ordering::Relaxed);

                    let a = transmute::<F128, [u64; 2]>(eq_evs[i] * ((0..128).map(|j| {
                        F128::basis(j) * (p_coords[j][2 * i] + p_coords[j][2 * i + 1]) * (q_coords[j][2 * i] + q_coords[j][2 * i + 1])
                    }).fold(F128::zero(), |a, b| a + b)));

                    poly_deg_2[4].fetch_xor(a[0], Ordering::Relaxed);
                    poly_deg_2[5].fetch_xor(a[1], Ordering::Relaxed);
                }).count();
            }

            let poly_deg_2 : [u64; 6] = poly_deg_2.iter().map(|x| x.load(Ordering::Relaxed)).collect_vec().try_into().unwrap();
            let mut poly_deg_2 = unsafe{ transmute::<[u64; 6], [F128; 3]>(poly_deg_2) };

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

            #[cfg(not(feature = "parallel"))]
            let iter = p_coords.iter_mut();
            #[cfg(feature = "parallel")]
            let iter = p_coords.par_iter_mut();

            iter.map(|arr| {
                for j in 0..half {
                    arr[j] = arr[2 * j] + (arr[2 * j + 1] + arr[2 * j]) * round_challenge
                };
                arr.truncate(half);
            }).count();

            #[cfg(not(feature = "parallel"))]
            let iter = q_coords.iter_mut();
            #[cfg(feature = "parallel")]
            let iter = q_coords.par_iter_mut();

            iter.map(|arr| {
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
            self.p_coords = Some(restrict_legacy(&p, &self.challenges, num_vars));
            self.q_coords = Some(restrict_legacy(&q, &self.challenges, num_vars));
            // TODO: we can avoid recomputing eq-s throughout the protocol in multiple places, including restrict
        }

        ret
    }


    pub fn finish(&self) -> AndcheckFinalClaim {
        assert!(self.curr_round() == self.num_vars(), "Protocol is not finished.");

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

        AndcheckFinalClaim { p_evs, q_evs }
    }
}



#[cfg(test)]
mod tests {
    use std::{iter::repeat_with, time::Instant};

    use itertools::Itertools;
    use num_traits::Zero;
    use rand::rngs::OsRng;

    use crate::protocols::utils::evaluate;

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

        let answer = restrict_legacy(&poly, &pt[..num_vars_to_restrict], num_vars);

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

        println!("Total time elapsed: {}", (end - start).as_millis());
    }

}

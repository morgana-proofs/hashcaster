use std::{sync::atomic::{AtomicU64, Ordering}};

use bytemuck::cast;
use itertools::Itertools;
use num_traits::Zero;
use rayon::iter::{ParallelIterator, IntoParallelIterator};

use crate::{field::F128, traits::{CompressedPoly, SumcheckObject}, utils::log2_exact};


/// A very simple sumcheck, only does product of 2 polynomials. It is used as main component for lincheck.
pub struct Prodcheck {
    pub p_polys: Vec<Vec<F128>>,
    pub q_polys: Vec<Vec<F128>>,
    pub claim: F128,
    challenges: Vec<F128>,
    num_vars: usize,

    cached_round_msg: Option<CompressedPoly>,
    // cached_p_bind: Option<Vec<F128>>,
    // cached_q_bind: Option<Vec<F128>>,

    rev_order: bool, 
}

impl Prodcheck {
    pub fn new(
        p_polys: Vec<Vec<F128>>,
        q_polys: Vec<Vec<F128>>,
        initial_claim: F128,
        check_init_claim: bool,
        in_reverse_order: bool,
    ) -> Self {

        let num_vars = log2_exact(p_polys[0].len());
        assert!(p_polys.len() == q_polys.len());
        for i in 0..p_polys.len() {
            assert!(p_polys[i].len() == 1 << num_vars);
            assert!(q_polys[i].len() == 1 << num_vars);
        }

        let l = p_polys.len();

        if check_init_claim {
            let mut expected_claim = F128::zero();
            for i in 0 .. l {
                for j in 0 .. 1 << num_vars {
                    expected_claim += p_polys[i][j] * q_polys[i][j]
                }
            } 


            assert!(initial_claim == expected_claim);
        }

        Self {
            p_polys,
            q_polys,
            claim: initial_claim,
            challenges: vec![],
            num_vars,
            // cached_p_bind: None,
            // cached_q_bind: None,
            cached_round_msg: None,
            rev_order: in_reverse_order,
        }
    }
}

impl SumcheckObject for Prodcheck {

    fn is_reverse_order(&self) -> bool {
        self.rev_order
    }

    fn bind(&mut self, challenge: F128) {
        if self.rev_order {
            panic!("Unsupported order.");
        }
        assert!(self.p_polys[0].len() > 1, "The protocol has already ended.");
        let half = self.p_polys[0].len() / 2;
        let l = self.p_polys.len();

        let round_poly = self.round_msg().coeffs(self.claim);
        // Decompressed round polynomial in a coefficient form.
        self.claim = round_poly[0] + challenge * round_poly[1] + challenge * challenge * round_poly[2];
        self.challenges.push(challenge);

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..l {
                for j in 0..half {
                    self.p_polys[i][j] = self.p_polys[i][2 * j] + (self.p_polys[i][2 * j + 1] + self.p_polys[i][2 * j]) * challenge;
                    self.q_polys[i][j] = self.q_polys[i][2 * j] + (self.q_polys[i][2 * j + 1] + self.q_polys[i][2 * j]) * challenge;
                    self.p_polys[i].truncate(half);
                    self.q_polys[i].truncate(half);
                }
            }
        }

        #[cfg(feature = "parallel")]
        {
            let mut p_new = vec![];
            let mut q_new = vec![];

            for i in 0..l {
                p_new.push((0..half).into_par_iter().map(|j| {
                    self.p_polys[i][2 * j] + (self.p_polys[i][2 * j + 1] + self.p_polys[i][2 * j]) * challenge
                }).collect());
                q_new.push((0..half).into_par_iter().map(|j| {
                    self.q_polys[i][2 * j] + (self.q_polys[i][2 * j + 1] + self.q_polys[i][2 * j]) * challenge
                }).collect());
            }
            
            self.p_polys = p_new;
            self.q_polys = q_new;
            
        }

        self.cached_round_msg = None;
    }

    fn round_msg(&mut self) -> CompressedPoly {

        assert!(self.p_polys[0].len() > 1, "The protocol has already ended.");
        let half = self.p_polys[0].len() / 2;

        if self.cached_round_msg.is_some() {
            return self.cached_round_msg.as_ref().unwrap().clone()
        }

        if self.rev_order {
            panic!("Unsupported order.");
        }

        let l = self.p_polys.len();

        #[cfg(not(feature = "parallel"))]
        let iter = (0 .. half).into_iter();

        #[cfg(feature = "parallel")]
        let iter = (0 .. half).into_par_iter();

        let iter = 
        iter.map(|i|{
            let mut pq_zero = self.p_polys[0][2 * i] * self.q_polys[0][2 * i];
            for j in 1..l {
                pq_zero += self.p_polys[j][2 * i] * self.q_polys[j][2 * i]
            }

            let mut pq_one = self.p_polys[0][2 * i + 1] * self.q_polys[0][2 * i + 1];
            for j in 1..l {
                pq_one += self.p_polys[j][2 * i + 1] * self.q_polys[j][2 * i + 1]
            }

            let mut pq_inf =
                (self.p_polys[0][2 * i] + self.p_polys[0][2 * i + 1])
                * (self.q_polys[0][2 * i] + self.q_polys[0][2 * i + 1]);
            
            for j in 1..l {
                pq_inf +=
                    (self.p_polys[j][2 * i] + self.p_polys[j][2 * i + 1])
                    * (self.q_polys[j][2 * i] + self.q_polys[j][2 * i + 1]);
            }

            [pq_zero, pq_one, pq_inf]
        });
        
        #[cfg(not(feature = "parallel"))]
        let mut response = iter.fold([F128::zero(), F128::zero(), F128::zero()], |[a, b, c], [d, e, f]| [a+d, b+e, c+f]);

        #[cfg(feature = "parallel")]
        let mut response = iter.reduce(|| [F128::zero(), F128::zero(), F128::zero()], |[a, b, c], [d, e, f]| [a+d, b+e, c+f]);

        // let acc : [u64; 6] = acc.iter().map(|x| x.load(Ordering::Relaxed)).collect_vec().try_into().unwrap();
        // let mut response = cast::<[u64; 6], [F128; 3]>(acc);

        // cast to coefficient form
        response[1] += response[0];
        response[1] += response[2];

        let (compressed_response, _) = CompressedPoly::compress(&response);

        self.cached_round_msg = Some(compressed_response.clone());
        compressed_response
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::OsRng;

    use crate::protocols::utils::evaluate;

    use super::*;

    #[test]
    fn prodcheck_works() {
        let rng = &mut OsRng;
        let num_vars = 15;

        let mut p_polys = vec![];
        let mut q_polys = vec![];
        for i in 0..5 {
            let p : Vec<_> = (0 .. 1 << num_vars).map(|_| F128::rand(rng)).collect();
            let q : Vec<_> = (0 .. 1 << num_vars).map(|_| F128::rand(rng)).collect();
            p_polys.push(p);
            q_polys.push(q);
        }
        let mut claim = p_polys.iter().flatten().zip(q_polys.iter().flatten()).map(|(a, b)| *a * b).fold(F128::zero(), |a, b| a + b);

        let mut prover = Prodcheck::new(p_polys.clone(), q_polys.clone(), claim, true, false);

        for i in 0..num_vars {
            let round_poly = prover.round_msg().coeffs(claim);
            let challenge = F128::rand(rng);
            claim = round_poly[0] + challenge * round_poly[1] + challenge * challenge * round_poly[2];
            prover.bind(challenge);
        }

        assert!(prover.p_polys[0].len() == 1);

        let ev_p : Vec<_> = p_polys.iter().map(|p| evaluate(&p, &prover.challenges)).collect();
        let ev_q : Vec<_> = q_polys.iter().map(|q| evaluate(&q, &prover.challenges)).collect();

        assert!(ev_p.iter().zip(ev_q.iter()).map(|(a, b)| *a * b).fold(F128::zero(), |a, b| a + b) == claim);
    }

}
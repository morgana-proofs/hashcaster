use num_traits::Zero;
use rayon::iter::{IndexedParallelIterator, ParallelIterator, IntoParallelIterator};

use crate::{field::F128, ptr_utils::{AsSharedConstPtr, AsSharedMutPtr, UnsafeIndexRaw, UnsafeIndexRawMut}, traits::{CompressedPoly, SumcheckObject}, utils::log2_exact};


/// A very simple sumcheck, only does product of 2 polynomials. It is used as main component for lincheck.
pub struct Prodcheck {
    pub p_polys: Vec<Vec<F128>>,
    pub q_polys: Vec<Vec<F128>>,
    p_scratch: Vec<Vec<F128>>,
    q_scratch: Vec<Vec<F128>>,
    active_len: usize,
    pub claim: F128,
    pub challenges: Vec<F128>,

    cached_round_msg: Option<CompressedPoly>,

    rev_order: bool, 
}

impl Prodcheck {
    pub fn new(
        p_polys: Vec<Vec<F128>>,
        q_polys: Vec<Vec<F128>>,
        p_scratch: Vec<Vec<F128>>,
        q_scratch: Vec<Vec<F128>>,
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
        assert!(p_scratch.len() == l);
        assert!(q_scratch.len() == l);
        for i in 0..l {
            assert!(p_scratch[i].len() == 1 << num_vars);
            assert!(q_scratch[i].len() == 1 << num_vars);
        }

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
            active_len: p_polys[0].len(),
            p_polys,
            q_polys,
            p_scratch,
            q_scratch,
            claim: initial_claim,
            challenges: vec![],
            cached_round_msg: None,
            rev_order: in_reverse_order,
        }
    }

    pub fn finish(self) -> ProdcheckOutput {
        assert!(self.active_len == 1);
        let p_evs = self.p_polys.iter().map(|poly| {poly[0]}).collect();
        let q_evs = self.q_polys.iter().map(|poly| {poly[0]}).collect();
        ProdcheckOutput{p_evs, q_evs}
    }
}

pub struct ProdcheckOutput {
    pub p_evs: Vec<F128>,
    pub q_evs: Vec<F128>
}

impl SumcheckObject for Prodcheck {

    fn is_reverse_order(&self) -> bool {
        self.rev_order
    }

    fn bind(&mut self, challenge: F128) {
        if self.rev_order {
            panic!("Unsupported order.");
        }
        assert!(self.active_len > 1, "The protocol has already ended.");
        let half = self.active_len / 2;
        let l = self.p_polys.len();

        let round_poly = self.round_msg().coeffs(self.claim);
        // Decompressed round polynomial in a coefficient form.
        self.claim = round_poly[0] + challenge * round_poly[1] + challenge * challenge * round_poly[2];
        self.challenges.push(challenge);

        let next_msg = if half > 1 {
            let next_half = half / 2;

            #[cfg(not(feature = "parallel"))]
            let mut response = {
                let mut response = [F128::zero(), F128::zero(), F128::zero()];
                for j in 0..next_half {
                    let mut pq_zero = F128::zero();
                    let mut pq_one = F128::zero();
                    let mut pq_inf = F128::zero();
                    for i in 0..l {
                        let p00 = self.p_polys[i][4 * j];
                        let p01 = self.p_polys[i][4 * j + 1];
                        let p10 = self.p_polys[i][4 * j + 2];
                        let p11 = self.p_polys[i][4 * j + 3];
                        let q00 = self.q_polys[i][4 * j];
                        let q01 = self.q_polys[i][4 * j + 1];
                        let q10 = self.q_polys[i][4 * j + 2];
                        let q11 = self.q_polys[i][4 * j + 3];
                        let p0 = p00 + (p01 + p00) * challenge;
                        let p1 = p10 + (p11 + p10) * challenge;
                        let q0 = q00 + (q01 + q00) * challenge;
                        let q1 = q10 + (q11 + q10) * challenge;
                        self.p_scratch[i][2 * j] = p0;
                        self.p_scratch[i][2 * j + 1] = p1;
                        self.q_scratch[i][2 * j] = q0;
                        self.q_scratch[i][2 * j + 1] = q1;
                        pq_zero += p0 * q0;
                        pq_one += p1 * q1;
                        pq_inf += (p0 + p1) * (q0 + q1);
                    }
                    response[0] += pq_zero;
                    response[1] += pq_one;
                    response[2] += pq_inf;
                }
                response
            };

            #[cfg(feature = "parallel")]
            let mut response = {
                let p_ptrs : Vec<_> = self.p_polys.iter().map(|p| p.as_shared_ptr()).collect();
                let q_ptrs : Vec<_> = self.q_polys.iter().map(|q| q.as_shared_ptr()).collect();
                let p_scratch_ptrs : Vec<_> = self.p_scratch.iter_mut().map(|p| p.as_shared_mut_ptr()).collect();
                let q_scratch_ptrs : Vec<_> = self.q_scratch.iter_mut().map(|q| q.as_shared_mut_ptr()).collect();

                (0..next_half).into_par_iter().with_min_len(1024).map(|j| {
                    let mut pq_zero = F128::zero();
                    let mut pq_one = F128::zero();
                    let mut pq_inf = F128::zero();
                    unsafe {
                        for i in 0..l {
                            let p_ptr = p_ptrs[i];
                            let q_ptr = q_ptrs[i];
                            let p_scratch_ptr = p_scratch_ptrs[i];
                            let q_scratch_ptr = q_scratch_ptrs[i];
                            let p00 = *p_ptr.get(4 * j);
                            let p01 = *p_ptr.get(4 * j + 1);
                            let p10 = *p_ptr.get(4 * j + 2);
                            let p11 = *p_ptr.get(4 * j + 3);
                            let q00 = *q_ptr.get(4 * j);
                            let q01 = *q_ptr.get(4 * j + 1);
                            let q10 = *q_ptr.get(4 * j + 2);
                            let q11 = *q_ptr.get(4 * j + 3);
                            let p0 = p00 + (p01 + p00) * challenge;
                            let p1 = p10 + (p11 + p10) * challenge;
                            let q0 = q00 + (q01 + q00) * challenge;
                            let q1 = q10 + (q11 + q10) * challenge;
                            *p_scratch_ptr.get_mut(2 * j) = p0;
                            *p_scratch_ptr.get_mut(2 * j + 1) = p1;
                            *q_scratch_ptr.get_mut(2 * j) = q0;
                            *q_scratch_ptr.get_mut(2 * j + 1) = q1;
                            pq_zero += p0 * q0;
                            pq_one += p1 * q1;
                            pq_inf += (p0 + p1) * (q0 + q1);
                        }
                    }
                    [pq_zero, pq_one, pq_inf]
                }).reduce(|| [F128::zero(), F128::zero(), F128::zero()], |[a, b, c], [d, e, f]| [a+d, b+e, c+f])
            };

            for i in 0..l {
                std::mem::swap(&mut self.p_polys[i], &mut self.p_scratch[i]);
                std::mem::swap(&mut self.q_polys[i], &mut self.q_scratch[i]);
            }
            response[1] += response[0];
            response[1] += response[2];
            let (compressed_response, _) = CompressedPoly::compress(&response);
            Some(compressed_response)
        } else {
            for i in 0..l {
                self.p_scratch[i][0] = self.p_polys[i][0] + (self.p_polys[i][1] + self.p_polys[i][0]) * challenge;
                self.q_scratch[i][0] = self.q_polys[i][0] + (self.q_polys[i][1] + self.q_polys[i][0]) * challenge;
                std::mem::swap(&mut self.p_polys[i], &mut self.p_scratch[i]);
                std::mem::swap(&mut self.q_polys[i], &mut self.q_scratch[i]);
            }
            None
        };

        self.active_len = half;
        self.cached_round_msg = next_msg;
    }

    fn round_msg(&mut self) -> CompressedPoly {

        assert!(self.active_len > 1, "The protocol has already ended.");
        let half = self.active_len / 2;

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
        let iter = (0 .. half).into_par_iter().with_min_len(1024);

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

        let p_scratch = p_polys.iter().map(|p| vec![F128::zero(); p.len()]).collect();
        let q_scratch = q_polys.iter().map(|q| vec![F128::zero(); q.len()]).collect();
        let mut prover = Prodcheck::new(p_polys.clone(), q_polys.clone(), p_scratch, q_scratch, claim, true, false);

        for i in 0..num_vars {
            let round_poly = prover.round_msg().coeffs(claim);
            let challenge = F128::rand(rng);
            claim = round_poly[0] + challenge * round_poly[1] + challenge * challenge * round_poly[2];
            prover.bind(challenge);
        }

        assert!(prover.active_len == 1);

        let mut eq = vec![F128::zero(); 1 << prover.challenges.len()];
        let ev_p : Vec<_> = p_polys.iter().map(|p| evaluate(&p, &prover.challenges, &mut eq)).collect();
        let ev_q : Vec<_> = q_polys.iter().map(|q| evaluate(&q, &prover.challenges, &mut eq)).collect();

        assert!(ev_p.iter().zip(ev_q.iter()).map(|(a, b)| *a * b).fold(F128::zero(), |a, b| a + b) == claim);
    }

}

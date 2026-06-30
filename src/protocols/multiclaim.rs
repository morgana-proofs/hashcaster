// This file contains a version of multiclaim argument, where a collection of polynomials P_i needs to be computed in
// an (inverse) Frobenius orbit of a point r.

// In this system, such collection of openings is given by Boolcheck protocol, and it caches some data - namely,
// the restrictions of coordinate polynomials (P_i)_j on sets of coordinates r_0, ..., r_k for k >= c - i.e. every
// restriction that occurs in a second phase.

// We exploit this, by doing the sumcheck in *reverse* order, starting from higher coordinates.

use std::iter::once;

use num_traits::{One, Zero};
use rayon::iter::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::{field::F128, precompute::frobenius_table::FROBENIUS, protocols::utils::frobenius_inv_lc, ptr_utils::{AsSharedMutPtr, UnsafeIndexRawMut}, traits::{CompressedPoly, SumcheckObject}};

use super::{prodcheck::Prodcheck, utils::{eq_poly, evaluate, evaluate_univar}};

pub struct MulticlaimCheck<'a, const N: usize> {
    polys: &'a [Vec<F128>; N],
    pt: &'a [F128],
    openings: &'a [F128],
}

impl<'a, const N: usize> MulticlaimCheck<'a, N> {
    pub fn new(polys: &'a [Vec<F128>; N], pt: &'a [F128], openings: &'a [F128]) -> Self {
        assert!(openings.len() == N * 128);
        for i in 0..N {
            assert!(polys[i].len() == 1 << pt.len());
        }
        Self { polys, pt, openings }
    }

    pub fn folding_challenge(self, gamma: F128, mut poly: Vec<F128>, eq: Vec<F128>, p_scratch: Vec<Vec<F128>>, q_scratch: Vec<Vec<F128>>) -> MulticlaimCheckSingle<'a, N> {
        let Self { polys, pt, openings } = self;
    
        let mut gamma_pows = Vec::with_capacity(128 * N);
        let mut tmp = F128::one();
        for i in 0..128*N {
            gamma_pows.push(tmp);
            tmp *= gamma;
        }

        let l = 1 << pt.len();
        assert!(poly.len() == l);

        #[cfg(not(feature = "parallel"))]
        let iter = (0..l);
        #[cfg(feature = "parallel")]
        let iter = (0..l).into_par_iter();

        let poly_ptr = poly.as_shared_mut_ptr();
        iter.map(|i| {
            let mut p = polys[0][i]; 
            for j in 1..N {
                p += polys[j][i] * gamma_pows[128 * j];
            }
            unsafe{ *poly_ptr.get_mut(i) = p; }
        }).count();

        let openings : Vec<F128> = (0..128).map(|i|{
            let mut o = openings[i];
            for j in 1..N {
                o += openings[i + j * 128] * gamma_pows[128 * j];
            }
            o
        }).collect();

        MulticlaimCheckSingle::new(poly, pt, openings, gamma_pows, polys, eq, p_scratch, q_scratch)       

    }
}

pub struct MulticlaimCheckSingle<'a, const N: usize> {
    polys: &'a [Vec<F128>; N],
    gamma128: F128,
    pub object: Prodcheck,
}

impl<'a, const N: usize> MulticlaimCheckSingle<'a, N> {
    pub fn new(poly: Vec<F128>, pt: &[F128], openings: Vec<F128>, gamma_pows: Vec<F128>, polys: &'a [Vec<F128>; N], mut eq: Vec<F128>, p_scratch: Vec<Vec<F128>>, q_scratch: Vec<Vec<F128>>) -> Self {
        assert!(eq.len() == 1 << pt.len());
        eq_poly(&pt, &mut eq);
        // We want to compute sum \gamma_i * eq(Frob^{-i}(r), x)
        // This can be done by applying matrix M_{\gamma} = (sum \gamma_i Frob^{-i}) to eq.
        let m = frobenius_inv_lc(&gamma_pows[0..128]);
        eq.par_iter_mut().map(|x| *x = m.apply(*x)).count();

        let initial_claim = &gamma_pows[0..128].iter().zip(openings.iter()).map(|(x, y)| *x * y).fold(F128::zero(), |x, y| x + y);

        Self{
            object: Prodcheck::new(
                vec![poly],
                vec![eq],
                p_scratch,
                q_scratch,
                *initial_claim,
                false,
                false
            ),
            polys,
            gamma128 : gamma_pows[127]*gamma_pows[1], // because gamma_pows[128] will not exist for N=1
        }
    }

    /// Returns openings.
    pub fn finish(self) -> Vec<F128> {
        let mut eq = vec![F128::zero(); 1 << self.object.challenges.len()];
        let mut ret : Vec<F128> = once(F128::zero()).chain((1..N).map(|i| evaluate(&self.polys[i], &self.object.challenges, &mut eq))).collect();
        let tmp = evaluate_univar(&ret, self.gamma128);
        ret[0] = tmp + self.object.p_polys[0][0];
        ret
    }

}

impl<'a, const N: usize> SumcheckObject for MulticlaimCheckSingle<'a, N> {
    fn is_reverse_order(&self) -> bool {
        self.object.is_reverse_order()
    }

    fn bind(&mut self, challenge: F128) {
        self.object.bind(challenge)
    }

    fn round_msg(&mut self) -> CompressedPoly {
        self.object.round_msg()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::protocols::utils::{eq_ev, evaluate};

    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn multiclaim_works() {
        let rng = &mut OsRng;
        let num_vars = 20;
        let poly : Vec<_> = (0 .. 1 << num_vars).map(|_| F128::rand(rng)).collect();
        let pt : Vec<_> = (0 .. num_vars).map(|_| F128::rand(rng)).collect();

        let mut pt_inv_orbit = vec![];
        for i in 0..128i32 {
            pt_inv_orbit.push(
                pt.iter().map(|x| x.frob(-i)).collect::<Vec<F128>>()
            )
        }
        let mut eq = vec![F128::zero(); 1 << pt.len()];
        let evs = (0..128).map(|i|{evaluate(&poly, &pt_inv_orbit[i], &mut eq)}).collect::<Vec<F128>>();

        let polys = [poly];

        let prover = MulticlaimCheck::new(&polys, &pt, &evs);
        
        let gamma = F128::rand(rng);
        

        let label0 = Instant::now();

        let multi_poly = vec![F128::zero(); 1 << num_vars];
        let multi_eq = vec![F128::zero(); 1 << num_vars];
        let multi_p_scratch = vec![vec![F128::zero(); 1 << num_vars]; 1];
        let multi_q_scratch = vec![vec![F128::zero(); 1 << num_vars]; 1];
        let mut prover = prover.folding_challenge(gamma, multi_poly, multi_eq, multi_p_scratch, multi_q_scratch);
        let mut gamma_pows = vec![];
        let mut tmp = F128::one();
        for _ in 0..128 {
            gamma_pows.push(tmp);
            tmp *= gamma;
        }
        let mut claim = gamma_pows.iter().zip(evs.iter()).map(|(x, y)| *x * y).fold(F128::zero(), |x, y| x + y);
        let mut rs = vec![];

        let label1 = Instant::now();

        let mut acc_round = 0;
        let mut acc_bind = 0;

        for i in 0..num_vars {
            let a = Instant::now();
            let rpoly = prover.round_msg().coeffs(claim);
            let b = Instant::now();
            
            acc_round += (b-a).as_millis();

            assert!(rpoly.len() == 3);
            let r = F128::rand(rng);
            claim = rpoly[0] + r * rpoly[1] + r * r * rpoly[2];
            rs.push(r);

            let c = Instant::now();
            prover.bind(r);
            let d = Instant::now();

            acc_bind += (d-c).as_millis();
        }
        let label2 = Instant::now();

        let eq_evs = 
            gamma_pows.iter()
                .zip(pt_inv_orbit.iter())
                .map(|(gamma, pt)| *gamma * eq_ev(&pt, &rs))
                .fold(F128::zero(), |x, y| x + y);

        println!("Matrix application time: {} ms", (label1 - label0).as_millis());
        println!("The rest: {} ms", (label2 - label1).as_millis());
        println!("Of these:");
        println!("round_msg(): {} ms", acc_round);
        println!("bind(): {} ms", acc_bind);

        let mut eq = vec![F128::zero(); 1 << rs.len()];
        assert!(evaluate(&polys[0], &rs, &mut eq) * eq_evs == claim);

    }
}

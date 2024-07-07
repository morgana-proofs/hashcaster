use std::mem::{transmute, MaybeUninit};

use num_traits::{One, Zero};
use crate::field::{pi, F128};
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
    assert_eq!(poly.len(), 1 << pt.len());
    poly.iter().zip_eq(eq_poly(pt)).fold(F128::zero(), |acc, (x, y)| acc + *x * y)
}

pub struct AndcheckProver {
    pt: Vec<F128>,
    p: Vec<F128>,
    q: Vec<F128>,
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
        let p_twists : Vec<_> = self.p_evs.iter().enumerate().map(|(i, x)|x.frob(i)).collect();
        let q_twists : Vec<_> = self.q_evs.iter().enumerate().map(|(i, x)|x.frob(i)).collect();
        for i in 0..128 {
            ret += F128::basis(i) * pi(i, &p_twists) * pi(i, &q_twists);
        }
        ret
    } 
}



impl AndcheckProver {
    pub fn new(pt: Vec<F128>, p: Vec<F128>, q: Vec<F128>, evaluation_claim: F128, check_correct: bool) -> Self {
        assert_eq!(1 << pt.len(), p.len());
        assert_eq!(1 << pt.len(), q.len());
        if check_correct {
            p.iter().zip_eq(q.iter()).zip_eq(eq_poly(&pt).iter()).fold(F128::zero(), |acc, ((&p, &q), &e)| {acc + (p & q) * e});
        }
        Self{pt, p, q, evaluation_claim, challenges: vec![]}
    }

    pub fn num_vars(&self) -> usize {
        self.pt.len()
    }

    pub fn curr_round(&self) -> usize {
        self.challenges.len()
    }

    pub fn round(&mut self, round_challenge: F128) -> RoundResponse {
        assert!(self.curr_round() < self.num_vars(), "Protocol has already finished.");

        println!("Starting round {}", self.curr_round());

        let p = &self.p;
        let q = &self.q;
        
        let pt = &self.pt;
        let round = self.curr_round();

        let pt_l = &pt[..round];
        let pt_g = &pt[(round + 1)..];
        let pt_r = pt[round];

        let half = 1 << (self.num_vars() - 1);

        // let mut p_and_q = vec![vec![MaybeUninit::<F128>::uninit(); half]; 3];
        // // (p_0 + t (p_1 - p_0)) & (q_0 + t (q_1 - q_0)), treating t as formal variable
        // // p0&q0 + t (p1&q0 + p0&q1) + t^2 (p0+p1)&(q0+q1)
        // for i in 0..(1 << round) {
        //     for j in 0..(1 << (self.num_vars() - round - 1)) {
        //         let bot_idx = (j << (round + 1)) + i;
        //         let top_idx = bot_idx + (1 << round);
        //         let idx = i + (j << round);

        //         let p0 = p[bot_idx];
        //         let p1 = p[top_idx];
        //         let q0 = q[bot_idx];
        //         let q1 = q[top_idx];

        //         p_and_q[0][idx] = MaybeUninit::new(p0 & q0);
        //         p_and_q[1][idx] = MaybeUninit::new((p0 & q1) + (p1 & q0));
        //         p_and_q[2][idx] = MaybeUninit::new((p0 + p1) & (q0 + q1));
        //     }
        // }

        let mut p_and_q = vec![vec![None; half]; 3];
        // (p_0 + t (p_1 - p_0)) & (q_0 + t (q_1 - q_0)), treating t as formal variable
        // p0&q0 + t (p1&q0 + p0&q1) + t^2 (p0+p1)&(q0+q1)
        for idx in 0..half {
                let hi = (idx >> round) << round;
                let lo = hi ^ idx;
                let bot_idx = (hi << 1) + lo;
                let top_idx = bot_idx + (1 << round);

                println!("bot: {}, top: {}, idx: {}", bot_idx, top_idx, idx);

                let p0 = p[bot_idx];
                let p1 = p[top_idx];
                let q0 = q[bot_idx];
                let q1 = q[top_idx];

                let a0 = p0 & q0;
                let a1 = (p0 & q1) + (p1 & q0);
                let a2 = a0 + a1 + (p1 & q1);

                p_and_q[0][idx] = Some(a0);
                p_and_q[1][idx] = Some(a1);
                p_and_q[2][idx] = Some(a2);
        }

        let p_and_q = p_and_q.iter().map(|v|v.iter().map(|x|x.unwrap()).collect_vec()).collect_vec();

        // We need to compute sum P(chall, t, x_>r) Q(chall, t, x_>r) eq(chall, t, x_>r; pt)
        // This is done (via n log n algo) by instead computing
        // P(y_<r, t, x_>r) Q(y_<r, t, x_>r) eq(chall, t, x_>r; pt) eq(y_<r; chall)
        // eq(chall, t, x_>r; pt) term is then replaced by eq(chall; pt_<r) eq(x_>r; pt_>r) eq(t, pt_r)

        let mut pt_concat = self.challenges.clone();
        pt_concat.extend(pt_g);

        let mult_table = eq_poly(&pt_concat); //eq(y_<r, chall) eq(x_>r, pt_>r)
        assert!(mult_table.len() == half);

        let mut poly = vec![F128::zero(); 3]; // This hosts sum of P AND Q * all terms not involving t.

        for s in 0..3 {
            for i in 0..half {
                poly[s] += p_and_q[s][i] * mult_table[i];
            } 
        }

        let eq_y_multiplier = eq_ev(&self.challenges, &pt_l);

        poly.iter_mut().map(|c| *c *= eq_y_multiplier).count();

        // eq(t, pt_r) = t pt_r + (1 - t) (1 - pt_r) = (1+pt_r) + t
        let eq_t = vec![pt_r + F128::one(), F128::one()];

        let prod = vec![
            eq_t[0] * poly[0],
            eq_t[0] * poly[1] + eq_t[1] * poly[0],
            eq_t[0] * poly[2] + eq_t[1] * poly[1],
            eq_t[1] * poly[2],
        ];

        let r2 = round_challenge * round_challenge;
        let r3 = round_challenge * r2;

        assert!(prod[1] + prod[2] + prod[3] == self.evaluation_claim);
        self.evaluation_claim = prod[0] + prod[1] * round_challenge + prod[2] * r2 + prod[3] * r3;
        self.challenges.push(round_challenge);

        RoundResponse { values: prod }
    }


    pub fn finish(&self) -> FinalClaim {
        assert!(self.curr_round() == self.num_vars(), "Protocol is not finished.");
        let mut inverse_orbit = vec![];
        let mut pt = self.challenges.clone();
        for _ in 0..128 {
            inverse_orbit.push(pt.clone());
            pt.iter_mut().map(|x| *x *= *x).count();
        }
        inverse_orbit.reverse();

        let mut p_evs = vec![];
        let mut q_evs = vec![];

        for i in 0..128 {
            p_evs.push(evaluate(&self.p, &inverse_orbit[i]));
            q_evs.push(evaluate(&self.q, &inverse_orbit[i]));
        }

        FinalClaim { p_evs, q_evs }
    }
}



#[cfg(test)]
mod tests {
    use std::iter::{repeat_with};

    use itertools::Itertools;
    use rand::rngs::OsRng;

    use crate::{andcheck::eq_ev, field::F128};

    use super::{eq_poly, evaluate, AndcheckProver};

    #[test]
    fn test_eq_ev() {
        let rng = &mut OsRng;
        let num_vars = 5;

        let x : Vec<_> = repeat_with(|| F128::rand(rng)).take(num_vars).collect();
        let y : Vec<_> = repeat_with(|| F128::rand(rng)).take(num_vars).collect();
        

        assert_eq!(eq_ev(&x, &y), evaluate(&eq_poly(&x), &y));
    }

    #[test]
    fn verify_prover() {
        let rng = &mut OsRng;
        let num_vars = 2;

        let pt : Vec<_> = repeat_with(|| F128::rand(rng)).take(num_vars).collect();
        let p : Vec<_> = repeat_with(|| F128::rand(rng)).take(1 << num_vars).collect();
        let q : Vec<_> = repeat_with(|| F128::rand(rng)).take(1 << num_vars).collect();

        let p_zip_q : Vec<_> = p.iter().zip_eq(q.iter()).map(|(x, y)| *x & *y).collect();
        let evaluation_claim = evaluate(&p_zip_q, &pt);

        let mut prover = AndcheckProver::new(pt, p, q, evaluation_claim, true);

        for i in 0..num_vars {
            let round_challenge = F128::rand(rng);
            prover.round(round_challenge);
        }

        assert_eq!(
            prover.finish().apply_algebraic_combinator() * eq_ev(&prover.pt, &prover.challenges),
            prover.evaluation_claim
        )
    }
}
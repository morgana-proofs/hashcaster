use std::time::Instant;

use num_traits::{One, Zero};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};
use rayon::slice::ParallelSlice;

use crate::protocols::utils::{evaluate, evaluate_univar};
use crate::traits::{CompressedPoly, SumcheckObject};
use crate::{field::F128, protocols::utils::eq_poly,};

use super::prodcheck::{Prodcheck, ProdcheckOutput};

pub trait LinOp {
    fn n_in(&self) -> usize;
    fn n_out(&self) -> usize;
    
    /// expects input of size n_in and output of size n_out
    /// adds result to already existing output using +=
    fn apply(&self, input: &[F128], output: &mut [F128]);
    /// expects input of size n_out and output of size n_in
    /// adds result to already existing output using +=
    fn apply_transposed(&self, input: &[F128], output: &mut [F128]);
}

pub struct Composition<A: LinOp, B: LinOp> {
    a: A,
    b: B,
}

impl<A: LinOp, B: LinOp> Composition<A, B> {
    pub fn new(a: A, b: B) -> Self {
        assert!(b.n_out() == a.n_in());
        Self { a, b }
    }
}

impl<A: LinOp, B: LinOp> LinOp for Composition<A, B> {
    fn n_in(&self) -> usize {
        self.b.n_in()
    }
    
    fn n_out(&self) -> usize {
        self.a.n_out()
    }

    fn apply(&self, input: &[F128], output: &mut [F128]) {
        let mid = self.b.n_out();
        let mut tmp = vec![F128::zero(); mid];
        self.b.apply(input, &mut tmp);
        self.a.apply(&tmp, output);
    }

    fn apply_transposed(&self, input: &[F128], output: &mut [F128]) {
        let mid = self.b.n_out();
        let mut tmp = vec![F128::zero(); mid];
        self.a.apply_transposed(input, &mut tmp);
        self.b.apply_transposed(&tmp, output);
    }
}

pub struct MatrixSum<A: LinOp, B: LinOp> {
    a: A,
    b: B,
}

impl<A: LinOp, B: LinOp> MatrixSum<A, B> {
    pub fn new(a: A, b: B) -> Self {
        assert!(b.n_in() == a.n_in());
        assert!(b.n_out() == a.n_out());
        Self { a, b }
    }
}

impl<A: LinOp, B: LinOp> LinOp for MatrixSum<A, B> {
    fn n_in(&self) -> usize {
        self.a.n_in()
    }

    fn n_out(&self) -> usize {
        self.a.n_out()
    }

    fn apply(&self, input: &[F128], output: &mut [F128]) {
        self.a.apply(input, output);
        self.b.apply(input, output);
    }

    fn apply_transposed(&self, input: &[F128], output: &mut [F128]) {
        self.a.apply_transposed(input, output);
        self.b.apply_transposed(input, output);
    }
}

pub struct IdentityMatrix {
    size: usize,
}

impl IdentityMatrix {
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}

impl LinOp for IdentityMatrix {
    fn n_in(&self) -> usize {
        self.size
    }

    fn n_out(&self) -> usize {
        self.size
    }

    fn apply(&self, input: &[F128], output: &mut [F128]) {
        for i in 0..self.size {
            output[i] += input[i]
        }
    }

    fn apply_transposed(&self, input: &[F128], output: &mut [F128]) {
        for i in 0..self.size {
            output[i] += input[i]
        }
    }
}

/// Represents a linear sumcheck of the form
/// M(pt_{n-a}, ... pt_{n-1}; x_{n-a}, ..., x_{n-1}) * P(pt_0, ..., pt_{n-a-1}, x_{n-a}, ..., x_{n-1}),
/// where `a` is a number of "active" variables.
/// It is very small, and the main computational work is computing the restriction
/// P(x_0, ..., x_{a-1}, pt_a, ..., pt_{n-1}).
/// M(pt_0, ... pt_{a-1}; x_0, ..., x_{a-1}) is computed by applying the transposition of matrix M
/// to the vector of values of a polynomial eq_poly(pt[0..a]).
/// Lincheck expects a matrix of size N*2^a x M*2^a, and it will be treated as matrix from N chunks of
/// size 2^a to M chunks of size 2^a.
pub struct Lincheck<const N: usize, const M: usize, L: LinOp> {
    matrix: L,
    polys: [Vec<F128>; N],
    pt: Vec<F128>,
    num_vars: usize,
    num_active_vars: usize,
    initial_claims: [F128; M],
}

impl<const N: usize, const M: usize, L: LinOp> Lincheck<N, M, L> {
    pub fn new(polys: [Vec<F128>; N], pt: Vec<F128>, matrix: L, num_active_vars: usize, initial_claims: [F128; M]) -> Self {
        assert!(matrix.n_in() == N * (1 << num_active_vars));
        assert!(matrix.n_out() == M * (1 << num_active_vars));
        let num_vars = pt.len();
        assert!(num_vars >= num_active_vars);
        for i in 0..N {
            assert!(polys[i].len() == 1 << num_vars);
        }
        Self { matrix, polys, pt, num_vars, num_active_vars, initial_claims }
    } 

    pub fn folding_challenge(self, gamma: F128) -> PreparedLincheck {
        let chunk_size = 1 << self.num_active_vars;
        let pt_active = &self.pt[ .. self.num_active_vars];
        let pt_dormant = &self.pt[self.num_active_vars .. ];
        // Restrict.
        
        let eq_dormant = eq_poly(&pt_dormant);
        let mut p_polys = vec![vec![F128::zero(); 1 << self.num_active_vars]; N];
        

        self.polys.into_iter().enumerate().map(|(i, poly)| {
            let poly_chunks = poly.chunks(chunk_size);
            poly_chunks.enumerate().map(|(j, chunk)| {
                p_polys[i].iter_mut().zip(chunk.iter()).map(|(p, c)| *p += eq_dormant[j] * c).count();
        }).count()}).count();

        let mut gamma_pows = Vec::with_capacity(M);
        let mut tmp = F128::one();
        for _ in 0..M {
            gamma_pows.push(tmp);
            tmp *= gamma;
        }
        let eq = eq_poly(&pt_active);
        let gamma_eqs: Vec<_> = gamma_pows.iter()
            .map(|gpow| (0..(1 << self.num_active_vars))
            .map(|i| *gpow * eq[i]))
            .flatten()
            .collect();

        let mut q = vec![F128::zero(); N * (1 << self.num_active_vars)];
        self.matrix.apply_transposed(&gamma_eqs, &mut q);
        // q(x) = M(pt[0..a], x)

        let mut q_polys = vec![];
        for _ in 0..N {
            let tmp = q.split_off(1 << self.num_active_vars);
            q_polys.push(q);
            q = tmp;
        }
        //sanity:
        assert_eq!(q.len(), 0);

        let claim = evaluate_univar(&self.initial_claims, gamma);

        PreparedLincheck{
            object: Prodcheck::new(p_polys, q_polys, claim, false, false)
        }
    }
}

pub struct PreparedLincheck {
    object: Prodcheck
}

impl PreparedLincheck {
    pub fn finish(self) -> LincheckOutput{
        self.object.finish()
    }
}

impl SumcheckObject for PreparedLincheck {
    fn is_reverse_order(&self) -> bool {
        false
    }

    fn round_msg(&mut self) -> CompressedPoly {
        self.object.round_msg()
    }

    fn bind(&mut self, challenge: F128) {
        self.object.bind(challenge)
    }
}

// Final claim of lincheck. Consists of 
pub type LincheckOutput = ProdcheckOutput;


#[cfg(test)]
mod tests {
    use std::time::Instant;

    use itertools::Itertools;
    use rand::rngs::OsRng;
    use rayon::iter::IndexedParallelIterator;
    use rayon::slice::ParallelSliceMut;

    use super::*;

    // Arbitrary matrix. Not efficient. We will use it for testing.
    #[derive(Clone)]
    pub struct GenericLinop {
        n_in: usize,
        n_out: usize,
        entries: Vec<Vec<F128>>,
    }

    impl GenericLinop {
        pub fn new(entries: Vec<Vec<F128>>) -> Self {
            let n_out = entries.len();
            let n_in = entries[0].len();
            for i in 1..n_out {
                assert!(entries[i].len() == n_in);
            }

            Self { n_in, n_out, entries }

        }
    }

    impl LinOp for GenericLinop {
        fn n_in(&self) -> usize {
            self.n_in
        }
    
        fn n_out(&self) -> usize {
            self.n_out
        }
    
        fn apply(&self, input: &[F128], output: &mut [F128]) {
            assert!(input.len() == self.n_in);
            assert!(output.len() == self.n_out);
            for i in 0..self.n_out {
                output[i] = F128::zero();
            }
            for i in 0..self.n_in {
                for j in 0..self.n_out {
                    output[j] += self.entries[j][i] * input[i];
                }
            }
        }
    
        fn apply_transposed(&self, input: &[F128], output: &mut [F128]) {
            assert!(input.len() == self.n_out);
            assert!(output.len() == self.n_in);
            for i in 0..self.n_in {
                output[i] = F128::zero();
            }
            for i in 0..self.n_out {
                for j in 0..self.n_in {
                    output[j] += self.entries[i][j] * input[i];
                }
            }
        }
    }

    #[test]
    fn generic_matrix() {
        let rng = &mut OsRng;

        let mut entries = vec![];

        for i in 0..4 {
            let mut tmp = vec![];
            for j in 0..5 {
                tmp.push(F128::rand(rng))
            }
            entries.push(tmp);
        }

        let linop = GenericLinop::new(entries);

        let v : Vec<_> = (0..5).map(|_| F128::rand(rng)).collect();
        let mut mv = vec![F128::zero(); 4];

        let w : Vec<_> = (0..4).map(|_| F128::rand(rng)).collect();
        let mut mtw = vec![F128::zero(); 5];

        linop.apply(&v, &mut mv);
        linop.apply_transposed(&w, &mut mtw);

        let lhs = v.iter().zip_eq(mtw.iter()).map(|(a, b)| *a * b).fold(F128::zero(), |a, b| a + b);
        let rhs = mv.iter().zip_eq(w.iter()).map(|(a, b)| *a * b).fold(F128::zero(), |a, b| a + b);

        assert!(lhs == rhs);

    }

    #[test]
    fn lincheck_works() {
        let rng = &mut OsRng;

        let num_vars = 20;
        let num_active_vars = 10;

        
        let mut entries = vec![];

        for i in 0..1 << num_active_vars {
            let mut tmp = vec![];
            for j in 0..1 << num_active_vars {
                tmp.push(F128::rand(rng))
            }
            entries.push(tmp);
        }

        let linop = GenericLinop::new(entries);

        let pt : Vec<_> = (0..num_vars).map(|_| F128::rand(rng)).collect();
        let poly : Vec<_> = (0 .. 1 << num_vars).map(|_| F128::rand(rng)).collect();
        let mut l_p : Vec<_> = vec![F128::zero(); 1 << num_vars]; 
        poly.par_chunks(1 << num_active_vars)
            .zip(l_p.par_chunks_mut(1 << num_active_vars))
            .map(|(src, dst)| linop.apply(src, dst)).count();
        // we will have more efficient witness computation later anyway

        let initial_claim = evaluate(&l_p, &pt);

        let label0 = Instant::now();

        let p_ = poly.clone();

        let label1_5 = Instant::now();

        let prover = Lincheck::<1, 1, _>::new([p_], pt.clone(), linop.clone(), num_active_vars, [initial_claim]);

        let mut prover = prover.folding_challenge(F128::rand(rng));

        let label1 = Instant::now();

        let mut rs = vec![];
        let mut claim = initial_claim;

        for _ in 0..num_active_vars {
            let rpoly = prover.round_msg().coeffs(claim);
            let r = F128::rand(rng);
            claim = rpoly[0] + rpoly[1] * r + rpoly[2] * r * r;
            prover.bind(r);
            rs.push(r);
        };

        let label2 = Instant::now();

        let LincheckOutput {p_evs, q_evs} = prover.finish();

        let eq1 = eq_poly(&pt[..num_active_vars]);
        let eq0 = eq_poly(&rs);
        let mut adj_eq_vec = vec![];
    
        let mut mult = F128::one();
        for i in 0..1 {
            adj_eq_vec.extend(eq1.iter().map(|x| *x * mult));
        };
        let mut target = vec![F128::zero(); 1 << num_active_vars];

        linop.apply_transposed(&eq1, &mut target);
    
        let s = target.iter()
            .zip(eq0.iter())
            .map(|(a, b)| *a * b)
            .fold(F128::zero(), |a, b| a + b);

        assert!(q_evs[0] == s);

        assert!(p_evs[0] * q_evs[0] == claim);

        let label3 = Instant::now();

        println!("Time elapsed: {} ms", (label3 - label0).as_millis());
        println!("> Clone: {} ms", (label1_5 - label0).as_millis());
        println!("> Init: {} ms", (label1 - label1_5).as_millis());
        println!("> Prodcheck maincycle: {} ms", (label2 - label1).as_millis());
        println!("> Finish: {} ms", (label3 - label2).as_millis());

        rs.extend(pt[num_active_vars..].iter().map(|x| *x));
        assert!(rs.len() == num_vars);

        assert!(p_evs[0] == evaluate(&poly, &rs));

    }

}
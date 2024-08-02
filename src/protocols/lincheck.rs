use num_traits::{One, Zero};

use crate::{field::F128, utils::log2_exact};

use super::prodcheck::Prodcheck;

pub trait LinOp {
    fn n_in(&self) -> usize;
    fn n_out(&self) -> usize;
    
    /// expects input of size n_in and output of size n_out
    fn apply(&self, input: &[F128], output: &mut [F128]);
        /// expects input of size n_out and output of size n_in
    fn apply_transposed(&self, input: &[F128], output: &mut [F128]);
}

pub struct Composition <A: LinOp, B: LinOp> {
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


/// Represents a linear sumcheck of the form
/// M(pt_0, ... pt_{a-1}; x_0, ..., x_{a-1}) * P(x_0, ..., x_{a-1}, pt_a, ..., pt_{n-1}),
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
}

impl<const N: usize, const M: usize, L: LinOp> Lincheck<N, M, L> {
    pub fn new(polys: [Vec<F128>; N], pt: Vec<F128>, matrix: L, num_active_vars: usize) -> Self {
        assert!(matrix.n_in() == N * (1 << num_active_vars));
        assert!(matrix.n_out() == M * (1 << num_active_vars));
        let num_vars = pt.len();
        assert!(num_vars >= num_active_vars);
        for i in 0..N {
            assert!(polys[i].len() == 1 << num_vars);
        }
        Self { matrix, polys, pt, num_vars, num_active_vars }
    } 

    pub fn folding_challenge(self, gamma: F128) -> PreparedLincheck {
        let mut gamma_pows = Vec::with_capacity(M);
        let mut tmp = F128::one();
        for _ in 0..M {
            gamma_pows.push(tmp);
            tmp *= gamma;
        }
        let eq = eq_poly();
    }
}

pub struct PreparedLincheck {
    object: Prodcheck
}

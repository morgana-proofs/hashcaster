use std::{iter::once, mem::MaybeUninit};
use num_traits::Zero;

use crate::{andcheck::{eq_ev, eq_poly}, field::F128};

pub struct CompressedPoly {
    pub compressed_coeffs: Vec<F128>,
}

impl CompressedPoly {
    /// Recovers full polynomial from its compressed form and previous claim (which is P(0) + P(1)).
    pub fn coeffs(&self, sum: F128) -> Vec<F128> {
        let ev_at_1 = self.compressed_coeffs.iter().fold(F128::zero(), |a, b| a + b);
        let ev_at_0 = sum + ev_at_1;
        once(ev_at_0).chain(self.compressed_coeffs.iter().map(|x|*x)).collect()
    }
}

pub trait TOpeningStatement {
    /// Computes the expected claim from the opening statement.
    fn apply_combinator(&self) -> F128;
}

pub trait SumcheckObject {
    type OpeningStatement : TOpeningStatement;
    type CachedData;

    /// Returns false if the order is standard (from small-bit coordinates to large-bit), and true otherwise.
    fn is_reverse_var_order(&self) -> bool;
    /// Current claim. Expected to be equal to either combinator of the final claim, or msg(0)+msg(1) --
    /// though because msg is compressed, it can actually be recovered using this claim.
    fn current_claim(&self) -> F128;
    /// Current univariate round polynomial. None means that the protocol has ended.
    fn msg(&self) -> Option<CompressedPoly>;
    /// Accept a new challenge. Will panic if the challenge is not expected.
    fn challenge(&mut self, challenge: F128);
    /// Finish the protocol. Returns an opening statement and arbitrary cached data (might be used by later protocols).
    fn finish(self) -> (Self::OpeningStatement, Self::CachedData);
}


/// This describes a matrix from I arrays of size 2^logsize_in, to O arrays of size 2^logsize_outp 
pub trait AdmissibleMatrix<const I: usize, const O: usize> {
    fn logsize_in(&self) -> usize;
    fn logsize_out(&self) -> usize;
    fn apply(&self, src: [&[F128]; I], dst: [&mut[MaybeUninit<F128>]; O]);
    /// Transposition of affine mapping (for example, v -> Mv + C) is separately
    /// w -> (M^t w, <C, w>)
    /// M^t w must be written into dst, and <C, w> returned from function.
    fn apply_transposed(&self, src: [&[F128]; O], dst: [&mut[MaybeUninit<F128>]; I]);
}
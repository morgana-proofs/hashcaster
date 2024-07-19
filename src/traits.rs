use std::{iter::once, mem::{transmute, MaybeUninit}};
use num_traits::Zero;
use rayon::iter::IntoParallelIterator;

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
pub trait AdmissibleMatrix{
    fn num_input_polys(&self) -> usize;
    fn num_output_polys(&self) -> usize;
    fn logsize_in(&self) -> usize;
    fn logsize_out(&self) -> usize;
    /// Unsafe contract: assumes that src.len() == num_input_polys, dst.len() == num_output_polys
    /// src[0].len() == logsize_in, dst[0].len() == logsize_out
    /// MUST initialize dst fully
    unsafe fn apply(&self, src: &[&[F128]], dst: &[&mut[MaybeUninit<F128>]]);
    /// Same as apply, with src and dst switched.
    unsafe fn apply_transposed(&self, src: &[&[F128]], dst: &[&mut[MaybeUninit<F128>]]);

    fn apply_full(&self, input: &[&[F128]]) -> Vec<Vec<F128>> {
        let num_input_polys = self.num_input_polys();
        let num_output_polys = self.num_output_polys();
        
        assert!(input.len() == num_input_polys);
        assert!(input.len() > 0, "Trivial case with 0 input unsupported because lazy.");
        
        let chunk_len_i = 1 << self.logsize_in();
        let chunk_len_o = 1 << self.logsize_out();
        
        assert!(input[0].len() % chunk_len_i == 0);
        let nchunks = input[0].len() / chunk_len_i;

        let mut ret = vec![];

        for _ in 0..num_output_polys {
            ret.push(vec![MaybeUninit::<F128>::uninit(); nchunks * chunk_len_o]);
        }

        let mut input_slices : Vec<_> = input.iter().map(|x| x.chunks(chunk_len_i)).collect();
        let mut output_slices : Vec<_> = ret.iter_mut().map(|x| x.chunks_mut(chunk_len_o)).collect();

        let mut in_slices : Vec<&[F128]> = vec![&[]; num_input_polys];
        let mut out_slices : Vec<&mut[MaybeUninit<F128>]> = vec![];
        for _ in 0..num_output_polys {
            out_slices.push(&mut[]);
        };

        // This can be parallelized if necessary, with high but acceptable amount of pain.
        for _ in 0..nchunks {
            for i in 0..num_input_polys {
                in_slices[i] = input_slices[i].next().unwrap();
            }
            for i in 0..num_output_polys {
                out_slices[i] = output_slices[i].next().unwrap();
            }
            unsafe{ self.apply(&in_slices, &mut out_slices) };
        }
        
        unsafe{transmute::<Vec<Vec<MaybeUninit<F128>>>, Vec<Vec<F128>>>(ret)}
    }
}
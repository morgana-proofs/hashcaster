use std::{iter::once, mem::{MaybeUninit}};
use num_traits::Zero;
use bytemuck;
use rayon::iter::IntoParallelIterator;

use crate::{field::F128, ptr_utils::UninitArr};

#[derive(Clone, Debug)]
pub struct CompressedPoly {
    pub compressed_coeffs: Vec<F128>,
}

impl CompressedPoly {
    /// Skips 1st coefficient.
    pub fn compress(poly: &[F128]) -> (Self, F128) {
        let sum = poly.iter().skip(1).fold(F128::zero(), |a, b| a + b);
        (
            Self{compressed_coeffs: once(&poly[0]).chain(poly[2..].iter()).map(|x|*x).collect()},
            sum
        )
    }
    
    /// Recovers full polynomial from its compressed form and previous claim (which is P(0) + P(1)).
    pub fn coeffs(&self, sum: F128) -> Vec<F128> {
        let coeff_0 = self.compressed_coeffs[0];
        let ev_1 = coeff_0 + sum;
        // evaluation in 1 is sum of all coefficients
        // therefore to compute 1st coefficient, we need to add all others to eval at 1
        let coeff_1 = self.compressed_coeffs.iter().fold(F128::zero(), |a, b| a + b) + ev_1;

        once(coeff_0).chain(once(coeff_1)).chain(self.compressed_coeffs[1..].iter().map(|x|*x)).collect()
    }
}

/// This describes a matrix from I arrays of size 2^logsize_in, to O arrays of size 2^logsize_outp 
pub trait AdmissibleMatrixLegacy<const N: usize, const M: usize>{
    fn logsize_in(&self) -> usize;
    fn logsize_out(&self) -> usize;
    /// Unsafe contract: assumes that src.len() == num_input_polys, dst.len() == num_output_polys
    /// src[0].len() == logsize_in, dst[0].len() == logsize_out
    /// MUST initialize dst fully
    unsafe fn apply(&self, src: &[&[F128]; N], dst: &[&mut[MaybeUninit<F128>]; M]);
    /// Same as apply, with src and dst switched.
    unsafe fn apply_transposed(&self, src: &[&[F128]], dst: &[&mut[MaybeUninit<F128>]]);

    fn apply_full(&self, input: &[&[F128]]) -> Vec<Vec<F128>> {        
        assert!(input.len() == N);
        assert!(input.len() > 0, "Trivial case with 0 input unsupported because lazy.");
        
        let chunk_len_i = 1 << self.logsize_in();
        let chunk_len_o = 1 << self.logsize_out();
        
        assert!(input[0].len() % chunk_len_i == 0);
        let nchunks = input[0].len() / chunk_len_i;

        let mut ret = vec![];

        for _ in 0..M {
            ret.push(UninitArr::<F128>::new(nchunks * chunk_len_o));
        }

        let mut input_slices : Vec<_> = input.iter().map(|x| x.chunks(chunk_len_i)).collect();
        let mut output_slices : Vec<_> = ret.iter_mut().map(|x| x.chunks_mut(chunk_len_o)).collect();

        let mut in_slices : [&[F128]; N] = [&[]; N];
        let mut out_slices : Vec<&mut[MaybeUninit<F128>]> = vec![];
        for _ in 0..M {
            out_slices.push(&mut[]);
        };

        let mut out_slices : [&mut [MaybeUninit<F128>]; M] = out_slices.try_into().unwrap();

        // This can be parallelized if necessary, with high but acceptable amount of pain.
        for _ in 0..nchunks {
            for i in 0..N {
                in_slices[i] = input_slices[i].next().unwrap();
            }
            for i in 0..M {
                out_slices[i] = output_slices[i].next().unwrap();
            }
            unsafe{ self.apply(&in_slices, &mut out_slices) };
        }

        ret.into_iter().map(|arr| unsafe{arr.assume_init()}).collect()
    }
}

pub trait SumcheckObject {

    fn is_reverse_order(&self) -> bool;
    /// Binds coordinates by the challenge.
    fn bind(&mut self, challenge: F128);
    /// Returns current round message.
    /// Receiver is mutable to give it an opportunity to cache some data. This operation MUST be idempotent.
    fn round_msg(&mut self) -> CompressedPoly;

}

// pub trait Protocol {
//     type InitClaim;
//     type RoundResponse;
//     type FinalClaim;
//     type Params;

//     type Prover : ProtocolProver<
//         InitClaim = Self::InitClaim,
//         RoundResponse = Self::RoundResponse,
//         FinalClaim = Self::FinalClaim,
//         Params = Self::Params,
//     >;
    
//     type Verifier : ProtocolVerifier<
//         InitClaim = Self::InitClaim,
//         RoundResponse = Self::RoundResponse,
//         FinalClaim = Self::FinalClaim,
//         Params = Self::Params,
//     >;

//     fn prover(
//         claim: Self::InitClaim,
//         params: Self::Params,
//         init_data: <Self::Prover as ProtocolProver>::InitData
//     ) -> Self::Prover;
    
//     fn verifier(
//         claim: Self::InitClaim,
//         params: Self::Params
//     ) -> Self::Verifier;

// }

// pub trait ProtocolProver {
//     type InitClaim;
//     type RoundResponse;
//     type FinalClaim;
//     type Params;

//     type InitData;
//     type CachedData;

//     fn challenge(&mut self, challenge: F128);
//     fn msg(&self) -> Self::RoundResponse;
//     fn finish(self) -> (Self::FinalClaim, Self::CachedData);
// }

// pub trait ProtocolVerifier {
//     type InitClaim;
//     type RoundResponse;
//     type FinalClaim;
//     type Params;

//     fn round(&mut self, msg: Self::RoundResponse, challenge: F128);
//     fn finish(self, final_claim: Self::FinalClaim);

// }
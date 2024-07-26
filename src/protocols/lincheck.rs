use std::marker::PhantomData;

use rayon::iter::IntoParallelIterator;

use crate::{field::F128, traits::{AdmissibleMatrix}};
use super::{prodcheck::Prodcheck, utils::{EvaluationClaim, RoundResponse}};

pub struct Lincheck<M : AdmissibleMatrix> {
    _marker: PhantomData<M>,
    prodcheck_object: Prodcheck,
}

impl<M: AdmissibleMatrix> Lincheck<M> {
    pub fn new(polys: Vec<Vec<F128>>, point: &[F128], folding_challenge: F128) -> Self {
        let num_vars = point.len();
        
        for p in &polys {
            assert!(polys.len() == 1 << num_vars);
        };

        #[cfg(not(feature = "parallel"))]
        let iter = (0 .. polys[0].len()).into_iter();

        #[cfg(feature = "parallel")]
        let iter = (0 .. polys[0].len()).into_par_iter();
        
        todo!();
    }
}


// pub struct Lincheck<M: AdmissibleMatrix> {
//     _marker: PhantomData<M>,
// }
// pub struct LincheckProver<M: AdmissibleMatrix> {
//     current_claim: F128,
//     folding_coeffs: Option<Vec<F128>>,
//     challenges: Vec<F128>,
//     polys: Vec<Vec<F128>>,
//     m_pullback: Vec<Vec<F128>>,
//     params: LincheckParams<M>,
// }
// pub struct LincheckVerifier<M: AdmissibleMatrix> {
//     params: LincheckParams<M>,
// }
// pub struct LincheckParams<M : AdmissibleMatrix> {
//     matrix: M,
//     input_logsize: usize,
//     n_batches: usize,
// }

// impl<M: AdmissibleMatrix> LincheckParams<M> {
//     pub fn new(matrix: M, input_logsize: usize, n_batches: usize) -> Self {
//         assert!(matrix.logsize_in() <= input_logsize);
//         Self { matrix, input_logsize, n_batches }
//     }
// }

// impl<M: AdmissibleMatrix> Protocol for Lincheck<M> {
//     type InitClaim = EvaluationClaim;
//     type RoundResponse = RoundResponse;
//     type FinalClaim = EvaluationClaim;
//     type Params = LincheckParams<M>;

//     type Prover = LincheckProver<M>;
//     type Verifier = LincheckVerifier<M>;

//     fn prover(
//         claim: Self::InitClaim,
//         params: Self::Params,
//         init_data: <Self::Prover as ProtocolProver>::InitData
//     ) -> Self::Prover {
//         todo!()
//     }

//     fn verifier(
//         claim: Self::InitClaim,
//         params: Self::Params
//     ) -> Self::Verifier {
//         todo!()
//     }
// }
// impl<M: AdmissibleMatrix> ProtocolProver for LincheckProver<M> {
//     type InitClaim = EvaluationClaim;
//     type RoundResponse = RoundResponse;
//     type FinalClaim = EvaluationClaim;
//     type Params = LincheckParams<M>;
//     type InitData = Vec<Vec<F128>>;
//     type CachedData = ();
    
//     fn challenge(&mut self, challenge: F128) {
//         todo!()
//     }
    
//     fn msg(&self) -> Self::RoundResponse {
//         todo!()
//     }
    
//     fn finish(self) -> (Self::FinalClaim, Self::CachedData) {
//         todo!()
//     }

// }
// impl<M: AdmissibleMatrix> ProtocolVerifier for LincheckVerifier<M> {
//     type InitClaim = EvaluationClaim;
//     type RoundResponse = RoundResponse;
//     type FinalClaim = EvaluationClaim;
//     type Params = LincheckParams<M>;
    
//     fn round(&mut self, msg: Self::RoundResponse, challenge: F128) {
//         todo!()
//     }
    
//     fn finish(self, final_claim: Self::FinalClaim) {
//         todo!()
//     }
// }
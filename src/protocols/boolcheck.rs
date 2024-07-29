use crate::field::F128;

pub struct BoolCheck<const N: usize, const M: usize, F: Fn([F128; N]) -> [F128; M] + Send + Sync> {
    f: F,
    pt: Vec<F128>,
    polys: [Vec<F128>; N], // Input polynomials.
    c: usize, // PHASE SWITCH, round < c => PHASE 1.
    evaluation_claims: [F128; M],
}

impl<const N: usize, const M: usize, F: Fn([F128; N]) -> [F128; M] + Send + Sync> BoolCheck<N, M, F> {

    pub fn new(polys: [Vec<F128>; N], c: usize, evaluation_claims: [F128; M], pt: Vec<F128>, f: F) -> Self {
        for poly in polys.iter() {
            assert!(polys.len() == 1 << pt.len());
        }
        assert!(c < pt.len());

        Self{f, pt, polys, c, evaluation_claims}
    }

}

pub struct FoldedBoolCheck<const N: usize, F: Fn([F128; N]) -> F128 + Send + Sync> {
    f_folded: F,
    pt: Vec<F128>,

    polys: [Vec<F128>; N], // Input polynomials.
    ext: Option<Vec<F128>>, // Extension of output on 3^{c+1} * 2^{n-c-1}, during first phase.
    polys_coords: Option<[Vec<Vec<F128>>; N]>, // Coordinates of input polynomials, in the second phase.

    c: usize, // PHASE SWITCH, round < c => PHASE 1.
    evaluation_claim: F128,
    challenges: Vec<F128>,
    bits_to_trits_map: Vec<u16>,
    eq_sequence: Vec<Vec<F128>>, // Precomputed eqs of all slices pt[i..].
}
use crate::traits::AdmissibleMatrix;

/// Sumcheck for the linear mapping M_{ij}(y, x) P^i(x)
pub struct Lincheck<M> {
    matrix: M,
}

impl<M: AdmissibleMatrix> Lincheck<M> {
    pub fn new(matrix: M) -> Self {
        Self{ matrix }
    }
}
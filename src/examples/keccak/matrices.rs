// This module describes matrices used in linear rounds of keccak.
// Each linear operator acts on 5 polynomials, each split into batches of size 1024.
// 
// batch of size 1024 is represented as 3 batches of size 320, and the tail of size 64.
// Matrices treat each batch x 5 as the state space of keccak (1600 bits), and the tail is zeroized.
//
// As we are actually acting on F128-elements, all our operations are 128-vectorized by default.

use crate::{field::F128, protocols::lincheck::{Composition, IdentityMatrix, LinOp, MatrixSum}};

fn idx(x: usize, y: usize, z: usize) -> usize {
    x * 320 + y * 64 + z
}

/// Matrix C[x] = A[x, 0] + A[x, 1] + A[x, 2] + A[x, 3] + A[x, 4]
pub struct ThetaAC {}

impl LinOp for ThetaAC {
    fn n_in(&self) -> usize {
        1600
    }

    fn n_out(&self) -> usize {
        320
    }

    fn apply(&self, input: &[F128], output: &mut [F128]) {
        for x in 0..5 {
            for y in 0..5 {
                for z in 0..64 {
                    output[x * 64 + z] += input[idx(x, y, z)];
                }
            }
        }
    }

    fn apply_transposed(&self, input: &[F128], output: &mut [F128]) {
        for x in 0..5 {
            for y in 0..5 {
                for z in 0..64 {
                    output[idx(x, y, z)] += input[x * 64 + z];
                }
            }
        }
    }
}

/// Matrix D[x] = C[x-1] + rot(C[x + 1], 1)
/// rotation is defined as moving bit i to position i+1, so I presume that it means that I need to access C[x+1, z-1]
pub struct ThetaCD {}

impl LinOp for ThetaCD {
    fn n_in(&self) -> usize {
        320
    }

    fn n_out(&self) -> usize {
        320
    }

    fn apply(&self, input: &[F128], output: &mut [F128]) {
        for x in 0..5 {
            for z in 0..64 {
                output[x * 64 + z] += input[((x + 4) % 5) * 64 + z] + input[((x + 1) % 5) * 64 + (z + 63) % 64]
            }
        }
    }

    fn apply_transposed(&self, input: &[F128], output: &mut [F128]) {
        for x in 0..5 {
            for z in 0..64 {
                output[x * 64 + z] += input[((x + 1) % 5) * 64 + z] + input[((x + 4) % 5) * 64 + (z + 1) % 64]
            }
        }
    }
}

/// Matrix E[x, y] = D[x]
/// In fact, it is dual to ThetaAC
pub struct ThetaDE {}

impl LinOp for ThetaDE {
    fn n_in(&self) -> usize {
        320
    }
    
    fn n_out(&self) -> usize {
        1600
    }
    
    fn apply(&self, input: &[F128], output: &mut [F128]) {
        for x in 0..5 {
            for y in 0..5 {
                for z in 0..64 {
                    output[idx(x, y, z)] += input[x * 64 + z];
                }
            }
        }
    }
    
    fn apply_transposed(&self, input: &[F128], output: &mut [F128]) {
        for x in 0..5 {
            for y in 0..5 {
                for z in 0..64 {
                    output[x * 64 + z] += input[idx(x, y, z)];
                }
            }
        }
    }
}

pub struct ThetaMatrix {
    m: MatrixSum<IdentityMatrix, Composition<ThetaDE, Composition<ThetaCD, ThetaAC>>>
}

pub const ROTATIONS : [[usize; 5]; 5] = [
    [0, 36, 3, 41, 18], // x = 0
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
];

/// Realizes B[y, 2x + 3y] = rot(A[x, y], r[x, y])
pub struct RhoPiMatrix {}

impl LinOp for RhoPiMatrix {
    fn n_in(&self) -> usize {
        1600
    }

    fn n_out(&self) -> usize {
        1600
    }

    fn apply(&self, input: &[F128], output: &mut [F128]) {
        for x in 0..5 {
            for y in 0..5 {
                for z in 0..64 {
                    output[idx(y, (2*x + 3*y) % 5, z)] += input[
                        idx(x, y, (z + 64 - ROTATIONS[x][y]) % 64)
                    ];
                }
            }
        }
    }

    fn apply_transposed(&self, input: &[F128], output: &mut [F128]) {
        for x in 0..5 {
            for y in 0..5 {
                for z in 0..64 {
                    output[idx(x, y, (z + 64 - ROTATIONS[x][y]) % 64)] += input[
                        idx(y, (2*x + 3*y) % 5, z)
                    ];
                }
            }
        }
    }
}


impl ThetaMatrix {
    pub fn new() -> Self {
        Self{
            m: MatrixSum::new(
                IdentityMatrix::new(1600), 
                Composition::new(ThetaDE{}, Composition::new(ThetaCD{}, ThetaAC{})),
            )
        }
    }
}

impl LinOp for ThetaMatrix {
    fn n_in(&self) -> usize {
        self.m.n_in()
    }

    fn n_out(&self) -> usize {
        self.m.n_out()
    }

    fn apply(&self, input: &[F128], output: &mut [F128]) {
        self.m.apply(input, output)
    }

    fn apply_transposed(&self, input: &[F128], output: &mut [F128]) {
        self.m.apply_transposed(input, output)
    }
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;
    use rand::rngs::OsRng;
    use super::*;

    #[test]
    fn theta_transpose() -> () {
        let rng = &mut OsRng;
        let theta = ThetaMatrix::new();

        assert!(theta.n_in() == 1600);
        assert!(theta.n_out() == 1600);

        let a : Vec<_> = (0..1600).map(|_| F128::rand(rng)).collect();
        let mut theta_a = vec![F128::zero(); 1600];
        theta.apply(&a, &mut theta_a);

        let b : Vec<_> = (0..1600).map(|_| F128::rand(rng)).collect();
        let mut theta_t_b = vec![F128::zero(); 1600];
        theta.apply_transposed(&b, &mut theta_t_b);
    
        let lhs = theta_a.iter().zip(b.iter()).map(|(a, b)| *a * b).fold(F128::zero(), |a, b| a + b);
        let rhs = theta_t_b.iter().zip(a.iter()).map(|(a, b)| *a * b).fold(F128::zero(), |a, b| a + b);
    
        assert!(lhs == rhs);
    }

    #[test]
    fn rhopi_transpose() -> () {
        let rng = &mut OsRng;
        let rhopi = RhoPiMatrix{};

        assert!(rhopi.n_in() == 1600);
        assert!(rhopi.n_out() == 1600);

        let a : Vec<_> = (0..1600).map(|_| F128::rand(rng)).collect();
        let mut theta_a = vec![F128::zero(); 1600];
        rhopi.apply(&a, &mut theta_a);

        let b : Vec<_> = (0..1600).map(|_| F128::rand(rng)).collect();
        let mut theta_t_b = vec![F128::zero(); 1600];
        rhopi.apply_transposed(&b, &mut theta_t_b);
    
        let lhs = theta_a.iter().zip(b.iter()).map(|(a, b)| *a * b).fold(F128::zero(), |a, b| a + b);
        let rhs = theta_t_b.iter().zip(a.iter()).map(|(a, b)| *a * b).fold(F128::zero(), |a, b| a + b);
    
        assert!(lhs == rhs);
    }
}
// This module describes matrices used in linear rounds of keccak.
// Each linear operator acts on 5 polynomials, each split into batches of size 1024.
// 
// batch of size 1024 is represented as 3 batches of size 320, and the tail of size 64.
// Matrices treat each batch x 5 as the state space of keccak (1600 bits), and the tail is zeroized.
//
// As we are actually acting on F128-elements, all our operations are 128-vectorized by default.

use num_traits::Zero;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::{field::F128, protocols::lincheck::{Composition, IdentityMatrix, LinOp, MatrixSum}, ptr_utils::{AsSharedMutPtr, UnsafeIndexRawMut}};

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


/// 1600 x 1600 matrix, implementing linear operations of keccak.
pub struct KeccakLinMatrixUnbatched {
    m: Composition<RhoPiMatrix, ThetaMatrix>,
}

impl KeccakLinMatrixUnbatched {
    pub fn new() -> Self {
        Self { m: Composition::new(RhoPiMatrix{}, ThetaMatrix::new()) }
    }
}

impl LinOp for KeccakLinMatrixUnbatched {
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

/// 5*1024 x 5*1024 matrix, implementing the real layout of our Keccak linear layer.
/// each 1024 batch is split into 3 pieces of size 320, which are united together into 3 vectors of size 1600 (
/// and on these we act with Keccak linear transforms).
/// The remaining tail of size 64 is filled with zeros.
pub struct KeccakLinMatrix {
    m: KeccakLinMatrixUnbatched,
}

impl KeccakLinMatrix {
    pub fn new() -> Self {
        Self { m: KeccakLinMatrixUnbatched::new() }
    }
}

impl LinOp for KeccakLinMatrix {
    fn n_in(&self) -> usize {
        5 * 1024
    }

    fn n_out(&self) -> usize {
        5 * 1024
    }

    fn apply(&self, input: &[F128], output: &mut [F128]) {
        let mut state = vec![vec![F128::zero(); 1600]; 3];
        for i in 0..5 {
            for j in 0..3 {
                state[j][i * 320 .. (i + 1) * 320].copy_from_slice(
                    &input[i * 1024 + j * 320 .. i * 1024 + (j + 1) * 320]
                );
            }
        }

        let mut output_state = vec![vec![F128::zero(); 1600]; 3];

        for j in 0..3 {
            self.m.apply(&state[j], &mut output_state[j]);
        }

        for i in 0..5 {
            for j in 0..3 {
                output[i * 1024 + j * 320 .. i * 1024 + (j + 1) * 320].copy_from_slice(
                    &output_state[j][i * 320 .. (i + 1) * 320]
                );
            }
        }

    }

    fn apply_transposed(&self, input: &[F128], output: &mut [F128]) {
        let mut state = vec![vec![F128::zero(); 1600]; 3];
        for i in 0..5 {
            for j in 0..3 {
                state[j][i * 320 .. (i + 1) * 320].copy_from_slice(
                    &input[i * 1024 + j * 320 .. i * 1024 + (j + 1) * 320]
                );
            }
        }

        let mut output_state = vec![vec![F128::zero(); 1600]; 3];

        for j in 0..3 {
            self.m.apply_transposed(&state[j], &mut output_state[j]);
        }

        for i in 0..5 {
            for j in 0..3 {
                output[i * 1024 + j * 320 .. i * 1024 + (j + 1) * 320].copy_from_slice(
                    &output_state[j][i * 320 .. (i + 1) * 320]
                );
            }
        }

    }
}

pub fn keccak_linround_witness_into(input: [&[F128]; 5], output: &mut [Vec<F128>; 5]) {
    let l = input[0].len();
    for i in 1..5 {
        assert!(input[i].len() == l);
        assert!(output[i].len() == l);
    }
    assert!(output[0].len() == l);
    assert!(l % 1024 == 0);

    let nbatches = l / 1024;
    let output_ptrs : Vec<_> = output.iter_mut().map(|arr| arr.as_shared_mut_ptr()).collect();

    #[cfg(not(feature = "parallel"))]
    let iter = 0 .. nbatches;

    #[cfg(feature = "parallel")]
    let iter = (0 .. nbatches).into_par_iter().with_min_len(8);

    iter.for_each(|batch_index| {
        let batch_offset = batch_index * 1024;

        for i in 0..5 {
            for z in 960..1024 {
                unsafe {
                    *output_ptrs[i].get_mut(batch_offset + z) = F128::zero();
                }
            }
        }

        for plane in 0..3 {
            let plane_offset = batch_offset + plane * 320;
            let mut c = [[F128::zero(); 64]; 5];
            let mut d = [[F128::zero(); 64]; 5];

            for x in 0..5 {
                for y in 0..5 {
                    for z in 0..64 {
                        c[x][z] += input[x][plane_offset + y * 64 + z];
                    }
                }
            }

            for x in 0..5 {
                for z in 0..64 {
                    d[x][z] = c[(x + 4) % 5][z] + c[(x + 1) % 5][(z + 63) % 64];
                }
            }

            for x in 0..5 {
                for y in 0..5 {
                    let out_poly = y;
                    let out_y = (2 * x + 3 * y) % 5;
                    let rotation = ROTATIONS[x][y];
                    for z in 0..64 {
                        let value = input[x][plane_offset + y * 64 + z] + d[x][z];
                        let out_z = (z + rotation) % 64;
                        unsafe {
                            *output_ptrs[out_poly].get_mut(plane_offset + out_y * 64 + out_z) = value;
                        }
                    }
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use itertools::Itertools;
    use num_traits::Zero;
    use rand::rngs::OsRng;
    use crate::{protocols::{lincheck::{Lincheck, LincheckOutput}, utils::{evaluate, evaluate_univar}}, traits::SumcheckObject};

    use super::*;

    #[test]
    fn keccak_matrix_ok() -> () {
        let rng = &mut OsRng;
        let m = KeccakLinMatrix::new();

        assert!(m.n_in() == 1024 * 5);
        assert!(m.n_out() == 1024 * 5);

        let a : Vec<_> = (0..1024 * 5).map(|_| F128::rand(rng)).collect();
        let mut m_a = vec![F128::zero(); 1024 * 5];
        m.apply(&a, &mut m_a);

        let b : Vec<_> = (0..1024 * 5).map(|_| F128::rand(rng)).collect();
        let mut m_t_b = vec![F128::zero(); 1024 * 5];
        m.apply_transposed(&b, &mut m_t_b);
    
        let lhs = m_a.iter().zip(b.iter()).map(|(a, b)| *a * b).fold(F128::zero(), |a, b| a + b);
        let rhs = m_t_b.iter().zip(a.iter()).map(|(a, b)| *a * b).fold(F128::zero(), |a, b| a + b);
    
        assert!(lhs == rhs);
    }

    #[test]

    fn keccak_lincheck_ok() {
        let rng = &mut OsRng;

        let num_vars = 20;
        let num_active_vars = 10;

        
        let m = KeccakLinMatrix::new();

        let pt : Vec<_> = (0..num_vars).map(|_| F128::rand(rng)).collect();
        let mut polys : Vec<Vec<_>> = vec![];
        for i in 0..5 {
            polys.push((0 .. 1 << num_vars).map(|_| F128::rand(rng)).collect())
        };
        let polys : [Vec<F128>; 5] = polys.try_into().unwrap();

        let polys_refs = polys.iter().map(|x| x.as_slice()).collect::<Vec<_>>().try_into().unwrap();
        let mut m_p : [Vec<F128>; 5] = (0..5).map(|_| vec![F128::zero(); 1 << num_vars]).collect::<Vec<_>>().try_into().unwrap();

        let label0 = Instant::now();

        keccak_linround_witness_into(polys_refs, &mut m_p);

        let mut eq = vec![F128::zero(); 1 << pt.len()];
        let initial_claims : [_; 5] = (0..5).map(|i| evaluate(&m_p[i], &pt, &mut eq)).collect::<Vec<_>>().try_into().unwrap();

        let label1 = Instant::now();

        let label2 = Instant::now();

        let prover = Lincheck::<5, 5, _>::new(&polys, &pt, m, num_active_vars, initial_claims);


        let gamma = F128::rand(rng);
        let chunk = 1 << num_active_vars;
        let eq_dormant = vec![F128::zero(); 1 << (pt.len() - num_active_vars)];
        let p_polys = vec![vec![F128::zero(); chunk]; 5];
        let eq = vec![F128::zero(); chunk];
        let gamma_eqs = vec![F128::zero(); 5 * chunk];
        let q = vec![F128::zero(); 5 * chunk];
        let q_polys = vec![vec![F128::zero(); chunk]; 5];
        let p_scratch = vec![vec![F128::zero(); chunk]; 5];
        let q_scratch = vec![vec![F128::zero(); chunk]; 5];
        let mut prover = prover.folding_challenge(gamma, eq_dormant, p_polys, eq, gamma_eqs, q, q_polys, p_scratch, q_scratch);

        let label3 = Instant::now();

        let mut rs = vec![];
        let mut claim = evaluate_univar(&initial_claims, gamma);

        for _ in 0..num_active_vars {
            let rpoly = prover.round_msg().coeffs(claim);
            let r = F128::rand(rng);
            claim = rpoly[0] + rpoly[1] * r + rpoly[2] * r * r;
            prover.bind(r);
            rs.push(r);
        };

        let label4 = Instant::now();

        let LincheckOutput {p_evs, q_evs} = prover.finish();

        let label5 = Instant::now();

        println!("Time elapsed: {} ms", (label5 - label0).as_millis());
        println!("> Witness gen: {} ms,", (label1 - label0).as_millis());
        println!("> Clone: {} ms", (label2 - label1).as_millis());
        println!("> Init: {} ms", (label3 - label2).as_millis());
        println!("> Prodcheck maincycle: {} ms", (label4 - label3).as_millis());
        println!("> Finish: {} ms", (label5 - label4).as_millis());

        let expected_claim = p_evs.iter()
            .zip_eq(q_evs.iter())
            .map(|(a, b)| *a * b)
            .fold(F128::zero(), |a, b| a + b);

        assert!(expected_claim == claim);


        rs.extend(pt[num_active_vars..].iter().map(|x| *x));
        assert!(rs.len() == num_vars);

        for i in 0..5 {
            let mut eq = vec![F128::zero(); 1 << rs.len()];
            assert!(p_evs[i] == evaluate(&polys[i], &rs, &mut eq));
        }

    }

}

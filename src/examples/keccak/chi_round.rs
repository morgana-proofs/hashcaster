use num_traits::{One, Zero};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{field::F128, protocols::boolcheck::FnPackage, ptr_utils::{AsSharedMUMutPtr, UninitArr, UnsafeIndexRawMut}};

/// 111111111...1 in standard basis
fn neg(x: F128) -> F128 {
    F128::from_raw(0u128.wrapping_sub(1)) + x
}

fn chi_compressed(arg : [F128; 5]) -> [F128; 5] {
    [
        arg[0] + (neg(arg[1]) & arg[2]),
        arg[1] + (neg(arg[2]) & arg[3]),
        arg[2] + (neg(arg[3]) & arg[4]),
        arg[3] + (neg(arg[4]) & arg[0]),
        arg[4] + (neg(arg[0]) & arg[1]),
    ]
}

fn chi_lin_compressed(arg : [F128; 5]) -> [F128; 5] {
    [
        arg[0] + arg[2],
        arg[1] + arg[3],
        arg[2] + arg[4],
        arg[3] + arg[0],
        arg[4] + arg[1],
    ]
}


fn chi_quad_compressed(arg : [F128; 5]) -> [F128; 5] {
    [
        arg[1] & arg[2],
        arg[2] & arg[3],
        arg[3] & arg[4],
        arg[4] & arg[0],
        arg[0] & arg[1],        
    ]
}

fn chi_algebraic(data: &[F128], mut idx_a: usize, offset: usize) -> [[F128; 5]; 3] {
    idx_a *= 2;
    let mut idxs = [
        idx_a,
        idx_a + offset * 128,
        idx_a + 2 * offset * 128,
        idx_a + 3 * offset * 128,
        idx_a + 4 * offset * 128
    ];

    let mut ret = [[F128::zero(); 5]; 3];

    for i in 0..128 {

        ret[0][0] += F128::basis(i) * (data[idxs[0]] + (F128::one() + data[idxs[1]]) * data[idxs[2]]);
        ret[0][1] += F128::basis(i) * (data[idxs[1]] + (F128::one() + data[idxs[2]]) * data[idxs[3]]);
        ret[0][2] += F128::basis(i) * (data[idxs[2]] + (F128::one() + data[idxs[3]]) * data[idxs[4]]);
        ret[0][3] += F128::basis(i) * (data[idxs[3]] + (F128::one() + data[idxs[4]]) * data[idxs[0]]);
        ret[0][4] += F128::basis(i) * (data[idxs[4]] + (F128::one() + data[idxs[0]]) * data[idxs[1]]);

        ret[1][0] += F128::basis(i) * (data[idxs[0] + 1] + (F128::one() + data[idxs[1] + 1]) * data[idxs[2] + 1]);
        ret[1][1] += F128::basis(i) * (data[idxs[1] + 1] + (F128::one() + data[idxs[2] + 1]) * data[idxs[3] + 1]);
        ret[1][2] += F128::basis(i) * (data[idxs[2] + 1] + (F128::one() + data[idxs[3] + 1]) * data[idxs[4] + 1]);
        ret[1][3] += F128::basis(i) * (data[idxs[3] + 1] + (F128::one() + data[idxs[4] + 1]) * data[idxs[0] + 1]);
        ret[1][4] += F128::basis(i) * (data[idxs[4] + 1] + (F128::one() + data[idxs[0] + 1]) * data[idxs[1] + 1]);

        ret[2][0] += F128::basis(i) * (
            (data[idxs[1]] + data[idxs[1] + 1]) * (data[idxs[2]] + data[idxs[2] + 1])
        );
        ret[2][1] += F128::basis(i) * (
            (data[idxs[2]] + data[idxs[2] + 1]) * (data[idxs[3]] + data[idxs[3] + 1])
        );
        ret[2][2] += F128::basis(i) * (
            (data[idxs[3]] + data[idxs[3] + 1]) * (data[idxs[4]] + data[idxs[4] + 1])
        );
        ret[2][3] += F128::basis(i) * (
            (data[idxs[4]] + data[idxs[4] + 1]) * (data[idxs[0]] + data[idxs[0] + 1])
        );
        ret[2][4] += F128::basis(i) * (
            (data[idxs[0]] + data[idxs[0] + 1]) * (data[idxs[1]] + data[idxs[1] + 1])
        );


        for j in 0..5 {
            idxs[j] += offset;
        }

    }

    ret
}

pub struct ChiPackage {}

impl FnPackage<5, 5> for ChiPackage {
    fn exec_lin_compressed(&self, arg: [F128; 5]) -> [F128; 5] {
        chi_lin_compressed(arg)
    }

    fn exec_quad_compressed(&self, arg: [F128; 5]) -> [F128; 5] {
        chi_quad_compressed(arg)
    }

    fn exec_alg(&self, data: &[F128], start: usize, offset: usize) -> [[F128; 5]; 3] {
        chi_algebraic(data, start, offset)
    }
}

pub fn chi_round_witness(polys: &[Vec<F128>; 5]) -> [Vec<F128>; 5] {
    let l = polys[0].len();
    for i in 1..5 {
        assert!(polys[i].len() == l);
    }

    let mut ret = vec![];
    for i in 0..5 {
        ret.push(UninitArr::<F128>::new(l));
    }

    let ret_ptrs : Vec<_> = ret.iter_mut().map(|arr| arr.as_shared_mut_ptr()).collect();

    #[cfg(not(feature = "parallel"))]
    let iter = (0..l);

    #[cfg(feature = "parallel")]
    let iter = (0..l).into_par_iter();

    iter.map(|i| {
        let tmp = chi_compressed([polys[0][i], polys[1][i], polys[2][i], polys[3][i], polys[4][i]]);
        unsafe{ 
            *ret_ptrs[0].get_mut(i) = tmp[0];
            *ret_ptrs[1].get_mut(i) = tmp[1];
            *ret_ptrs[2].get_mut(i) = tmp[2];
            *ret_ptrs[3].get_mut(i) = tmp[3];
            *ret_ptrs[4].get_mut(i) = tmp[4];
        }
    }).count();

    ret.into_iter().map(|arr| unsafe{arr.assume_init()}).collect::<Vec<_>>().try_into().unwrap()
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use itertools::Itertools;
    use num_traits::One;
    use rand::rngs::OsRng;

    use crate::{protocols::{boolcheck::{BoolCheck, BoolCheckOutput}, utils::{compute_trit_mappings, eq_ev, evaluate, evaluate_univar, extend_n_tables, untwist_evals}}, traits::SumcheckObject, utils::u128_idx};

    use super::*;

    #[test]

    fn chi_compressed_correct() {
        let rng = &mut OsRng;
        let input : [F128; 5] = (0..5).map(|_| F128::rand(rng)).collect::<Vec<_>>().try_into().unwrap();

        let quad_o = chi_quad_compressed(input);
        let quad_l = chi_lin_compressed(input);

        let lhs : [F128; 5] = (0..5).map(|i| quad_o[i] + quad_l[i]).collect::<Vec<_>>().try_into().unwrap();
        let rhs = chi_compressed(input);

        assert!(lhs == rhs);
    }

    #[test]
    fn chi_alg_correct() {
        let rng = &mut OsRng;

        let input_a : [F128; 5] = (0..5).map(|_| F128::rand(rng)).collect::<Vec<_>>().try_into().unwrap();
        let input_b : [F128; 5] = (0..5).map(|_| F128::rand(rng)).collect::<Vec<_>>().try_into().unwrap();

        let lhs_a = chi_compressed(input_a);
        let lhs_b = chi_compressed(input_b);
        let lhs_ab = chi_compressed(input_a.iter().zip(input_b.iter()).map(|(a, b)| *a + b).collect::<Vec<_>>().try_into().unwrap());

        let data = input_a.iter().map(|x| {
            (0..128).map(|i| {
                F128::new(u128_idx(&x.raw(), i))
            })
        }).flatten().interleave(input_b.iter().map(|x| {
            (0..128).map(|i| {
                F128::new(u128_idx(&x.raw(), i))
            })
        }).flatten())
        .collect::<Vec<_>>();
    
        let rhs = chi_algebraic(&data, 0, 2);

        assert!(rhs[0] == lhs_a);
        assert!(rhs[1] == lhs_b);
        assert!(rhs[2] == lhs_ab);
    }


    #[test]
    fn chi_round_test() {
        let rng = &mut OsRng;
        let num_vars = 20;
        let c = 5;

        let pt : Vec<F128> = (0..num_vars).map(|_| F128::rand(rng)).collect();
    
        let mut polys : Vec<Vec<F128>> = vec![];
        for _ in 0..5 {
            polys.push((0 .. 1 << num_vars).map(|_| F128::rand(rng)).collect());
        }
        let polys : [Vec<F128>; 5] = polys.try_into().unwrap();
    
//        let (bit_mapping, trit_mapping) = compute_trit_mappings(c);
        // let expected_ext = (0..5).map(|i| {
        //     let polys : Vec<&[F128]>= polys.iter().map(|v|v.as_slice()).collect();
        //     let f = |arg: [F128; 5]| {
        //         chi_compressed(arg)[i]
        //     };
        //     extend_n_tables(&polys, c, &trit_mapping, &f)
        // }).collect::<Vec<_>>();

        let output = chi_round_witness(&polys);

        let evaluation_claims : [F128; 5] = output.iter().map(|poly| evaluate(&poly, &pt)).collect::<Vec<F128>>().try_into().unwrap();
        let f = ChiPackage{};

        let start = Instant::now();
 
        let prover = BoolCheck::new(
            f,
            polys.clone(), 
            c,
            evaluation_claims,
            pt.clone()
        );


        let gamma = F128::rand(rng);
        let mut prover = prover.folding_challenge(gamma);

        // let ext_l = expected_ext[0].len();
        // let expected_ext = (0..ext_l).map(|i| {
        //     evaluate_univar(
        //         &(0..5).map(|j| expected_ext[j][i]).collect::<Vec<_>>(),
        //         gamma
        //     )
        // }).collect::<Vec<_>>();

        // assert!(&expected_ext == prover.ext.as_ref().unwrap());

        // Initialize expected (folded) claim.
        let mut claim = evaluate_univar(&evaluation_claims, gamma);

        assert!(claim == prover.claim, "Failed at entry");

        let mut rs = vec![];

        for i in 0..num_vars {
            let rpoly = prover.round_msg().coeffs(claim);

            let r = F128::rand(rng);
            assert!(rpoly.len() == 4);
            claim = evaluate_univar(&rpoly, r);
            prover.bind(r);
            rs.push(r);

            assert!(claim == prover.claim, "Failed after round {}", i);
        }

        let BoolCheckOutput { mut frob_evals, .. } = prover.finish();

        let end = Instant::now();

        assert!(frob_evals.len() == 128 * 5);
        frob_evals.chunks_mut(128).map(|chunk| untwist_evals(chunk)).count();

        let expected_coord_evals : Vec<_> = polys.iter().map(|poly| {
            (0..128).map(|i| {
                let poly_i : Vec<F128> = poly.iter().map(|value| {
                    F128::new(u128_idx(&value.raw(), i))
                }).collect();
                evaluate(&poly_i, &rs)
            })
        }).flatten().collect();

        assert!(expected_coord_evals.len() == 128 * 5);
        assert!(expected_coord_evals == frob_evals);

        frob_evals.push(F128::zero());

        let claimed_ev = chi_algebraic(&frob_evals, 0, 1)[0];
        let folded_claimed_ev = evaluate_univar(&claimed_ev, gamma);

        assert!(folded_claimed_ev * eq_ev(&pt, &rs) == claim);

        println!("Time elapsed: {} ms", (end-start).as_millis());
    }

}
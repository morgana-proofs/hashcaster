use std::{sync::atomic::{AtomicU64, Ordering}, time::Instant};

use bytemuck::{cast, cast_slice};
use num_traits::{One, Zero};
use rayon::{iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator}, slice::{ParallelSlice, ParallelSliceMut}};
use crate::{
    backend::autodetect::{v_movemask_epi8, v_slli_epi64}, field::{pi, F128}, precompute::frobenius_table::FROBENIUS, ptr_utils::{AsSharedConstPtr, AsSharedMutPtr, UnsafeIndexRaw, UnsafeIndexRawMut}, utils::{log2_exact, u128_idx}
};
use itertools::Itertools;

pub fn inv_frob_orbit(r: &[F128]) -> Vec<Vec<F128>> {
    let mut inverse_orbit = vec![];
    let mut r = r.to_vec();
    for _ in 0..128 {
        r.iter_mut().map(|x| *x *= *x).count();
        inverse_orbit.push(r.clone());
    }
    inverse_orbit.reverse();
    inverse_orbit
}

/// Given evaluations of coordinate polynomials P_i (r), return evaluations of P in inverse Frobenius orbit of r.
pub fn twist_evals(evals: &mut [F128]) {
    let mut twisted_evals = vec![];
    for _ in 0..128 {
        evals.iter_mut().map(|x| *x *= *x).count();
        twisted_evals.push(
            (0..128).map(|i| {
                F128::basis(i) * evals[i]
            }).fold(F128::zero(), |a, b| a + b)
        );
    }
    twisted_evals.reverse();
    evals.clone_from_slice(&twisted_evals);
}

/// Given evaluations of a polynomial P in inverse Frobenius orbit r, compute evaluations of P_i in r.
/// TODO: rewrite more efficiently, .frob(i) is not good at all. Though right now I don't care about
/// verifier that much.
pub fn untwist_evals(twisted_evals: &mut [F128]) {
    for i in 0..128 {
        twisted_evals[i] = twisted_evals[i].frob(i as i32);
    }

    let untwisted : Vec<_> = (0..128).map(|i| pi(i, &twisted_evals)).collect();
    twisted_evals.copy_from_slice(&untwisted);
}

pub fn eq_poly(pt: &[F128], ret: &mut [F128]) {
    let l = pt.len();
    assert!(ret.len() == 1 << l);
    let ptr = ret.as_shared_mut_ptr();
    unsafe{
        *ptr.get_mut(0) = F128::one();
        for i in 0..l {
            let half = 1 << i;
            #[cfg(not(feature = "parallel"))]
            let iter = (0 .. half).into_iter();

            #[cfg(feature = "parallel")]
            let iter = (0 .. half).into_par_iter();
            iter.map(|j| {
                *ptr.get_mut(j + half) = pt[i] * *ptr.get(j);
                *ptr.get_mut(j) += *ptr.get(j + half);
            }).count();
        }
    }
}

pub fn eq_ev(x: &[F128], y: &[F128]) -> F128 {
    x.iter().zip_eq(y.iter()).fold(F128::one(), |acc, (x, y)| acc * (F128::one() + x + y))
}

pub fn evaluate(poly: &[F128], pt: &[F128], eq: &mut [F128]) -> F128 {
    assert!(poly.len() == 1 << pt.len());
    eq_poly(pt, eq);
    #[cfg(not(feature = "parallel"))]
    let ret = poly.iter().zip_eq(eq.iter()).fold(F128::zero(), |acc, (x, y)| acc + *x * *y);
    #[cfg(feature = "parallel")]
    let ret = poly.par_iter().zip(eq.par_iter()).map(|(x, y)| *x * *y).reduce(||F128::zero(), |a, b| a + b);
    ret
}

pub fn evaluate_univar(poly: &[F128], at: F128) -> F128 {
    let l = poly.len();
    let mut ret = poly[l-1];
    for i in 0..l-1 {
        ret *= at;
        ret += poly[l-2-i];
    }
    ret
}

pub fn bits_to_trits(mut x: usize) -> usize {
    let mut multiplier = 1;
    let mut ret = 0;
    while x > 0 {
        ret += multiplier * (x % 2);
        x >>= 1;
        multiplier *= 3;
    }
    ret
}

/// Returns some useful mappings.
/// Trit mapping is an array of size 3^(c+1). For an index which has a digit 2 in ternary,
/// it has a value = 3^k, where k is index of the corresponding (largest) digit. I.e. i - trit_mapping[i]
/// will set this digit to 1, and i - 2*trit_mapping[i] will set this digit to 0.
/// 
/// For values not of this form, it returns 2 * binary number with the same digits (multiplier 2 is added
/// to always differentiate from the first case). bin_mapping is a reverse map, i.e. it maps number with some
/// binary decomposition into ternary number with same digits.
pub fn compute_trit_mappings(c: usize)  -> (Vec<u16>, Vec<u16>) {
    let pow3 = 3usize.pow((c+1) as u32);
    
    let mut trits = vec![0u8; c + 1];

    let mut bit_mapping = Vec::<u16>::with_capacity(1 << (c + 1));
    let mut trit_mapping = Vec::<u16>::with_capacity(pow3);
    
    let mut i = 0;
    loop {
        let mut bin_value = 0u16;
        let mut j = c;
        let mut flag = true;
        let mut bad_offset = 1u16;
        loop {
            if flag {
                bad_offset *= 3;
            }
            bin_value *= 2;
            if trits[j] == 2 {
                flag = false;
            } else {
                bin_value += trits[j] as u16;
            }

            if j == 0 {break}
            j -= 1;
        }
        if flag {
            trit_mapping.push(bin_value << 1);
            bit_mapping.push(i as u16);
        } else {
            trit_mapping.push(pow3 as u16 / bad_offset);
        }

        i += 1;
        if i == pow3 {
            break;
        }
        // add 1 to trits
        // this would go out of bounds for (2, 2, 2, 2, ..., 2) but this never happens because we leave
        // the main cycle before this
        let mut j = 0;
        loop {
            if trits[j] < 2 {
                trits[j] += 1;
                break;
            } else {
                trits[j] = 0;
                j += 1;
            }
        }
    }

    (bit_mapping, trit_mapping)
}

/// Extends n tables and applies a formula F to them.
/// Warning: n must be low enough, or you will get a lot of cache misses (each table_ext consumes ~ 3^c * 32 bytes of cache).
/// For standard value of c, it gives 8Kb.
/// Depending on cache size, one should switch to separate extension if amount of tables is too large, for 128kb cache
/// I recommend doing it for > ~10, to be on a safe side.
pub fn extend_n_tables<
    const N: usize,
    F_LIN: Fn([F128; N]) -> F128 + Send + Sync,
    F_QUAD: Fn([F128; N]) -> F128 + Send + Sync,
>(
    tables: &[&[F128]],
    c: usize,
    trit_mapping: &[u16],
    f_lin: F_LIN,
    f_quad: F_QUAD,
    ret: &mut [F128],
    tables_ext: &mut [Vec<F128>],
) {
    assert!(tables.len() == N);
    let dims = log2_exact(tables[0].len());
    for table in tables {
        assert!(table.len() == 1 << dims);
    }
    assert!(c < dims);
    let pow3 = 3usize.pow((c + 1) as u32);
    let pow3_adj = pow3 / 3 * 2;
    assert!(pow3 < (1 << 15) as usize, "This is too large anyway ;)");
    let pow2 = 2usize.pow((dims - c - 1) as u32);
    assert!(ret.len() == pow3 * pow2);
    assert!(tables_ext.len() % N == 0);
    for table_ext in tables_ext.iter() {
        assert!(table_ext.len() == pow3_adj);
    }

    // And we don't have multizip, so I guess I'm gonna write it with raw accesses once again. shrug

    unsafe{
        let table_ptrs_ : Vec<_> = tables.iter().map(|&table| table.as_shared_ptr()).collect();
        let table_ptrs = table_ptrs_.as_shared_ptr();
        let mut table_ext_ptrs_ : Vec<_> = tables_ext.iter_mut().map(|table| table.as_shared_mut_ptr()).collect();
        let table_ext_ptrs = table_ext_ptrs_.as_shared_mut_ptr();
    
        let ret_ptr = ret.as_shared_mut_ptr();

        // We have pow2 chunks in total - in tables, they are of size (1 << (c + 1)), in ret they are of size pow3
        // and in extended tables they are of size (2/3) * pow3, which is pow3_adj.

        #[cfg(not(feature = "parallel"))]
        let chunk_id_iter = (0..pow2);
        #[cfg(feature = "parallel")]
        let chunk_id_iter = (0..pow2).into_par_iter().with_min_len(64);

        chunk_id_iter.map(|chunk_id| {
            let mut args = [F128::zero(); N]; 

            let global_tab_offset = chunk_id * (1 << (c+1));
            let global_ret_offset = chunk_id * pow3;
            #[cfg(not(feature = "parallel"))]
            let scratch_offset = 0;
            #[cfg(feature = "parallel")]
            let scratch_offset = rayon::current_thread_index().unwrap() * N;
            assert!(scratch_offset + N <= tables_ext.len());

            for j in 0..pow3_adj {
//                println!("entry, j = {}", j);
                let offset = trit_mapping[j] as usize;
                if offset % 2 == 0 {
                    for z in 0..N {
                        let table_z = *table_ptrs.get(z);
                        let table_ext_z = *table_ext_ptrs.get_mut(scratch_offset + z); 
                        let value = *table_z.get(global_tab_offset + (offset >> 1));
                        *table_ext_z.get_mut(j) = value;
                        args[z] = value;
                    }
                    *ret_ptr.get_mut(global_ret_offset + j) = f_quad(args) + f_lin(args);
                } else {
                    for z in 0..N {
                        let table_ext_z = *(table_ext_ptrs.get(scratch_offset + z));
                        let value =
                            (*table_ext_z.get(j - offset)) +
                            (*table_ext_z.get(j - 2 * offset));
                        *table_ext_z.get_mut(j) = value;
                        args[z] = value;
                    }
                    *ret_ptr.get_mut(global_ret_offset + j) = f_quad(args);
                }
            }

            for j in pow3_adj..pow3 {
                let offset = trit_mapping[j] as usize;
                for z in 0..N {
                    let table_ext_z = *(table_ext_ptrs.get_mut(scratch_offset + z));
                    args[z] =
                        (*table_ext_z.get(j - offset)) +
                        (*table_ext_z.get(j - 2 * offset))
                    ;
                }
                *ret_ptr.get_mut(global_ret_offset + j) = f_quad(args);
            }
        }).count();
        
    }
}


//#[unroll::unroll_for_loops]
pub fn drop_top_bit(x: usize) -> (usize, usize) {
    let mut s = 0;
    for i in 0..8 {
        let bit = (x >> i) % 2;
        s = i * bit + s * (1 - bit);
    }
    (x - (1 << s), s)
}

//#[unroll::unroll_for_loops]
/// Restricts all coordinates of all polynomials and writes them in one contiguous array.
pub fn restrict(polys: &[&[F128]], coords: &[F128], dims: usize, eq: &mut [F128], eq_sums: &mut [F128], ret: &mut [F128]) {
    let n = polys.len();
    for poly in polys.iter() {
        assert!(poly.len() == 1 << dims);
    }
    assert!(coords.len() <= dims);

    let chunk_size = (1 << coords.len());
    let num_chunks = 1 << (dims - coords.len());
    assert!(ret.len() == num_chunks * 128 * n);
    assert!(eq.len() == 1 << coords.len());
    assert!(eq_sums.len() == 256 * eq.len() / 8);

    eq_poly(coords, eq);

    assert!(eq.len() % 16 == 0, "Technical condition for now.");

    for i in 0..eq.len()/8 {
        eq_sums[i * 256] = F128::zero();
        for j in 1..256 {
            let (sum_idx, eq_idx) = drop_top_bit(j);
            let tmp = eq[i * 8 + eq_idx] + eq_sums[i * 256 + sum_idx];
            eq_sums[i * 256 + j] = tmp;
        }
    }

    let ret_ptr = ret.as_shared_mut_ptr();

    const TILE: usize = 8;
    let num_tiles = (num_chunks + TILE - 1) / TILE;

    for q in 0..n {
        #[cfg(not(feature = "parallel"))]
        let iter = (0..num_tiles).into_iter();
        #[cfg(feature = "parallel")]
        let iter = (0..num_tiles).into_par_iter().with_min_len(4);
        iter.map(|tile| {
            let start = tile * TILE;
            let len = (num_chunks - start).min(TILE);
            let mut acc = [[F128::zero(); 128]; TILE];

            for b in 0..len {
                let i = start + b;
                for j in 0 .. eq.len() / 16 { // Step by 16
                    let v0 = &eq_sums[j * 512 .. j * 512 + 256];
                    let v1 = &eq_sums[j * 512 + 256 .. j * 512 + 512];
                    let bytearr = cast_slice::<F128, [u8; 16]>(
                        &polys[q][i * chunk_size + j * 16 .. i * chunk_size + (j + 1) * 16]
                    );

                    // Iteration over bytes
                    for s in 0..16 {
                        let mut t = [
                            bytearr[0][s], bytearr[1][s], bytearr[2][s], bytearr[3][s],
                            bytearr[4][s], bytearr[5][s], bytearr[6][s], bytearr[7][s],
                            bytearr[8][s], bytearr[9][s], bytearr[10][s], bytearr[11][s],
                            bytearr[12][s], bytearr[13][s], bytearr[14][s], bytearr[15][s],
                        ];

                        for u in 0..8 {
                            let bits = v_movemask_epi8(t) as u16;
                            acc[b][s*8 + 7 - u] += v0[(bits & 255) as usize];
                            acc[b][s*8 + 7 - u] += v1[((bits >> 8) & 255) as usize];
                            t = v_slli_epi64::<1>(t);
                        }
                    }
                }
            }

            unsafe {
                for coord in 0..128 {
                    let out = (coord + q * 128) * num_chunks + start;
                    for b in 0..len {
                        *ret_ptr.get_mut(out + b) = acc[b][coord];
                    }
                }
            }
        }
        ).count();
    }
}

/// This implements efficient matrices using method of 4 Russians, 128x128.
/// Technically we could implements 128 x N, and use in restrict, but I will avoid it for now. 
#[derive(Clone, Debug)]
pub struct EfficientMatrix {
    precomp: Vec<F128>, // array of size 256 * 16, containing all results for each byte.
}

impl EfficientMatrix {
    pub fn new_from_rows(rows: &[F128]) -> Self {
        let mut cols = Vec::with_capacity(128 * 2);
        for _ in 0..128*2 {
            cols.push(AtomicU64::new(0));
        }

        assert!(rows.len() == 128);
        let rows = cast_slice::<F128, [u8; 16]>(rows);

        let iter = rows.par_chunks(16);

        iter.enumerate().map(|(chunk_idx, chunk)|{
            let idx_u64 = chunk_idx / 4;
            let shift = 16 * (chunk_idx % 4);

            let mut t;
            for i in 0..16 {
                t = 
                    [
                        chunk[0][i], chunk[1][i], chunk[2][i], chunk[3][i],
                        chunk[4][i], chunk[5][i], chunk[6][i], chunk[7][i],
                        chunk[8][i], chunk[9][i], chunk[10][i], chunk[11][i],
                        chunk[12][i], chunk[13][i], chunk[14][i], chunk[15][i],
                    ];
                for j in 0..8 {
                    let bits = (v_movemask_epi8(t) as u64) << shift;
                    cols[2 * (8 * i + 7 - j) + idx_u64].fetch_xor(bits, Ordering::Relaxed);
                    t = v_slli_epi64::<1>(t);
                }
            }
        }).count();

        let cols : Vec<u64> = cols.iter().map(|x| x.load(Ordering::Relaxed)).collect();

        let cols = cast_slice::<u64, F128>(&cols);

        Self::new_from_cols(cols)
    }

    pub fn new_from_cols(cols: &[F128]) -> Self {
        assert!(cols.len() == 128);
        let mut precomp = vec![F128::zero(); 256 * 16];

        #[cfg(not(feature = "parallel"))]
        let row_iter = cols.chunks(8);
        #[cfg(not(feature = "parallel"))]
        let sums_iter = precomp.chunks_mut(256);

        #[cfg(feature = "parallel")]
        let row_iter = cols.par_chunks(8);
        #[cfg(feature = "parallel")]
        let sums_iter = precomp.par_chunks_mut(256);

        row_iter.zip(sums_iter).map(|(cols, sums)| {
            sums[0] = F128::zero();
            for i in 1..256 {
                let (sum_idx, row_idx) = drop_top_bit(i);
                sums[i] = sums[sum_idx] + cols[row_idx];
            }
        }).count();

        Self{precomp}
    }

    pub fn apply(&self, elt: F128) -> F128 {
        let elt = cast::<F128, [u8; 16]>(elt);
        let mut ret = self.precomp[elt[0] as usize];
        for i in 1..16 {
            ret += self.precomp[elt[i] as usize + 256 * i]
        }
        ret
    }
}


/// Creates matrix sum_i gamma_i Fr^i 
pub fn frobenius_lc(gammas: &[F128]) -> EfficientMatrix{
    assert!(gammas.len() == 128);
    
    let mut ret = vec![F128::zero(); 128];

    for i in 0..128 {
        for j in 0..128 {
            ret[j] += gammas[i] * F128::from_raw(FROBENIUS[i][j])
        }
    };

    EfficientMatrix::new_from_cols(&ret)
}

pub fn frobenius_inv_lc(gammas: &[F128]) -> EfficientMatrix{
    assert!(gammas.len() == 128);
    
    let mut ret = vec![F128::zero(); 128];

    for i in 0..128 {
        let minus_i = (128 - i) % 128;
        for j in 0..128 {
            ret[j] += gammas[i] * F128::from_raw(FROBENIUS[minus_i][j])
        }
    };

    EfficientMatrix::new_from_cols(&ret)
}

#[cfg(test)]
mod tests {
    use std::iter::{once, repeat_with};

    use rand::rngs::OsRng;

    use crate::utils::Matrix;

    use super::*;

    #[test]
    fn twist_untwist() {
        let rng = &mut OsRng;
        let lhs : Vec<_> = repeat_with(|| F128::rand(rng)).take(128).collect();
        let mut rhs = lhs.clone();

        twist_evals(&mut rhs);
        untwist_evals(&mut rhs);

        assert!(lhs == rhs);

        untwist_evals(&mut rhs);
        twist_evals(&mut rhs);

        assert!(lhs == rhs);
    }

    #[test]

    fn twist_computes_expected_openings() {
        let num_vars = 10;
        let rng = &mut OsRng;
        let pt : Vec<_> = repeat_with(|| F128::rand(rng)).take(num_vars).collect();
        let poly : Vec<_> = repeat_with(|| F128::rand(rng)).take(1 << num_vars).collect();

        let mut coord_evs = vec![];

        let mut eq = vec![F128::zero(); 1 << pt.len()];
        for i in 0..128 {
            let poly_i : Vec<_> = poly.iter().map(|x| F128::new(u128_idx(&x.raw, i))).collect();
            coord_evs.push(evaluate(&poly_i, &pt, &mut eq));
        }

        let mut tmp = F128::zero();
        for i in 0..128 {
            tmp += coord_evs[i] * F128::basis(i);
        }

        assert!(tmp == evaluate(&poly, &pt, &mut eq));

        let mut pt_inv_orbit = vec![];
        for i in 0..128i32 {
            pt_inv_orbit.push(
                pt.iter().map(|x| x.frob(-i)).collect::<Vec<F128>>()
            )
        }

        let twisted_evs : Vec<_> = (0..128).map(|i| evaluate(&poly, &pt_inv_orbit[i], &mut eq)).collect();

        twist_evals(&mut coord_evs);
        assert!(twisted_evs == coord_evs);

    }

    #[test]

    fn matrices() {
        let rng = &mut OsRng;
        let cols : Vec<_> = repeat_with(|| F128::rand(rng)).take(128).collect();
        let eff = EfficientMatrix::new_from_cols(&cols);
        let naive = Matrix::new(cols.iter().map(|x| x.raw()).collect_vec());
        let test_vec = F128::rand(rng);
        assert!(naive.apply(test_vec.raw) == eff.apply(test_vec).raw);
    }

    #[test]

    fn left_mult() {
        let rng = &mut OsRng;
        let x = F128::rand(rng);
        let cols : Vec<_> = (0..128).map(|i| F128::basis(i)*x).collect();

        let mult_by_x = EfficientMatrix::new_from_cols(&cols);

        let mut lhs = F128::rand(rng);
        let mut rhs = lhs;

        let label0 = Instant::now();
        for i in 0..1<<22 {
            lhs = x * lhs
        }
        let label1 = Instant::now();
        for i in 0..1<<22 {
            rhs = mult_by_x.apply(rhs)
        }
        let label2 = Instant::now();

        println!("Normal multiplication took {} ms", (label1 - label0).as_millis());
        println!("Matrix multiplication took {} ms", (label2 - label1).as_millis());

        assert!(lhs == rhs);
    }

    #[test]
    fn frobenius_lc_as_expected() {
        let rng = &mut OsRng;
        let x = F128::rand(rng);
        let gammas : Vec<_> = (0..128).map(|_|F128::rand(rng)).collect();

        let mut lhs = F128::zero();
        for i in 0..128 {
            lhs += gammas[i] * x.frob(i as i32);
        }
        let m = frobenius_lc(&gammas);
        let rhs = m.apply(x);
        assert!(lhs == rhs);

        let mut lhs = F128::zero();
        for i in 0..128 {
            lhs += gammas[i] * x.frob(- (i as i32));
        }
        let m = frobenius_inv_lc(&gammas);
        let rhs = m.apply(x);
        assert!(lhs == rhs);
    }

}

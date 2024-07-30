use std::{mem::{MaybeUninit}, sync::atomic::{AtomicU64, Ordering}, thread::sleep, time::{Duration, Instant}};

use bytemuck::cast_slice;
use num_traits::{One, Zero};
use rayon::{iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator}, slice::{ParallelSlice, ParallelSliceMut}};
use crate::{
    backend::autodetect::{v_movemask_epi8, v_slli_epi64},
    field::{pi, F128},
    ptr_utils::{AsSharedConstPtr, AsSharedMUConstPtr, AsSharedMUMutPtr, AsSharedMutPtr, UninitArr, UnsafeIndexMut, UnsafeIndexRaw, UnsafeIndexRawMut},
    utils::log2_exact
};
use itertools::Itertools;

pub fn eq_poly(pt: &[F128]) -> Vec<F128> {
    let l = pt.len();
    let mut ret = Vec::with_capacity(1 << l);
    ret.push(F128::one());
    for i in 0..l {
//        let pt_idx = l - i - 1;
        let half = 1 << i;
        for j in 0..half {
            ret.push(pt[i] * ret[j]);
        }
        for j in 0..half{
            let tmp = ret[half + j];
            ret[j] += tmp;
        }
    }
    ret
}

pub fn eq_poly_sequence(pt: &[F128]) -> Vec<Vec<F128>> {

    let l = pt.len();
    let mut ret = Vec::with_capacity(l + 1);
    ret.push(vec![F128::one()]);

    for i in 1..(l+1) {
        let last = &ret[i-1];
        let multiplier = pt[l-i];
        let mut incoming = UninitArr::<F128>::new(1 << i);
        unsafe{
        let ptr = incoming.as_shared_mut_ptr();

            #[cfg(not(feature = "parallel"))]
            let iter = (0 .. (1 << (i-1))).into_iter();

            #[cfg(feature = "parallel")]
            let iter = (0 .. 1 << (i-1)).into_par_iter();

            iter.map(|j|{
                let w = last[j];
                let m = multiplier * w;
                * ptr.get_mut(2*j) = w + m;
                * ptr.get_mut(2*j + 1) = m;
            }).count();
            ret.push(incoming.assume_init());
        }
    }

    ret
}


pub fn eq_ev(x: &[F128], y: &[F128]) -> F128 {
    x.iter().zip_eq(y.iter()).fold(F128::one(), |acc, (x, y)| acc * (F128::one() + x + y))
}

pub fn evaluate(poly: &[F128], pt: &[F128]) -> F128 {
    assert!(poly.len() == 1 << pt.len());
    poly.iter().zip_eq(eq_poly(pt)).fold(F128::zero(), |acc, (x, y)| acc + *x * y)
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

/// Makes table 3^{c+1} * 2^{dims - c - 1}
pub fn extend_table(table: &[F128], dims: usize, c: usize, trits_mapping: &[u16]) -> Vec<F128> {
    assert!(table.len() == 1 << dims);
    assert!(c < dims);
    let pow3 = 3usize.pow((c + 1) as u32);
    assert!(pow3 < (1 << 15) as usize, "This is too large anyway ;)");
    let pow2 = 2usize.pow((dims - c - 1) as u32);
    let mut ret = UninitArr::new(pow3 * pow2);
    unsafe{

        #[cfg(feature = "parallel")]
        let tchunks = table.par_chunks(1 << (c + 1));
        #[cfg(feature = "parallel")]
        let rchunks = ret.par_chunks_mut(pow3);

        #[cfg(not(feature = "parallel"))]
        let tchunks = table.chunks(1 << (c + 1));
        #[cfg(not(feature = "parallel"))]
        let rchunks = ret.chunks_mut(pow3);


        tchunks.zip(rchunks).map(|(table_chunk, ret_chunk)| {
            for j in 0..pow3 {
                let offset = trits_mapping[j];
                if offset % 2 == 0 {
                    ret_chunk[j] = MaybeUninit::new(table_chunk[(offset >> 1) as usize]);
                } else {
                    ret_chunk[j] = MaybeUninit::new(
                        ret_chunk[j - offset as usize].assume_init()
                        + ret_chunk[j - 2 * offset as usize].assume_init()
                    );
                }
            }
        }).count();
    }
    unsafe {ret.assume_init()}
}

/// Extends two tables at the same time and ANDs them
/// Gives some advantage because we skip 1/3 of writes into p_ext and q_ext.
pub fn extend_2_tables_legacy(p: &[F128], q: &[F128], dims: usize, c: usize, trit_mapping: &[u16]) -> Vec<F128> {
    assert!(p.len() == 1 << dims);
    assert!(q.len() == 1 << dims);
    assert!(c < dims);
    let pow3 = 3usize.pow((c + 1) as u32);
    let pow3_adj = pow3 / 3 * 2;
    assert!(pow3 < (1 << 15) as usize, "This is too large anyway ;)");
    let pow2 = 2usize.pow((dims - c - 1) as u32);
    let mut p_ext = vec![MaybeUninit::uninit(); (pow3 * 2) / 3  * pow2];
    let mut q_ext = vec![MaybeUninit::uninit(); (pow3 * 2) / 3 * pow2];
    let mut ret = UninitArr::new(pow3 * pow2);

    // Slice management seems to have some small overhead at this scale, possibly replace with
    // raw pointer accesses? *Insert look what they have to do to mimic the fraction of our power meme*
    unsafe{
        #[cfg(not(feature = "parallel"))]
        let pchunks = p.chunks(1 << (c + 1));
        #[cfg(not(feature = "parallel"))]
        let qchunks = q.chunks(1 << (c + 1));
        #[cfg(not(feature = "parallel"))]
        let p_ext_chunks = p_ext.chunks_mut(pow3_adj);
        #[cfg(not(feature = "parallel"))]
        let q_ext_chunks = q_ext.chunks_mut(pow3_adj);
        #[cfg(not(feature = "parallel"))]
        let ret_chunks = ret.chunks_mut(pow3);

        #[cfg(feature = "parallel")]
        let pchunks = p.par_chunks(1 << (c + 1));
        #[cfg(feature = "parallel")]
        let qchunks = q.par_chunks(1 << (c + 1));
        #[cfg(feature = "parallel")]
        let p_ext_chunks = p_ext.par_chunks_mut(pow3_adj);
        #[cfg(feature = "parallel")]
        let q_ext_chunks = q_ext.par_chunks_mut(pow3_adj);
        #[cfg(feature = "parallel")]
        let ret_chunks = ret.par_chunks_mut(pow3);

        pchunks.zip(qchunks).zip(
        p_ext_chunks.zip(q_ext_chunks)
        ).zip(
        ret_chunks).map(|(((p, q), (p_ext, q_ext)), ret)| {
            for j in 0..pow3_adj {
                let offset = trit_mapping[j] as usize;
                if offset % 2 == 0 {
                    p_ext[j] = MaybeUninit::new(
                        p[offset >> 1]
                    );
                    q_ext[j] = MaybeUninit::new(
                        q[offset >> 1]
                    );
                } else {
                    p_ext[j] = MaybeUninit::new(
                        p_ext[j - offset].assume_init()
                        + p_ext[j - 2 * offset].assume_init()
                    );
                    q_ext[j] = MaybeUninit::new(
                        q_ext[j - offset].assume_init()
                        + q_ext[j - 2 * offset].assume_init()
                    );
                }
                ret[j] = MaybeUninit::new(p_ext[j].assume_init() & q_ext[j].assume_init())

            };
            for j in pow3_adj..pow3{
                let offset = trit_mapping[j] as usize;
                ret[j] = MaybeUninit::new(
                    (p_ext[j - offset].assume_init() + p_ext[j - 2 * offset].assume_init()) &
                    (q_ext[j - offset].assume_init() + q_ext[j - 2 * offset].assume_init())
                )
            }
        }).count();
    }
    unsafe{ret.assume_init()}
}

/// Extends n tables and applies a formula F to them.
/// Warning: n must be low enough, or you will get a lot of cache misses (each table_ext consumes ~ 3^c * 32 bytes of cache).
/// For standard value of c, it gives 8Kb.
/// Depending on cache size, one should switch to separate extension if amount of tables is too large, for 128kb cache
/// I recommend doing it for > ~10, to be on a safe side.
pub fn extend_n_tables<const N: usize, F: Fn([F128; N]) -> F128 + Send + Sync>(
    tables: &[&[F128]],
    c: usize,
    trit_mapping: &[u16],
    f: F
) -> Vec<F128> {
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

    let mut tables_ext = vec![];
    for _ in 0..N {
        tables_ext.push(UninitArr::new((pow3 * 2) / 3  * pow2))
    }

    let mut ret = UninitArr::new(pow3 * pow2);

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
        let chunk_id_iter = (0..pow2).into_par_iter();

        chunk_id_iter.map(|chunk_id| {
            let mut args = [F128::zero(); N]; 

            let global_tab_offset = chunk_id * (1 << (c+1));
            let global_ext_offset = chunk_id * pow3_adj;
            let global_ret_offset = chunk_id * pow3;

            for j in 0..pow3_adj {
//                println!("entry, j = {}", j);
                let offset = trit_mapping[j] as usize;
                if offset % 2 == 0 {
                    for z in 0..N {
                        let table_z = *table_ptrs.get(z);
                        let table_ext_z = *table_ext_ptrs.get_mut(z); 
                         *table_ext_z.get_mut(global_ext_offset + j) =
                             *table_z.get(global_tab_offset + (offset >> 1));
                         args[z] = (*table_ext_z.get(global_ext_offset + j));
                    }
                } else {
                    for z in 0..N {
                        let table_ext_z = *(table_ext_ptrs.get(z));
                         *table_ext_z.get_mut(global_ext_offset + j) =
                             (*table_ext_z.get(global_ext_offset + j - offset)) +
                             (*table_ext_z.get(global_ext_offset + j - 2 * offset));
                        args[z] = (*table_ext_z.get(global_ext_offset + j));
                    }
                }
                *ret_ptr.get_mut(global_ret_offset + j) = f(args);
            }

            for j in pow3_adj..pow3 {
                let offset = trit_mapping[j] as usize;
                for z in 0..N {
                    let table_ext_z = *(table_ext_ptrs.get_mut(z));
                    args[z] =
                        (*table_ext_z.get(global_ext_offset + j - offset)) +
                        (*table_ext_z.get(global_ext_offset + j - 2 * offset))
                    ;
                }
                *ret_ptr.get_mut(global_ret_offset + j) = f(args);
            }
        }).count();
        
    }
    unsafe{ret.assume_init()}
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
/// A new version of restrict, to work with boolcheck's contigious array API
/// It returns restrictions of all coordinates of all polynomials, and writes them in a single contigious array.
pub fn restrict(polys: &[&[F128]], coords: &[F128], dims: usize) -> Vec<F128> {
    let n = polys.len();
    for poly in polys.iter() {
        assert!(poly.len() == 1 << dims);
    }
    assert!(coords.len() <= dims);

    let chunk_size = (1 << coords.len());
    let num_chunks = 1 << (dims - coords.len());

    let eq = eq_poly(coords);

    assert!(eq.len() % 16 == 0, "Technical condition for now.");

    let mut eq_sums = Vec::with_capacity(256 * eq.len() / 8);

    for i in 0..eq.len()/8 {
        eq_sums.push(F128::zero());
        for j in 1..256 {
            let (sum_idx, eq_idx) = drop_top_bit(j);
            let tmp = eq[i * 8 + eq_idx] + eq_sums[i * 256 + sum_idx];
            eq_sums.push(tmp);
        }
    }

    let mut ret = vec![F128::zero(); num_chunks * 128 * n];
    let ret_ptr = ret.as_shared_mut_ptr();

    let n128 = n * 128;

    for q in 0..n {
        #[cfg(not(feature = "parallel"))]
        let iter = (0..num_chunks).into_iter();   
        #[cfg(feature = "parallel")]
        let iter = (0..num_chunks).into_par_iter(); 
        iter.map(|i| {
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

                        unsafe{
                            * ret_ptr.get_mut((s*8 + 7 - u + q * 128) * num_chunks + i) += v0[(bits & 255) as usize];
                            * ret_ptr.get_mut((s*8 + 7 - u + q * 128) * num_chunks + i) += v1[((bits >> 8) & 255) as usize];
                        }
                        t = v_slli_epi64::<1>(t);
                    }
                }

            }
        }
        ).count();
    }
    ret
}

//#[unroll::unroll_for_loops]
pub fn restrict_legacy(poly: &[F128], coords: &[F128], dims: usize) -> Vec<Vec<F128>> {
    assert!(poly.len() == 1 << dims);
    assert!(coords.len() <= dims);

    let chunk_size = (1 << coords.len());
    let num_chunks = 1 << (dims - coords.len());

    let eq = eq_poly(coords);

    assert!(eq.len() % 16 == 0, "Technical condition for now.");

    let mut eq_sums = Vec::with_capacity(256 * eq.len() / 8);

    for i in 0..eq.len()/8 {
        eq_sums.push(F128::zero());
        for j in 1..256 {
            let (sum_idx, eq_idx) = drop_top_bit(j);
            let tmp = eq[i * 8 + eq_idx] + eq_sums[i * 256 + sum_idx];
            eq_sums.push(tmp);
        }
    }

    let mut ret = vec![vec![F128::zero(); num_chunks]; 128];
    let ret_ptrs : [_; 128] = ret.iter_mut().map(|v| v.as_shared_mut_ptr())
        .collect_vec()
        .try_into()
        .unwrap_or_else(|_|panic!());

    #[cfg(feature = "parallel")]
    let iter = (0..num_chunks).into_par_iter();

    #[cfg(not(feature = "parallel"))]
    let iter = (0..num_chunks).into_iter();

    iter.map(move |i| {
        for j in 0 .. eq.len() / 16 { // Step by 16 
            let v0 = &eq_sums[j * 512 .. j * 512 + 256];
            let v1 = &eq_sums[j * 512 + 256 .. j * 512 + 512];
            let bytearr = cast_slice::<F128, [u8; 16]>(
                &poly[i * chunk_size + j * 16 .. i * chunk_size + (j + 1) * 16]
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

                    unsafe{
                        * ret_ptrs[s*8 + 7 - u].get_mut(i) += v0[(bits & 255) as usize];
                        * ret_ptrs[s*8 + 7 - u].get_mut(i) += v1[((bits >> 8) & 255) as usize];
                    }
                    t = v_slli_epi64::<1>(t);
                }
            }

        }
    }
    ).count();

    ret
}

#[cfg(test)]
mod tests {
    use std::iter::repeat_with;

    use rand::rngs::OsRng;

    use super::*;

    #[test]
    fn restrict_vs_restrict_legacy() {
        let rng = &mut OsRng;
        let num_vars = 8;
        let num_vars_to_restrict = 5;
        let pt : Vec<_> = repeat_with(|| F128::rand(rng)).take(num_vars).collect();
        let poly0 : Vec<_> = repeat_with(|| F128::rand(rng)).take(1 << num_vars).collect();
        let poly1 : Vec<_> = repeat_with(|| F128::rand(rng)).take(1 << num_vars).collect();
        let poly2 : Vec<_> = repeat_with(|| F128::rand(rng)).take(1 << num_vars).collect();

        let polys = [poly0.as_slice(), poly1.as_slice(), poly2.as_slice()];

        let new_answer = restrict(&polys, &pt[..num_vars_to_restrict], num_vars);

        let mut old_answer = vec![];
        for i in 0..3 {
            old_answer.push(
                restrict_legacy(polys[i], &pt[..num_vars_to_restrict], num_vars)
            );
        }

        assert!(old_answer.into_iter().map(|x|x.into_iter().flatten()).flatten().collect::<Vec<_>>() == new_answer);
    }
}
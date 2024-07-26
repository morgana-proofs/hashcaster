use std::{mem::{transmute, MaybeUninit}, sync::atomic::{AtomicU64, Ordering}, thread::sleep, time::{Duration, Instant}};

use num_traits::{One, Zero};
use rayon::{iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator}, slice::{ParallelSlice, ParallelSliceMut}};
use crate::{backend::autodetect::{v_movemask_epi8, v_slli_epi64}, field::{pi, F128}};
use itertools::Itertools;

pub enum RoundResponse {
    AwaitsFoldingChallenge,
    RoundPoly(Vec<F128>),
}

pub struct EvaluationClaim {
    pub point: Vec<F128>,
    pub eval: F128,
}

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
        let mut incoming = vec![MaybeUninit::<F128>::uninit(); 1 << i];
        unsafe{
        let ptr = transmute::<*mut MaybeUninit<F128>, usize>(incoming.as_mut_ptr());

            #[cfg(not(feature = "parallel"))]
            let iter = (0 .. (1 << (i-1))).into_iter();

            #[cfg(feature = "parallel")]
            let iter = (0 .. 1 << (i-1)).into_par_iter();

            iter.map(|j|{
                let ptr = transmute::<usize, *mut MaybeUninit<F128>>(ptr);
                let w = last[j];
                let m = multiplier * w;
                *ptr.offset(2*j as isize) = MaybeUninit::new(w + m);
                *ptr.offset((2*j + 1) as isize) = MaybeUninit::new(m);
            }).count();
            ret.push(transmute::<Vec<MaybeUninit<F128>>, Vec<F128>>(incoming));
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
    let mut ret = vec![MaybeUninit::uninit(); pow3 * pow2];
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
    unsafe{transmute::<Vec<MaybeUninit<F128>>, Vec<F128>>(ret)}
}

/// Extends two tables at the same time and ANDs them
/// Gives some advantage because we skip 1/3 of writes into p_ext and q_ext.
pub fn extend_2_tables(p: &[F128], q: &[F128], dims: usize, c: usize, trit_mapping: &[u16]) -> Vec<F128> {
    assert!(p.len() == 1 << dims);
    assert!(q.len() == 1 << dims);
    assert!(c < dims);
    let pow3 = 3usize.pow((c + 1) as u32);
    let pow3_adj = pow3 / 3 * 2;
    assert!(pow3 < (1 << 15) as usize, "This is too large anyway ;)");
    let pow2 = 2usize.pow((dims - c - 1) as u32);
    let mut p_ext = vec![MaybeUninit::uninit(); (pow3 * 2) / 3  * pow2];
    let mut q_ext = vec![MaybeUninit::uninit(); (pow3 * 2) / 3 * pow2];
    let mut ret = vec![MaybeUninit::uninit(); pow3 * pow2];

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
    unsafe{transmute::<Vec<MaybeUninit<F128>>, Vec<F128>>(ret)}
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
pub fn restrict(poly: &[F128], coords: &[F128], dims: usize) -> Vec<Vec<F128>> {
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
    let ret_ptrs : [usize; 128] = ret.iter_mut().map(|v| unsafe{
        transmute::<*mut F128, usize>((*v).as_mut_ptr()) // This is extremely ugly.  
    }).collect_vec().try_into().unwrap();

    #[cfg(feature = "parallel")]
    let iter = (0..num_chunks).into_par_iter();

    #[cfg(not(feature = "parallel"))]
    let iter = (0..num_chunks).into_iter();

    iter.map(move |i| {
        for j in 0 .. eq.len() / 16 { // Step by 16 
            let v0 = &eq_sums[j * 512 .. j * 512 + 256];
            let v1 = &eq_sums[j * 512 + 256 .. j * 512 + 512];
            let bytearr = unsafe{ transmute::<&[F128], &[[u8; 16]]>(
                &poly[i * chunk_size + j * 16 .. i * chunk_size + (j + 1) * 16]
            ) };

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
                        let ret_ptrs = transmute::<[usize; 128], [*mut F128; 128]>(ret_ptrs);
                        * ret_ptrs[s*8 + 7 - u].offset(i as isize) += v0[(bits & 255) as usize];
                        * ret_ptrs[s*8 + 7 - u].offset(i as isize) += v1[((bits >> 8) & 255) as usize];
                    }
                    t = v_slli_epi64::<1>(t);
                }
            }

        }
    }
    ).count();

    ret
}
use std::{mem::transmute, ops::{Add, AddAssign, BitAnd, BitAndAssign, Mul, MulAssign}};
use crate::u8_mult_table::MULT_TABLE;
use num_traits::{One, Zero};
use rand::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct F128 {
    raw: u128,
}

impl F128 {
    pub fn new(raw: u128) -> Self {
        Self{raw}
    }

    pub fn raw(&self) -> u128 {
        self.raw
    }

    pub fn into_raw(self) -> u128 {
        self.raw
    }

    pub fn rand<RNG: Rng>(rng: &mut RNG) -> Self {
        let a = rng.next_u64();
        let b = rng.next_u64();
        Self::new(unsafe{transmute::<(u64, u64), u128>((a, b))})
    }
}

impl Zero for F128 {
    fn zero() -> Self {
        Self{raw: 0}
    }

    fn is_zero(&self) -> bool {
        self.raw == 0
    }
}

impl One for F128 {
    fn one() -> Self {
        Self{raw: 1}
    }
}

impl Add<F128> for F128 {
    type Output = F128;

    fn add(self, rhs: Self) -> Self::Output {
        Self{raw: self.into_raw() ^ rhs.into_raw()}
    }
}

impl Add<&F128> for F128 {
    type Output = F128;

    fn add(self, rhs: &F128) -> Self::Output {
        Self{raw: self.into_raw() ^ rhs.raw()}
    }
}

impl BitAnd<F128> for F128 {
    type Output = F128;
    
    fn bitand(self, rhs: F128) -> Self::Output {
        Self{raw: self.into_raw() & rhs.into_raw()}
    }
}

impl BitAnd<&F128> for F128 {
    type Output = F128;
    
    fn bitand(self, rhs: &F128) -> Self::Output {
        Self{raw: self.into_raw() & rhs.raw()}
    }
}

impl AddAssign<F128> for F128 {
    fn add_assign(&mut self, rhs: F128) {
        self.raw ^= rhs.into_raw()
    }
}

impl AddAssign<&F128> for F128 {
    fn add_assign(&mut self, rhs: &F128) {
        self.raw ^= rhs.raw()
    }
}

impl BitAndAssign<F128> for F128 {
    fn bitand_assign(&mut self, rhs: F128) {
        self.raw &= rhs.into_raw();
    }
}

impl BitAndAssign<&F128> for F128 {
    fn bitand_assign(&mut self, rhs: &F128) {
        self.raw &= rhs.raw();
    }
}

impl Mul<F128> for F128 {
    type Output = F128;

    fn mul(self, rhs: F128) -> Self::Output {
        Self::new(m128(self.into_raw(), rhs.into_raw()))
    }
}

impl Mul<&F128> for F128 {
    type Output = F128;

    fn mul(self, rhs: &F128) -> Self::Output {
        Self::new(m128(self.into_raw(), rhs.raw()))
    }
}

impl MulAssign<F128> for F128 {
    fn mul_assign(&mut self, rhs: F128) {
        *self = *self * rhs;
    }
}

impl MulAssign<&F128> for F128 {
    fn mul_assign(&mut self, rhs: &F128) {
        *self = *self * rhs;
    }
}

pub fn m128(v1: u128, v2: u128) -> u128 {
    let (l1, h1) = unsafe{transmute::<u128, (u64, u64)>(v1)};
    let (l2, h2) = unsafe{transmute::<u128, (u64, u64)>(v2)};

    let l1l2 = m64(l1, l2);
    let h1h2 = m64(h1, h2);

    let h1h2_reduce = m64(1 << 32, h1h2);

    let z3 = m64(l1 ^ h1, l2 ^ h2);

    let q = l1l2 ^ h1h2;
    unsafe{transmute::<(u64, u64), u128>((q, h1h2_reduce ^ z3 ^ q))}
}

pub fn m64(v1: u64, v2: u64) -> u64 {
    let (l1, h1) = unsafe{transmute::<u64, (u32, u32)>(v1)};
    let (l2, h2) = unsafe{transmute::<u64, (u32, u32)>(v2)};

    let l1l2 = m32(l1, l2);
    let h1h2 = m32(h1, h2);

    let h1h2_reduce = m32(1 << 16, h1h2);

    let z3 = m32(l1 ^ h1, l2 ^ h2);

    let q = l1l2 ^ h1h2;
    unsafe{transmute::<(u32, u32), u64>((q, h1h2_reduce ^ z3 ^ q))}
}

pub fn m32(v1: u32, v2: u32) -> u32 {
    let (l1, h1) = unsafe{transmute::<u32, (u16, u16)>(v1)};
    let (l2, h2) = unsafe{transmute::<u32, (u16, u16)>(v2)};

    let l1l2 = m16(l1, l2);
    let h1h2 = m16(h1, h2);

    let h1h2_reduce = m16(1 << 8, h1h2);

    let z3 = m16(l1 ^ h1, l2 ^ h2);

    let q = l1l2 ^ h1h2;
    unsafe{transmute::<(u16, u16), u32>((q, h1h2_reduce ^ z3 ^ q))}
}

pub fn m16(v1: u16, v2: u16) -> u16 {

    let (l1, h1) = unsafe{transmute::<u16, (u8, u8)>(v1)};
    let (l2, h2) = unsafe{transmute::<u16, (u8, u8)>(v2)};

    let l1l2 = m8(l1, l2);
    let h1h2 = m8(h1, h2);

    let h1h2_reduce = m8(1 << 4, h1h2);

    let z3 = m8(l1 ^ h1, l2 ^ h2);

    let q = l1l2 ^ h1h2;
    unsafe{transmute::<(u8, u8), u16>((q, h1h2_reduce ^ z3 ^ q))}
}

pub fn m8(v1: u8, v2: u8) -> u8 {
    MULT_TABLE[v1 as usize][v2 as usize]
}

/// This is only used for precomputations, so it is fine if it is not very optimal.
pub fn m8_l(v1: u8, v2: u8, loglength: usize) -> u8 {
        
    assert!(loglength <= 3);
    let len = 1 << loglength;
    if len < 8 {
        assert!(v1 >> len == 0);
        assert!(v2 >> len == 0);
    }
    if len == 1 {
        return v1 * v2;
    }

    let h1 = v1 >> (len / 2);
    let l1 = v1 - (h1 << (len / 2));
    let h2 = v2 >> (len / 2);
    let l2 = v2 - (h2 << (len / 2));

    let l1l2 = m8_l(l1, l2, loglength - 1);
    let h1h2 = m8_l(h1, h2, loglength - 1);

    let h1h2_reduce = m8_l(1 << (len / 4), h1h2, loglength - 1);

    let z3 = m8_l(l1 ^ h1, l2 ^ h2, loglength - 1);

    let q = l1l2 ^ h1h2;

    ((h1h2_reduce ^ z3 ^ q) << (len / 2)) + q
}


#[cfg(test)]
mod tests {
    use std::{fs::File, io::Write, process::exit};
    use rand::rngs::OsRng;

    use super::*;
    #[test]
    fn m8_assoc() {
        let mut vals: Vec<Vec<u8>> = vec![vec![0; 256]; 256];
        for a in 0..256 {
            for b in 0..256 {
                vals[a][b] = m8(a as u8,b as u8);
            }
        }

        for a in 0..256 {
            for b in 0..256 {
                for c in 0..256 {
                    assert!(vals[vals[a][b] as usize][c] == vals[a][vals[b][c] as usize])
                }
            }
        }

        for a in 0..256 {
            assert!(vals[a][0] == 0);
            assert!(vals[a][1] == a as u8);
        }
    }

    #[test]
    fn precompute_m8() {
        let mut file = File::create("precomp.txt").unwrap_or_else(|_|{exit(1)});
        let mut vals: Vec<Vec<u8>> = vec![vec![0; 256]; 256];
        for a in 0..256 {
            for b in 0..256 {
                vals[a][b] = m8_l(a as u8,b as u8, 3);
            }
        }
        let ret = format!("{:?}", vals);
        file.write_all(ret.as_bytes()).unwrap();
    }

    #[test]
    fn f128_is_field() {
        let rng = &mut OsRng;
        let a = F128::rand(rng);
        let b = F128::rand(rng);
        let c = F128::rand(rng);

        let one = F128::new(1);

        assert_eq!(a * one, a);

        assert_eq!(a + b, b + a);
        assert_eq!(a * b, b * a);
        assert_eq!((a + b) * c, a * c + b * c);
        assert_eq!((a * b) * c, a * (b * c));

        let fr = |x: F128| {x * x};

        assert_eq!(fr(a) + fr(b), fr(a + b));

        let mut x = a;
        for _ in 0..128 {
            x = fr(x);    
        }
        assert_eq!(a, x);
    }


    #[test]
    fn frob_basis() {
        for i in 0..128 {
            let ai = F128::new(1 << i);
            let bi = ai * ai;
            println!("{:#0128b}", bi.raw());
        }
    }
}
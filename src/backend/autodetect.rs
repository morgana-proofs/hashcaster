#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use super::x86 as stuff;

#[cfg(all(target_arch = "aarch64"))]
use super::arm as stuff;

pub fn mul_128(a: u128, b: u128) -> u128 {
    stuff::mul_128(a, b)
}

pub fn v_movemask_epi8(x: [u8; 16]) -> i32 {
    unsafe{stuff::v_movemask_epi8(x)}
}
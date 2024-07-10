
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use super::clmul as mul;

#[cfg(all(target_arch = "aarch64"))]
use super::pmull as mul;

pub fn mul_128(a: u128, b: u128) -> u128 {
    mul::mul_128(a, b)
}
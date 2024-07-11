use cfg_if::cfg_if;

cfg_if! {
    if #[cfg(all(target_arch = "aarch64"))] {
        pub mod autodetect;
        mod arm;
    } else if #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86")))] {
            pub mod autodetect;
            mod x86;
        }
    }
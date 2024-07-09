// API to parallel iterator, allowing user to specify the task size.
// With parallel feature turned off, it won't parallelize anything - useful for single thread benchmarks / debugging.

use rayon::current_num_threads;
use rayon::prelude::*;

/// Executes f in parallel.
/// f expects a chunk of data, and an offset of initial element of the chunk
/// Chunks are guaranteed to be either task_size, or remainder.
pub fn parallelize<T : Send, F>(f: F, mut data: &mut[T], task_size: usize) where
    F: Fn(&mut[T], usize) + Send + Sync
{
    let l = data.len();
    let num_full_chunks = l / task_size;
    let mut chunks = Vec::with_capacity(num_full_chunks + 1);
    let mut chunk;
    for _ in 0..num_full_chunks {
        (chunk, data) = data.split_at_mut(task_size);
        chunks.push(chunk);
    }
    if data.len() > 0 {
        chunks.push(data);
    }

    #[cfg(not(feature = "parallel"))]
    chunks.into_iter().enumerate().map(|(i, chunk)|{
        f(chunk, i * task_size)
    }).count();

    #[cfg(feature = "parallel")]
    if chunks.len() < 4 * current_num_threads() {
        chunks.into_iter().enumerate().map(|(i, chunk)|{
            f(chunk, i * task_size)
        }).count();
    } else {
        chunks.into_par_iter().enumerate().map(|(i, chunk)|{
            f(chunk, i * task_size)
        }).count();
    }
}
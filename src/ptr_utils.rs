use std::{mem::{ManuallyDrop, MaybeUninit}, ops::{Index, IndexMut}, slice::{Chunks, ChunksMut}};

use rayon::slice::{ParallelSlice, ParallelSliceMut};

pub trait UnsafeIndex<Idx> {
    type Output;
    unsafe fn get(&self, index: Idx) -> &Self::Output;
}

pub trait UnsafeIndexMut<Idx> : UnsafeIndex<Idx>{
    unsafe fn get_mut(&mut self, index: Idx) -> &mut Self::Output;
}

pub trait UnsafeIndexRaw<Idx> {
    type Output;
    unsafe fn get<'a>(self, index: Idx) -> &'a Self::Output;
}
pub trait UnsafeIndexRawMut<Idx> : UnsafeIndexRaw<Idx> {
    unsafe fn get_mut<'a>(self, index: Idx) -> &'a mut Self::Output;
}

#[derive(Clone, Copy, Debug)]
pub struct ConstPtr<T> {
    ptr: *const T,
}

unsafe impl<T : Send + Sync> Send for ConstPtr<T> {}
unsafe impl<T : Send + Sync> Sync for ConstPtr<T> {}

impl<T> UnsafeIndexRaw<usize> for ConstPtr<T> {
    type Output = T;

    unsafe fn get<'a>(self, index: usize) -> &'a Self::Output {
        &* self.ptr.wrapping_add(index)
    }
} 

#[derive(Clone, Copy, Debug)]
pub struct MutPtr<T> {
    ptr: *mut T,
}

unsafe impl<T : Send + Sync> Send for MutPtr<T> {}
unsafe impl<T : Send + Sync> Sync for MutPtr<T> {}

impl<T> UnsafeIndexRaw<usize> for MutPtr<T> {
    type Output = T;

    unsafe fn get<'a>(self, index: usize) -> &'a Self::Output {
        &* self.ptr.wrapping_add(index)
    }
} 

impl<T> UnsafeIndexRawMut<usize> for MutPtr<T> {
    unsafe fn get_mut<'a>(self, index: usize) -> &'a mut Self::Output {
        &mut *self.ptr.wrapping_add(index)
    }
}

#[derive(Debug)]
pub struct UninitArr<T> {
    val: Box<[MaybeUninit<T>]>
}


impl<T> UninitArr<T> {
    pub fn new(length: usize) -> Self {
        let boxed_slice = Box::new_uninit_slice(length);
        Self{ val: boxed_slice }
    }

    pub unsafe fn assume_init(self) -> Vec<T> {
        self.val.assume_init().into_vec()
    }

    pub fn chunks(&self, chunk_size: usize) -> Chunks<MaybeUninit<T>> {
        self.val.chunks(chunk_size)
    }

    pub fn chunks_mut(&mut self, chunk_size: usize) -> ChunksMut<MaybeUninit<T>> {
        self.val.chunks_mut(chunk_size)
    }
}

impl<T: Send + Sync> UninitArr<T> {
    pub fn par_chunks(&self, chunk_size: usize) -> rayon::slice::Chunks<MaybeUninit<T>> {
        self.val.as_parallel_slice().par_chunks(chunk_size)
    }

    pub fn par_chunks_mut(&mut self, chunk_size: usize) -> rayon::slice::ChunksMut<MaybeUninit<T>> {
        self.val.as_parallel_slice_mut().par_chunks_mut(chunk_size)
    }
}

impl<T> UnsafeIndex<usize> for UninitArr<T> {
    type Output = T;

    unsafe fn get(&self, index: usize) -> &Self::Output {
        self.val.as_ref()[index].assume_init_ref()
    }
}

impl<T> UnsafeIndexMut<usize> for UninitArr<T> {
    unsafe fn get_mut(&mut self, index: usize) -> &mut Self::Output {
        self.val.as_mut()[index].assume_init_mut()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MUConstPtr<T> {
    ptr: *const MaybeUninit<T>,
}

unsafe impl<T : Send + Sync> Send for MUConstPtr<T> {}
unsafe impl<T : Send + Sync> Sync for MUConstPtr<T> {}

impl<T> UnsafeIndexRaw<usize> for MUConstPtr<T> {
    type Output = T;

    unsafe fn get<'a>(self, index: usize) -> &'a Self::Output {
        (&*self.ptr.wrapping_add(index)).assume_init_ref()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MUMutPtr<T> {
    ptr: *mut MaybeUninit<T>,
}

unsafe impl<T : Send + Sync> Send for MUMutPtr<T> {}
unsafe impl<T : Send + Sync> Sync for MUMutPtr<T> {}

impl<T> UnsafeIndexRaw<usize> for MUMutPtr<T> {
    type Output = T;

    unsafe fn get<'a>(self, index: usize) -> &'a Self::Output {
        (&*self.ptr.wrapping_add(index)).assume_init_ref()
    }
}

impl<T> UnsafeIndexRawMut<usize> for MUMutPtr<T> {
    unsafe fn get_mut<'a>(self, index: usize) -> &'a mut Self::Output {
        (&mut *self.ptr.wrapping_add(index)).assume_init_mut()
    }
}

pub trait AsSharedConstPtr {
    type T;
    fn as_shared_ptr(&self) -> ConstPtr<Self::T>;
}

pub trait AsSharedMutPtr {
    type T;
    fn as_shared_mut_ptr(&mut self) -> MutPtr<Self::T>;
}

pub trait AsSharedMUConstPtr {
    type T;
    fn as_shared_ptr(&self) -> MUConstPtr<Self::T>;
}

pub trait AsSharedMUMutPtr {
    type T;
    fn as_shared_mut_ptr(&mut self) -> MUMutPtr<Self::T>;
}

impl<T> AsSharedMUConstPtr for UninitArr<T> {
    type T = T;
    fn as_shared_ptr(&self) -> MUConstPtr<Self::T> {
        MUConstPtr { ptr: self.val.as_ptr() }
    }
}

impl<T> AsSharedMUMutPtr for UninitArr<T> {
    type T = T;
    fn as_shared_mut_ptr(&mut self) -> MUMutPtr<Self::T> {
        MUMutPtr { ptr: self.val.as_mut_ptr() }
    }
}

impl<T> AsSharedConstPtr for Vec<T> {
    type T = T;

    fn as_shared_ptr(&self) -> ConstPtr<Self::T> {
        ConstPtr { ptr: self.as_ptr() }
    }
}

impl<T> AsSharedMutPtr for Vec<T> {
    type T = T;

    fn as_shared_mut_ptr(&mut self) -> MutPtr<Self::T> {
        MutPtr { ptr: self.as_mut_ptr() }
    }
}

impl<T> AsSharedConstPtr for [T] {
    type T = T;

    fn as_shared_ptr(&self) -> ConstPtr<Self::T> {
        ConstPtr { ptr: self.as_ptr() }
    }
}

impl<T> AsSharedMutPtr for [T] {
    type T = T;

    fn as_shared_mut_ptr(&mut self) -> MutPtr<Self::T> {
        MutPtr { ptr: self.as_mut_ptr() }
    }
}

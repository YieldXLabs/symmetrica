use algebra::{BinaryKernel, Data, ReduceKernel, StreamKernel, UnaryKernel};
use std::fmt::Debug;

pub trait Storage: Debug + Clone + Send + Sync {
    type Elem: Data;

    fn len(&self) -> usize;
    fn alloc(n: usize) -> Self;
    fn as_slice(&self) -> &[Self::Elem];
    fn as_mut_slice(&mut self) -> &mut [Self::Elem];
}

pub trait Backend {
    type Storage<T: Data>: Storage<Elem = T>;

    // Memory management
    fn pure<T: Data>(&mut self, data: &[T]) -> Self::Storage<T>;
    // TODO: Async / Non-blocking Transfers.
    fn to_host<T: Data>(&mut self, device_data: &Self::Storage<T>) -> Vec<T>;

    fn compact<T: Data>(
        &mut self,
        src: &Self::Storage<T>,
        shape: &[usize],
        strides: &[usize],
        offset: usize,
    ) -> Self::Storage<T>;

    // Generic Kernels
    // The Kernel 'K' defines the math
    fn unary<I: Data, K: UnaryKernel<I>>(
        &mut self,
        input: &Self::Storage<I>,
        kernel: K,
    ) -> Self::Storage<K::Output>
    where
        K::Output: Data;

    fn binary<L: Data, R: Data, K: BinaryKernel<L, R>>(
        &mut self,
        lhs: &Self::Storage<L>,
        rhs: &Self::Storage<R>,
        kernel: K,
    ) -> Self::Storage<K::Output>
    where
        K::Output: Data;

    fn reduce<I: Data, K: ReduceKernel<I>>(
        &mut self,
        input: &Self::Storage<I>,
        kernel: K,
    ) -> Self::Storage<K::Output>
    where
        K::Output: Data;

    fn stream<I: Data, K: StreamKernel<I>>(
        &mut self,
        input: &Self::Storage<I>,
        kernel: K,
    ) -> Self::Storage<K::Output>
    where
        K::Output: Data;

    // TODO: GEMM / Linear Algebra.
    // `reduce` is generic, but Matrix Multiplication requires specialized tiling/cache algorithms.
    // A `matmul` or `gemm` function should be part of the Backend trait to hook into BLAS/cuBLAS.

    // TODO: Random Number Generation.
    // fn random_uniform(&mut self, shape: &[usize], min: f32, max: f32) -> Self::Storage<f32>;
    // fn random_normal(&mut self, shape: &[usize], mean: f32, std: f32) -> Self::Storage<f32>;
}

// TODO: Checkpoints
// pub struct CheckpointId(String);
// pub trait CheckpointBackend<F: Data>: Backend<F> {
//     fn checkpoint(&self, data: &Self::Repr) -> CheckpointId;
//     fn restore(&mut self, id: CheckpointId) -> Self::Repr;
//     fn delete(&mut self, id: CheckpointId);
// }
// TODO: Distributed computation and shared storage
// TODO: Quantization

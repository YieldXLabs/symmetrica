use algebra::{BinaryKernel, Data, ReduceKernel, StreamKernel, UnaryKernel};
use std::fmt::Debug;

pub trait Storage<F>: Debug + Clone + Send + Sync {
    fn len(&self) -> usize;
    fn alloc(n: usize) -> Self;
    fn as_slice(&self) -> &[F];
    fn as_mut_slice(&mut self) -> &mut [F];
}

pub trait Backend<F: Data> {
    type Repr: Storage<F>;

    // Memory management
    fn pure(&mut self, data: &[F]) -> Self::Repr;
    fn to_host(&mut self, device_data: &Self::Repr) -> Vec<F>;
    fn compact(
        &mut self,
        src: &Self::Repr,
        shape: &[usize],
        strides: &[usize],
        offset: usize,
    ) -> Self::Repr;

    // Generic Kernels
    // The Kernel 'K' defines the math
    fn unary<K: UnaryKernel<F>>(&mut self, input: &Self::Repr, kernel: K) -> Self::Repr;
    fn binary<K: BinaryKernel<F>>(
        &mut self,
        lhs: &Self::Repr,
        rhs: &Self::Repr,
        kernel: K,
    ) -> Self::Repr;
    fn stream<K: StreamKernel<F>>(&mut self, input: &Self::Repr, kernel: K) -> Self::Repr;
    fn reduce<K: ReduceKernel<F>>(&mut self, input: &Self::Repr) -> F;
}

// TODO: Checkpoints
// pub struct CheckpointId(String);
// pub trait CheckpointBackend<F: Data>: Backend<F> {
//     fn checkpoint(&self, data: &Self::Repr) -> CheckpointId;
//     fn restore(&mut self, id: CheckpointId) -> Self::Repr;
//     fn delete(&mut self, id: CheckpointId);
// }

// TODO: Distributed computation and shared storage

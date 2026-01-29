use algebra::{BinaryKernel, Data, ReduceKernel, Semiring, StreamKernel, UnaryKernel};
use std::fmt::Debug;

pub trait Storage<F>: Debug + Clone + Send + Sync {
    fn len(&self) -> usize;
    fn alloc(n: usize) -> Self;
    fn as_slice(&self) -> &[F];
    fn as_mut_slice(&mut self) -> &mut [F];
}

#[derive(Debug, Clone, Copy)]
pub struct Layout<'a, const R: usize> {
    pub shape: &'a [usize; R],
    pub strides: &'a [usize; R],
    pub offset: usize,
}

pub trait Backend<F: Data> {
    type Repr: Storage<F>;

    // Memory management
    fn pure(&mut self, data: &[F]) -> Self::Repr;
    fn to_host(&mut self, device_data: &Self::Repr) -> Vec<F>;
    fn compact<const R: usize>(&mut self, src: &Self::Repr, layout: Layout<R>) -> Self::Repr;

    // Contraction (Matrix Mul and Tensor Dot)
    fn contract<const RL: usize, const RR: usize>(
        &mut self,
        lhs: &Self::Repr,
        lhs_layout: Layout<RL>,
        axis_l: usize,
        rhs: &Self::Repr,
        rhs_layout: Layout<RR>,
        axis_r: usize,
    ) -> Self::Repr
    where
        F: Semiring;

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

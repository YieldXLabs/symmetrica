use super::Base;
use algebra::Data;
use backend::Backend;

// TODO: Memory Layout Strategies.
// `PackDense` implies a standard Row-Major (C-contiguous) packing.
// High-performance libraries often need:
// 1. `PackColMajor` (Fortran-contiguous) for BLAS/LAPACK compatibility.
// 2. `PackNCHW` / `PackNHWC` for image data optimization (SIMD friendly).
pub struct PackDense;

// TODO: Kernel Compilation Pass.
// The `Lower` trait is effectively a compiler pass.
// Currently, it handles memory compaction.
// It should eventually handle "Kernel Fusion" -- taking a graph of Lazy expressions
// and lowering them into a single compiled Backend::Kernel.
pub trait Lower<Target, B> {
    type Output;

    fn lower(&self, backend: &mut B) -> Self::Output;
}

impl<F, B, const R: usize> Lower<PackDense, B> for Base<B::Storage<F>, F, R>
where
    F: Data,
    B: Backend,
    B::Storage<F>: Clone,
{
    type Output = B::Storage<F>;

    fn lower(&self, backend: &mut B) -> Self::Output {
        if self.is_dense() {
            self.storage.clone()
        } else {
            backend.compact(&self.storage, &self.shape, &self.strides, self.offset)
        }
    }
}

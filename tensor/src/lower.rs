use super::Base;
use algebra::Data;
use backend::Backend;

pub struct PackDense;

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

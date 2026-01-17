use super::DenseExpr;
use algebra::Real;
use backend::{Backend, Evaluator};

impl<F: Real, B: Backend<F>, const R: usize> Evaluator<F, B> for DenseExpr<Vec<F>, R> {
    fn eval(&self, backend: &mut B) -> (B::Repr, Vec<usize>) {
        let data = self.data.as_slice();

        let raw_slice = &data[self.offset..];

        let storage = backend.pure(raw_slice);

        (storage, self.shape.to_vec())
    }
}

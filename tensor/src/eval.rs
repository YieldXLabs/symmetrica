use super::DenseExpr;
use algebra::{AddExpr, Real};
use backend::{AddKernel, Backend};

pub trait Evaluator<F: Real, B: Backend<F> + ?Sized> {
    fn eval(&self, backend: &mut B) -> (B::Repr, Vec<usize>);
}

impl<F: Real, B: Backend<F>, const RANK: usize> Evaluator<F, B> for DenseExpr<Vec<F>, RANK> {
    fn eval(&self, backend: &mut B) -> (B::Repr, Vec<usize>) {
        let data = self.data.as_slice();

        let raw_slice = &data[self.offset..];

        let storage = backend.pure(raw_slice);

        (storage, self.shape.to_vec())
    }
}

impl<F: Real, B: Backend<F>, L, R> Evaluator<F, B> for AddExpr<L, R>
where
    L: Evaluator<F, B>,
    R: Evaluator<F, B>,
{
    fn eval(&self, backend: &mut B) -> (B::Repr, Vec<usize>) {
        let (lhs_data, lhs_shape) = self.left.eval(backend);
        let (rhs_data, rhs_shape) = self.right.eval(backend);

        assert_eq!(lhs_shape, rhs_shape, "Shape mismatch in Add");

        let result_data = backend.binary::<AddKernel>(&lhs_data, &rhs_data);

        (result_data, lhs_shape)
    }
}

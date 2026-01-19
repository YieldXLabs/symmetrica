use super::{Base, Evaluator};
use algebra::{AddExpr, Real, ScaleExpr, SubExpr};
use backend::{AddKernel, Backend, SubKernel};

impl<F, B, L, Rhs, const R: usize> Evaluator<F, B, R> for AddExpr<L, Rhs>
where
    F: Real,
    B: Backend<F>,
    L: Evaluator<F, B, R>,
    Rhs: Evaluator<F, B, R>,
{
    fn eval(&self, backend: &mut B) -> Base<B::Repr, F, R> {
        let l = self.left.eval(backend);
        let r = self.right.eval(backend);

        let storage = backend.binary::<AddKernel>(&l.storage, &r.storage);

        Base::new(storage, l.shape)
    }
}

impl<F, B, L, Rhs, const R: usize> Evaluator<F, B, R> for SubExpr<L, Rhs>
where
    F: Real,
    B: Backend<F>,
    L: Evaluator<F, B, R>,
    Rhs: Evaluator<F, B, R>,
{
    fn eval(&self, backend: &mut B) -> Base<B::Repr, F, R> {
        let l = self.left.eval(backend);
        let r = self.right.eval(backend);

        let storage = backend.binary::<SubKernel>(&l.storage, &r.storage);

        Base::new(storage, l.shape)
    }
}

impl<F, B, Op, const R: usize> Evaluator<F, B, R> for ScaleExpr<Op, F>
where
    F: Real,
    B: Backend<F>,
    Op: Evaluator<F, B, R>,
{
    fn eval(&self, backend: &mut B) -> Base<B::Repr, F, R> {
        let view = self.op.eval(backend);

        let storage = backend.scale(&view.storage, self.factor);

        Base::from_parts(storage, view.shape, view.strides, view.offset)
    }
}

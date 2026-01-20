use super::{Base, Evaluator};
use algebra::{AddExpr, Real, ReshapeExpr, ScaleExpr, SubExpr, TransposeExpr};
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

impl<F, B, E, const R: usize> Evaluator<F, B, R> for TransposeExpr<E, R>
where
    F: Real,
    B: Backend<F>,
    E: Evaluator<F, B, R>,
{
    fn eval(&self, backend: &mut B) -> Base<B::Repr, F, R> {
        let view = self.op.eval(backend);

        let mut new_shape = [0; R];
        let mut new_strides = [0; R];

        for i in 0..R {
            let src_idx = self.perm[i];
            new_shape[i] = view.shape[src_idx];
            new_strides[i] = view.strides[src_idx];
        }

        Base::from_parts(view.storage, new_shape, new_strides, view.offset)
    }
}

impl<F, B, E, const R_IN: usize, const R_OUT: usize> Evaluator<F, B, R_OUT>
    for ReshapeExpr<E, R_IN, R_OUT>
where
    F: Real,
    B: Backend<F>,
    E: Evaluator<F, B, R_IN>,
{
    fn eval(&self, backend: &mut B) -> Base<B::Repr, F, R_OUT> {
        let view = self.op.eval(backend);

        let new_strides = Base::<B::Repr, F, R_OUT>::compute_strides(&self.new_shape);

        Base::from_parts(view.storage, self.new_shape, new_strides, 0)
    }
}

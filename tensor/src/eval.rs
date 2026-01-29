use super::{Base, Evaluator};
use algebra::{BinaryKernel, Data, MapExpr, ReshapeExpr, TransposeExpr, UnaryKernel, ZipExpr};
use backend::Backend;

impl<F, B, L, R, K, const RANK: usize> Evaluator<F, B, RANK> for ZipExpr<L, R, K>
where
    F: Data,
    B: Backend<F>,
    K: BinaryKernel<F>,
    L: Evaluator<F, B, RANK>,
    R: Evaluator<F, B, RANK>,
{
    fn eval(&self, backend: &mut B) -> Base<B::Repr, F, RANK> {
        let l = self.left.eval(backend);
        let r = self.right.eval(backend);

        let storage = backend.binary(&l.storage, &r.storage, self.kernel);

        Base::new(storage, l.shape)
    }
}

impl<F, B, Op, K, const RANK: usize> Evaluator<F, B, RANK> for MapExpr<Op, K>
where
    F: Data,
    B: Backend<F>,
    K: UnaryKernel<F>,
    Op: Evaluator<F, B, RANK>,
{
    fn eval(&self, backend: &mut B) -> Base<B::Repr, F, RANK> {
        let view = self.op.eval(backend);

        let storage = backend.unary(&view.storage, self.kernel);

        Base::from_parts(storage, view.shape, view.strides, view.offset)
    }
}

// Structural Ops
impl<F, B, E, const R: usize> Evaluator<F, B, R> for TransposeExpr<E, R>
where
    F: Data,
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
    F: Data,
    B: Backend<F>,
    E: Evaluator<F, B, R_IN>,
{
    fn eval(&self, backend: &mut B) -> Base<B::Repr, F, R_OUT> {
        let view = self.op.eval(backend);

        let new_strides = Base::<B::Repr, F, R_OUT>::compute_strides(&self.new_shape);

        Base::from_parts(view.storage, self.new_shape, new_strides, 0)
    }
}

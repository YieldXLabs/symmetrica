use super::{Base, Evaluator, Lower, PackDense};
use algebra::{BinaryKernel, Data, MapExpr, ReshapeExpr, TransposeExpr, UnaryKernel, ZipExpr};
use backend::Backend;

impl<F, B, L, R, K, const RANK: usize> Evaluator<F, B, RANK> for ZipExpr<L, R, K>
where
    F: Data,
    B: Backend,
    K: BinaryKernel<F, F, Output = F>,
    L: Evaluator<F, B, RANK>,
    R: Evaluator<F, B, RANK>,
{
    fn eval(&self, backend: &mut B) -> Base<B::Storage<F>, F, RANK> {
        let l_view = self.left.eval(backend);
        let r_view = self.right.eval(backend);

        let l_dense = Lower::<PackDense, B>::lower(&l_view, backend);
        let r_dense = Lower::<PackDense, B>::lower(&r_view, backend);

        let storage = backend.binary(&l_dense, &r_dense, self.kernel);

        Base::new(storage, l_view.shape)
    }
}

impl<F, B, Op, K, const RANK: usize> Evaluator<F, B, RANK> for MapExpr<Op, K>
where
    F: Data,
    B: Backend,
    K: UnaryKernel<F, Output = F>,
    Op: Evaluator<F, B, RANK>,
{
    fn eval(&self, backend: &mut B) -> Base<B::Storage<F>, F, RANK> {
        let view = self.op.eval(backend);
        let input = Lower::<PackDense, B>::lower(&view, backend);

        let storage = backend.unary(&input, self.kernel);

        Base::from_parts(storage, view.shape, view.strides, view.offset)
    }
}

// Structural Ops
impl<F, B, E, const R: usize> Evaluator<F, B, R> for TransposeExpr<E, R>
where
    F: Data,
    B: Backend,
    E: Evaluator<F, B, R>,
{
    fn eval(&self, backend: &mut B) -> Base<B::Storage<F>, F, R> {
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
    B: Backend,
    E: Evaluator<F, B, R_IN>,
{
    fn eval(&self, backend: &mut B) -> Base<B::Storage<F>, F, R_OUT> {
        let view = self.op.eval(backend);

        let new_strides = Base::<B::Storage<F>, F, R_OUT>::compute_strides(&self.new_shape);

        Base::from_parts(view.storage, self.new_shape, new_strides, 0)
    }
}

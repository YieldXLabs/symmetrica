use super::{Base, Evaluator, Lower, PackDense};
use algebra::{BinaryKernel, Data, MapExpr, ReshapeExpr, TransposeExpr, UnaryKernel, ZipExpr};
use backend::Backend;

impl<B, L, R, K, const RANK: usize> Evaluator<B, RANK> for ZipExpr<L, R, K>
where
    B: Backend,
    L: Evaluator<B, RANK>,
    R: Evaluator<B, RANK>,
    K: BinaryKernel<L::Data, R::Data>,
    K::Output: Data,
{
    type Data = K::Output;

    fn eval(&self, backend: &mut B) -> Base<B::Storage<Self::Data>, Self::Data, RANK> {
        let l_view = self.left.eval(backend);
        let r_view = self.right.eval(backend);

        let l_dense = Lower::<PackDense, B>::lower(&l_view, backend);
        let r_dense = Lower::<PackDense, B>::lower(&r_view, backend);

        let storage = backend.binary(&l_dense, &r_dense, self.kernel);

        Base::new(storage, l_view.shape)
    }
}

impl<B, Op, K, const RANK: usize> Evaluator<B, RANK> for MapExpr<Op, K>
where
    B: Backend,
    Op: Evaluator<B, RANK>,
    K: UnaryKernel<Op::Data>,
    K::Output: Data,
{
    type Data = K::Output;

    fn eval(&self, backend: &mut B) -> Base<B::Storage<Self::Data>, Self::Data, RANK> {
        let view = self.op.eval(backend);
        let input = Lower::<PackDense, B>::lower(&view, backend);

        let storage = backend.unary(&input, self.kernel);

        Base::from_parts(storage, view.shape, view.strides, view.offset)
    }
}

impl<B, E, const R: usize> Evaluator<B, R> for TransposeExpr<E, R>
where
    B: Backend,
    E: Evaluator<B, R>,
{
    type Data = E::Data;

    fn eval(&self, backend: &mut B) -> Base<B::Storage<Self::Data>, Self::Data, R> {
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

impl<B, E, const R_IN: usize, const R_OUT: usize> Evaluator<B, R_OUT>
    for ReshapeExpr<E, R_IN, R_OUT>
where
    B: Backend,
    E: Evaluator<B, R_IN>,
{
    type Data = E::Data;

    fn eval(&self, backend: &mut B) -> Base<B::Storage<Self::Data>, Self::Data, R_OUT> {
        let view = self.op.eval(backend);
        let dense_view = Lower::<PackDense, B>::lower(&view, backend);

        let new_strides =
            Base::<B::Storage<Self::Data>, Self::Data, R_OUT>::compute_strides(&self.new_shape);

        Base::from_parts(dense_view, self.new_shape, new_strides, 0)
    }
}

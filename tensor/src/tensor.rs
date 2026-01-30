use super::{Differentiable, Evaluator, GradientTape, LeafAdjoint, Lift, Lower, PackDense};
use algebra::{
    BroadcastExpr, BroadcastMap, Data, DynRank, MapExpr, Permutation, Real, ReshapeExpr,
    ScaleKernel, Semiring, Shape, TransposeExpr,
};
use backend::Backend;
use std::{marker::PhantomData, sync::Arc};

// A view over tensor data
#[derive(Debug, Clone)]
pub struct Base<S, F, const R: usize> {
    pub storage: S,
    pub shape: [usize; R],
    pub strides: [usize; R],
    pub offset: usize,
    _marker: PhantomData<F>,
}

impl<S, F, const R: usize> Base<S, F, R> {
    pub fn new(storage: S, shape: [usize; R]) -> Self {
        Self {
            storage,
            shape,
            strides: Self::compute_strides(&shape),
            offset: 0,
            _marker: PhantomData,
        }
    }

    pub fn from_parts(storage: S, shape: [usize; R], strides: [usize; R], offset: usize) -> Self {
        debug_assert!(shape.iter().all(|&d| d > 0));
        Self {
            storage,
            shape,
            strides,
            offset,
            _marker: PhantomData,
        }
    }

    pub fn is_dense(&self) -> bool {
        if self.offset != 0 {
            return false;
        }

        let mut stride = 1;

        for i in (0..R).rev() {
            if self.shape[i] > 1 {
                if self.strides[i] != stride {
                    return false;
                }
                stride *= self.shape[i];
            }
        }

        true
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn compute_strides(shape: &[usize; R]) -> [usize; R] {
        let mut strides = [0; R];
        let mut product = 1;

        for i in (0..R).rev() {
            strides[i] = product;
            product *= shape[i];
        }

        strides
    }
}

pub type Dense<F, const R: usize> = Base<Arc<Vec<F>>, F, R>;

impl<F: Data, B: Backend<F>, const R: usize> Evaluator<F, B, R> for Dense<F, R> {
    fn eval(&self, backend: &mut B) -> Base<B::Repr, F, R> {
        let storage = backend.pure(&self.storage);
        Base::from_parts(storage, self.shape, self.strides, self.offset)
    }
}

impl<F: Real, B: Backend<F>, const R: usize> Differentiable<F, B, R> for Dense<F, R> {
    type Adjoint = LeafAdjoint;

    fn forward(&self, backend: &mut B) -> (Base<B::Repr, F, R>, Self::Adjoint) {
        let res = self.eval(backend);
        (res, LeafAdjoint)
    }
}

// TODO: implement toeplitz(), zeros(), ones(), full(), eye(), one_hot(), random()
// TODO: ones_like, zeros_like, full_like
// TODO: implement slice over axes
#[derive(Debug, Clone)]
pub struct Tensor<F, Sh: Shape, E = Dense<F, { Sh::RANK }>> {
    pub expr: E,
    pub _marker: PhantomData<(F, Sh)>,
}

impl<F: Data, Sh: Shape, E> Tensor<F, Sh, E> {
    pub(crate) fn wrap(expr: E) -> Self {
        Self {
            expr,
            _marker: PhantomData,
        }
    }

    pub fn align<NewShape>(self) -> Tensor<F, NewShape, TransposeExpr<E, { Sh::RANK }>>
    where
        NewShape: Shape,
        Sh: Permutation<NewShape>,
    {
        let vec_idx = <Sh as Permutation<NewShape>>::indices();
        let array_idx: [usize; Sh::RANK] = vec_idx.try_into().expect("Rank mismatch");

        Tensor::wrap(TransposeExpr {
            op: self.expr,
            perm: array_idx,
        })
    }

    pub fn expand<Target>(
        self,
        target_sizes: [usize; Target::RANK],
    ) -> Tensor<F, Target, BroadcastExpr<E, { Sh::RANK }, { Target::RANK }>>
    where
        Target: Shape + BroadcastMap<Sh>,
    {
        let vec_map = <Target as BroadcastMap<Sh>>::mapping();
        let array_map: [Option<usize>; Target::RANK] =
            vec_map.try_into().ok().expect("Mapping length mismatch");

        Tensor::wrap(BroadcastExpr {
            op: self.expr,
            target_shape: target_sizes,
            mapping: array_map,
        })
    }

    pub fn reshape<const NEW_R: usize>(
        self,
        new_shape: [usize; NEW_R],
    ) -> Tensor<F, DynRank<NEW_R>, ReshapeExpr<E, { Sh::RANK }, NEW_R>> {
        Tensor::wrap(ReshapeExpr {
            op: self.expr,
            new_shape,
        })
    }

    pub fn collect<B: Backend<F>>(&self, backend: &mut B) -> Base<B::Repr, F, { Sh::RANK }>
    where
        E: Evaluator<F, B, { Sh::RANK }>,
    {
        self.expr.eval(backend)
    }

    pub fn to_vec<B: Backend<F>>(&self, backend: &mut B) -> Vec<F>
    where
        E: Evaluator<F, B, { Sh::RANK }>,
    {
        let view = self.expr.eval(backend);
        let dense = Lower::<PackDense, B>::lower(&view, backend);
        backend.to_host(&dense)
    }
}

impl<F: Data, const R: usize> Tensor<F, DynRank<R>, Dense<F, R>> {
    pub fn from_vec(data: Vec<F>, shape: [usize; R]) -> Self {
        debug_assert_eq!(data.len(), shape.iter().product::<usize>());
        Tensor::wrap(Dense::new(Arc::new(data), shape))
    }

    pub fn from_slice(data: &[F], shape: [usize; R]) -> Self {
        Self::from_vec(data.to_vec(), shape)
    }

    pub fn from_expr<L>(input: L) -> Tensor<F, DynRank<R>, L::Output>
    where
        L: Lift<F>,
    {
        Tensor::wrap(input.lift())
    }

    pub fn into_named<NewSh: Shape>(self) -> Tensor<F, NewSh, Dense<F, R>> {
        debug_assert_eq!(NewSh::RANK, R, "Rank mismatch");
        Tensor::wrap(self.expr)
    }
}
// Algebraic Ops
// TODO: Add .matmul() and .scan()
impl<F: Semiring, Sh: Shape, E> Tensor<F, Sh, E>
where
    [(); Sh::RANK]: Sized,
{
    pub fn scale(self, factor: F) -> Tensor<F, Sh, MapExpr<E, ScaleKernel<F>>> {
        Tensor::wrap(MapExpr {
            op: self.expr,
            kernel: ScaleKernel { factor },
        })
    }
}

// Calculus Ops (Gradients, Physics)
impl<F: Real, Sh: Shape, E> Tensor<F, Sh, E> {
    pub fn forward<B: Backend<F>>(
        &self,
        backend: &mut B,
    ) -> (Base<B::Repr, F, { Sh::RANK }>, GradientTape<E::Adjoint>)
    where
        E: Differentiable<F, B, { Sh::RANK }>,
    {
        let (res, adjoint) = self.expr.forward(backend);
        (res, GradientTape::new(adjoint))
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! __count {
    () => (0usize);
    ($head:expr $(, $tail:expr)*) => (1usize + $crate::__count!($($tail),*));
}

#[doc(hidden)]
#[macro_export]
macro_rules! __flatten_1d {
    ($vec:ident; $($x:expr),* $(,)?) => {{
        use ::algebra::TradingFloat;
        $(
            $vec.push(TradingFloat::try_from($x).expect("Invalid float"));
        )*
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __flatten_2d {
    ($vec:ident; $([$($x:expr),* $(,)?]),* $(,)?) => {{
        $(
            $crate::__flatten_1d!($vec; $($x),*);
        )*
    }};
}

#[macro_export]
macro_rules! tensor {
    ($($x:expr),+ $(,)?) => {{
        use ::algebra::TradingFloat;
        let mut data = Vec::<TradingFloat>::new();
        $crate::__flatten_1d!(data; $($x),*);
        let shape = [$crate::__count!($($x),*)];
        Tensor::from_vec(data, shape)
    }};

    ([$([$($x:expr),* $(,)?]),+ $(,)?]) => {{
        use ::algebra::TradingFloat;
        let mut data = Vec::<TradingFloat>::new();
        $crate::__flatten_2d!(data; $([$($x),*]),*);
        let rows = $crate::__count!($([$($x),*]),*);
        let cols = $crate::__count!($($x),*);
        let shape = [rows, cols];
        Tensor::from_vec(data, shape)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use algebra::TradingFloat;
    use backend::GenericBackend;

    #[test]
    fn test_tensor_add() {
        let mut backend = GenericBackend::<TradingFloat>::new();
        let a = tensor![2.0, 3.0, 5.0];
        let b = tensor![1.0, 3.0, 2.0];

        let c = a + b;

        let result = c.to_vec(&mut backend);

        assert_eq!(result[0].to_f64(), 3.0);
        assert_eq!(result[1].to_f64(), 6.0);
        assert_eq!(result[2].to_f64(), 7.0);
    }
}

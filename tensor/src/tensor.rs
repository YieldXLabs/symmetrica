use super::{Differentiable, Evaluator, GradientTape, LeafAdjoint, Lift};
use algebra::{DynRank, Real, ScaleExpr, Shape};
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

// TODO: ones_like, zeros_like, full_like
impl<S, F, const R: usize> Base<S, F, R> {
    pub fn new(storage: S, shape: [usize; R]) -> Self {
        let strides = Self::compute_strides(&shape);

        Self {
            storage,
            shape,
            strides,
            offset: 0,
            _marker: PhantomData,
        }
    }

    pub fn from_parts(storage: S, shape: [usize; R], strides: [usize; R], offset: usize) -> Self {
        Self {
            storage,
            shape,
            strides,
            offset,
            _marker: PhantomData,
        }
    }

    fn compute_strides(shape: &[usize; R]) -> [usize; R] {
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

impl<F: Real, B: Backend<F>, const R: usize> Evaluator<F, B, R> for Dense<F, R> {
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

// TODO: implement toeplitz(), zeros(), ones(), full(), eye()
// TODO: implement slice over axes
#[derive(Debug, Clone)]
pub struct Tensor<F: Real, Sh: Shape, const R: usize, E = Dense<F, R>> {
    pub expr: E,
    pub _marker: PhantomData<(F, Sh)>,
}

impl<F: Real, Sh: Shape, const R: usize, E> Tensor<F, Sh, R, E> {
    pub(crate) fn wrap(expr: E) -> Self {
        Self {
            expr,
            _marker: PhantomData,
        }
    }
}

impl<F: Real, const R: usize> Tensor<F, DynRank<R>, R, Dense<F, R>> {
    pub fn from_vec(data: Vec<F>, shape: [usize; R]) -> Self {
        debug_assert_eq!(data.len(), shape.iter().product::<usize>());
        Tensor::wrap(Dense::new(Arc::new(data), shape))
    }

    pub fn from_slice(data: &[F], shape: [usize; R]) -> Self {
        Self::from_vec(data.to_vec(), shape)
    }

    pub fn name_axes<Sh: Shape>(self) -> Tensor<F, Sh, R, Dense<F, R>> {
        debug_assert_eq!(
            Sh::RANK,
            R,
            "Rank mismatch between DynRank and semantic axes"
        );

        Tensor::wrap(self.expr)
    }
}

impl<F: Real, Sh: Shape, const R: usize> Tensor<F, Sh, R, Dense<F, R>> {
    pub fn from_expr<L>(input: L) -> Tensor<F, Sh, R, L::Output>
    where
        L: Lift<F>,
    {
        Tensor::wrap(input.lift())
    }
}

impl<F: Real, Sh: Shape, const R: usize, E> Tensor<F, Sh, R, E> {
    pub fn scale(self, factor: F) -> Tensor<F, Sh, R, ScaleExpr<E, F>> {
        Tensor::wrap(ScaleExpr {
            op: self.expr,
            factor,
        })
    }

    pub fn collect<B: Backend<F>>(&self, backend: &mut B) -> Base<B::Repr, F, R>
    where
        E: Evaluator<F, B, R>,
    {
        self.expr.eval(backend)
    }

    pub fn forward<B: Backend<F>>(
        &self,
        backend: &mut B,
    ) -> (Base<B::Repr, F, R>, GradientTape<E::Adjoint>)
    where
        E: Differentiable<F, B, R>,
    {
        let (res, adjoint) = self.expr.forward(backend);

        (res, GradientTape::new(adjoint))
    }

    pub fn to_vec<B: Backend<F>>(&self, backend: &mut B) -> Vec<F>
    where
        E: Evaluator<F, B, R>,
    {
        let view = self.expr.eval(backend);

        backend.to_host(&view.storage)
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

use super::Evaluator;
use algebra::{AddExpr, Lift, Real, Shape};
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

// TODO: implement toeplitz(), zeros(), ones(), full(), eye()
// TODO: implement slice over axes
#[derive(Debug, Clone)]
pub struct Tensor<F: Real, Sh: Shape, const R: usize, E = Dense<F, R>> {
    pub expr: E,
    pub _marker: PhantomData<(F, Sh)>,
}

impl<F: Real, Sh: Shape, const R: usize> Tensor<F, Sh, R, Dense<F, R>> {
    pub fn from_vec(data: Vec<F>, shape: [usize; R]) -> Self {
        debug_assert_eq!(R, Sh::RANK);
        debug_assert_eq!(data.len(), shape.iter().product::<usize>());

        Tensor {
            expr: Dense::new(Arc::new(data), shape),
            _marker: PhantomData,
        }
    }

    pub fn from_slice(data: &[F], shape: [usize; R]) -> Self {
        Self::from_vec(data.to_vec(), shape)
    }

    pub fn from_expr<L>(input: L) -> Tensor<F, Sh, R, L::Output>
    where
        L: Lift<F>,
    {
        Tensor {
            expr: input.lift(),
            _marker: PhantomData,
        }
    }
}

impl<F: Real, Sh: Shape, const R: usize, E> Tensor<F, Sh, R, E> {
    pub fn collect<B: Backend<F>>(&self, backend: &mut B) -> Base<B::Repr, F, R>
    where
        E: Evaluator<F, B, R>,
    {
        self.expr.eval(backend)
    }
}

use std::ops::Add;

impl<F, Sh, const R: usize, L, Rhs> Add<Tensor<F, Sh, R, Rhs>> for Tensor<F, Sh, R, L>
where
    F: Real,
    Sh: Shape,
{
    type Output = Tensor<F, Sh, R, AddExpr<L, Rhs>>;

    fn add(self, rhs: Tensor<F, Sh, R, Rhs>) -> Self::Output {
        Tensor {
            expr: AddExpr {
                left: self.expr,
                right: rhs.expr,
            },
            _marker: PhantomData,
        }
    }
}
// TODO: provide a default shape type (like (), or a DynRank struct)
// pub trait IntoTensor<F: Real, Sh: Shape, const R: usize> {
//     type Expr;
//     fn into_tensor(self) -> Tensor<F, Sh, R, Self::Expr>;
// }

// impl<F: Real, Sh: Shape, const N: usize> IntoTensor<F, Sh, 1> for [F; N] {
//     type Expr = DenseExpr<Vec<F>, 1>;

//     fn into_tensor(self) -> Tensor<F, Sh, 1, Self::Expr> {
//         Tensor::from_vec(self.to_vec(), [N])
//     }
// }

// impl<'a, F: Real, Sh: Shape> IntoTensor<F, Sh, 1> for &'a [F] {
//     type Expr = DenseExpr<Vec<F>, 1>;

//     fn into_tensor(self) -> Tensor<F, Sh, 1, Self::Expr> {
//         Tensor::from_slice(self, [self.len()])
//     }
// }

// impl<F: Real, Sh: Shape> IntoTensor<F, Sh, 1> for Vec<F> {
//     type Expr = DenseExpr<Vec<F>, 1>;

//     fn into_tensor(self) -> Tensor<F, Sh, 1, Self::Expr> {
//         let len = self.len();
//         Tensor::from_vec(self, [len])
//     }
// }

// Tensor with dynamic/unknown axis labels
// type DynamicTensor<const R: usize> = Tensor<TradingFloat, (Dynamic,), R>;

#[doc(hidden)]
#[macro_export]
macro_rules! __count {
    () => (0usize);
    ($head:expr $(, $tail:expr)*) => (1usize + $crate::__count!($($tail),*));
}

// TODO: Improve float conversion here.
// Just push the value. Rust inference will handle the conversion
// if F implements From<f64> or similar.
// Or use: $vec.push($crate::algebra::Real::from_f64($x as f64).unwrap());
#[doc(hidden)]
#[macro_export]
macro_rules! __flatten_1d {
    ($vec:ident; $($x:expr),* $(,)?) => {{
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
        let mut data = Vec::new();
        $crate::__flatten_1d!(data; $($x),*);
        let shape = [$crate::__count!($($x),*)];
        Tensor::from_vec(data, shape)
    }};

    ([$([$($x:expr),* $(,)?]),+ $(,)?]) => {{
        let mut data = Vec::new();
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
    use algebra::{Shape, TradingFloat};
    use backend::GenericBackend;

    struct TestShape;
    impl Shape for TestShape {
        const RANK: usize = 1;
        type Axes = ();
    }

    #[test]
    fn test_tensor_add() {
        let mut backend = GenericBackend::<TradingFloat>::new();
        let a: Tensor<TradingFloat, TestShape, 1> = tensor![2.0, 3.0, 5.0];
        let b: Tensor<TradingFloat, TestShape, 1> = tensor![1.0, 3.0, 2.0];

        let c = a + b;

        let result = c.collect(&mut backend);

        assert_eq!(
            result.shape,
            [3],
            "Result shape should be [3] for vector addition"
        );
    }
}

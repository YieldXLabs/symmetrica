use algebra::{EyeExpr, Lift, Real, Shape};
use backend::{Backend, Evaluator};
use std::{marker::PhantomData, sync::Arc};

// TODO: implement toeplitz(), zeros(), ones(), full()
// TODO: implement slice over axes
#[derive(Debug, Clone)]
pub struct TensorValue<S, F: Real, const R: usize> {
    pub storage: S,
    pub shape: [usize; R],
    pub strides: [usize; R],
    pub offset: usize,
    _marker: PhantomData<F>,
}

impl<S, F: Real, const R: usize> TensorValue<S, F, R> {
    pub fn new(storage: S, shape: [usize; R]) -> Self {
        let strides = compute_strides(&shape);

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
}

#[derive(Debug, Clone)]
pub struct DenseExpr<S, const R: usize> {
    pub data: Arc<S>,
    pub shape: [usize; R],
    pub strides: [usize; R],
    pub offset: usize,
}

#[derive(Debug, Clone)]
pub enum TensorExpr<S, Expr, const R: usize> {
    Dense(DenseExpr<S, R>),
    Algebraic(Expr),
}

#[derive(Debug, Clone)]
pub struct Tensor<F: Real, Sh: Shape, const R: usize, Expr = DenseExpr<Vec<F>, R>> {
    pub expr: TensorExpr<Vec<F>, Expr, R>,
    _marker: PhantomData<Sh>,
}

impl<F: Real, Sh: Shape, const R: usize> Tensor<F, Sh, R, ()> {
    pub fn from_expr<L>(input: L) -> Tensor<F, Sh, R, L::Output>
    where
        L: Lift<F>,
    {
        Tensor {
            expr: TensorExpr::Algebraic(input.lift()),
            _marker: PhantomData,
        }
    }
}

impl<F: Real, Sh: Shape, const R: usize> Tensor<F, Sh, R, DenseExpr<Vec<F>, R>> {
    pub fn from_vec(data: Vec<F>, shape: [usize; R]) -> Self {
        // assert_eq!(R, Sh::RANK, "Shape generic Sh does not match Rank R");
        assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "Data length mismatch"
        );

        let strides = compute_strides(&shape);

        Tensor {
            expr: TensorExpr::Dense(DenseExpr {
                data: Arc::new(data),
                shape,
                strides,
                offset: 0,
            }),
            _marker: PhantomData,
        }
    }

    pub fn from_slice(data: &[F], shape: [usize; R]) -> Self {
        Self::from_vec(data.to_vec(), shape)
    }
}

// TODO: provide a default shape type (like (), or a DynRank struct)
pub trait IntoTensor<F: Real, Sh: Shape, const R: usize> {
    type Expr;
    fn into_tensor(self) -> Tensor<F, Sh, R, Self::Expr>;
}

impl<F: Real, Sh: Shape, const N: usize> IntoTensor<F, Sh, 1> for [F; N] {
    type Expr = DenseExpr<Vec<F>, 1>;

    fn into_tensor(self) -> Tensor<F, Sh, 1, Self::Expr> {
        Tensor::from_vec(self.to_vec(), [N])
    }
}

impl<'a, F: Real, Sh: Shape> IntoTensor<F, Sh, 1> for &'a [F] {
    type Expr = DenseExpr<Vec<F>, 1>;

    fn into_tensor(self) -> Tensor<F, Sh, 1, Self::Expr> {
        Tensor::from_slice(self, [self.len()])
    }
}

impl<F: Real, Sh: Shape> IntoTensor<F, Sh, 1> for Vec<F> {
    type Expr = DenseExpr<Vec<F>, 1>;

    fn into_tensor(self) -> Tensor<F, Sh, 1, Self::Expr> {
        let len = self.len();
        Tensor::from_vec(self, [len])
    }
}

impl<F: Real, Sh: Shape> Tensor<F, Sh, 2, ()> {
    pub fn eye(n: usize) -> Tensor<F, Sh, 2, EyeExpr> {
        assert_eq!(Sh::RANK, 2);
        Tensor {
            expr: TensorExpr::Algebraic(EyeExpr { n }),
            _marker: PhantomData,
        }
    }
}

impl<F: Real, Sh: Shape, const R: usize, Expr> Tensor<F, Sh, R, Expr> {
    pub fn collect<B: Backend<F>>(&self, backend: &mut B) -> TensorValue<B::Repr, F, R>
    where
        Expr: Evaluator<F, B>,
        DenseExpr<Vec<F>, R>: Evaluator<F, B>,
    {
        match &self.expr {
            TensorExpr::Dense(dense) => {
                let (repr, _) = dense.eval(backend);

                TensorValue::from_parts(repr, dense.shape, dense.strides, dense.offset)
            }

            TensorExpr::Algebraic(expr) => {
                let (repr, shape_vec) = expr.eval(backend);
                let shape_arr: [usize; R] =
                    shape_vec.try_into().expect("Evaluator returned wrong rank");

                TensorValue::new(repr, shape_arr)
            }
        }
    }
}

fn compute_strides<const R: usize>(shape: &[usize; R]) -> [usize; R] {
    let mut strides = [0; R];
    let mut product = 1;
    for i in (0..R).rev() {
        strides[i] = product;
        product *= shape[i];
    }
    strides
}

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
    use algebra::TradingFloat;

    #[test]
    fn test_tensor_expr() {
        let a: Tensor<TradingFloat, (), 1> = tensor![2.0, 3.0, 5.0];

        match &a.expr {
            TensorExpr::Dense(dense) => {
                assert_eq!(dense.shape, [3], "Shape should be [3]");
                assert_eq!(dense.strides, [1], "Strides should be [1]");
                assert_eq!(dense.offset, 0, "Offset should be 0");

                let vec_data = &dense.data;

                assert_eq!(vec_data.len(), 3, "Data length mismatch");
                assert_eq!(vec_data[0], TradingFloat::try_from(2.0).unwrap());
                assert_eq!(vec_data[1], TradingFloat::try_from(3.0).unwrap());
                assert_eq!(vec_data[2], TradingFloat::try_from(5.0).unwrap());
            }
            _ => panic!("Expected Dense variant"),
        }
    }
}

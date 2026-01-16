use algebra::{EyeExpr, Lift, PureExpr, Real, Shape};
use backend::{Backend, Evaluator};
use std::marker::PhantomData;

// TODO: implement eye
// TODO: implement slice over axes
#[derive(Debug, Clone)]
pub struct TensorValue<S, F: Real, const R: usize> {
    pub storage: S,
    pub shape: [usize; R],
    pub strides: [usize; R],
    _marker: PhantomData<F>,
}

impl<S, F: Real, const R: usize> TensorValue<S, F, R> {
    pub fn new(storage: S, shape: [usize; R]) -> Self {
        let strides = Self::compute_strides(&shape);
        Self {
            storage,
            shape,
            strides,
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

#[derive(Debug, Clone)]
pub enum TensorExpr<'a, F: Real, Expr, const R: usize> {
    View {
        data: PureExpr<'a, F>,
        shape: [usize; R],
    },
    Owned {
        data: Vec<F>,
        shape: [usize; R],
    },
    Algebraic(Expr),
}

#[derive(Debug, Clone)]
pub struct Tensor<'a, F: Real, Sh: Shape, const R: usize, Expr = ()> {
    pub expr: TensorExpr<'a, F, Expr, R>,
    _marker: PhantomData<Sh>,
}

impl<'a, F: Real, Sh: Shape, const R: usize> Tensor<'a, F, Sh, R, ()> {
    pub fn from_vec(data: Vec<F>, shape: [usize; R]) -> Self {
        assert_eq!(R, Sh::RANK, "Shape generic Sh does not match Rank R");

        assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "Data length mismatch"
        );

        Tensor {
            expr: TensorExpr::Owned { data, shape },
            _marker: PhantomData,
        }
    }

    pub fn from_slice(data: &'a [F], shape: [usize; R]) -> Self {
        assert_eq!(R, Sh::RANK, "Shape generic Sh does not match Rank R");

        assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "Data length mismatch"
        );
        Tensor {
            expr: TensorExpr::View {
                data: data.lift(),
                shape,
            },
            _marker: PhantomData,
        }
    }

    pub fn from_expr<L>(input: L) -> Tensor<'a, F, Sh, R, L::Output>
    where
        L: Lift<'a, F>,
    {
        Tensor {
            expr: TensorExpr::Algebraic(input.lift()),
            _marker: PhantomData,
        }
    }
}

// TODO: provide a default shape type (like (), or a DynRank struct)
pub trait IntoTensor<'a, F: Real, Sh: Shape, const R: usize> {
    type Expr;
    fn into_tensor(self) -> Tensor<'a, F, Sh, R, Self::Expr>;
}

impl<F: Real, Sh: Shape, const N: usize> IntoTensor<'static, F, Sh, 1> for [F; N] {
    type Expr = ();
    fn into_tensor(self) -> Tensor<'static, F, Sh, 1, ()> {
        Tensor::from_vec(self.to_vec(), [N])
    }
}

impl<'a, F: Real, Sh: Shape> IntoTensor<'a, F, Sh, 1> for &'a [F] {
    type Expr = ();
    fn into_tensor(self) -> Tensor<'a, F, Sh, 1, ()> {
        Tensor::from_slice(self, [self.len()])
    }
}

impl<'a, F: Real, Sh: Shape> Tensor<'a, F, Sh, 2, ()> {
    pub fn eye(n: usize) -> Tensor<'a, F, Sh, 2, EyeExpr> {
        assert_eq!(Sh::RANK, 2, "Tensor::eye requires a Rank 2 Shape type");

        Tensor {
            expr: TensorExpr::Algebraic(EyeExpr { n }),
            _marker: PhantomData,
        }
    }
}

impl<'a, F: Real, Sh: Shape, const R: usize, Expr> Tensor<'a, F, Sh, R, Expr> {
    pub fn collect<B: Backend<F>>(&self, backend: &mut B) -> TensorValue<B::Repr, F, R>
    where
        Expr: Evaluator<F, B>,
    {
        match &self.expr {
            TensorExpr::View { data, shape } => {
                let repr = backend.pure(data.data);
                TensorValue::new(repr, *shape)
            }
            TensorExpr::Owned { data, shape } => {
                let repr = backend.pure(data.as_slice());
                TensorValue::new(repr, *shape)
            }
            TensorExpr::Algebraic(e) => {
                let (repr, shape_vec) = e.eval(backend);
                let shape_arr: [usize; R] =
                    shape_vec.try_into().expect("Evaluator returned wrong rank");
                TensorValue::new(repr, shape_arr)
            }
        }
    }
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
            TensorExpr::Owned { data, shape } => {
                assert_eq!(data.len(), 3, "Data length should be 3");
                assert_eq!(shape, &[3], "Shape should be [3]");
                assert_eq!(data[0], TradingFloat::try_from(2.0).unwrap());
                assert_eq!(data[1], TradingFloat::try_from(3.0).unwrap());
                assert_eq!(data[2], TradingFloat::try_from(5.0).unwrap());
            }
            _ => panic!("Expected Owned variant"),
        }
    }
}

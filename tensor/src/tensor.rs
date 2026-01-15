use algebra::{Lift, PureExpr, Real, Shape};
use backend::{Backend, Evaluator};
use std::convert::TryInto;
use std::marker::PhantomData;

// ------------------------
// 1. Materialized Tensor Value
// ------------------------
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

// ------------------------
// 2. Symbolic / hybrid tensor expression
// ------------------------
#[derive(Debug, Clone)]
pub enum TensorExpr<'a, F: Real, Expr, const R: usize> {
    Pure {
        data: PureExpr<'a, F>,
        shape: [usize; R],
    },
    Owned {
        data: Vec<F>,
        shape: [usize; R],
    },
    Algebraic(Expr),
}

// ------------------------
// 3. Main Tensor struct
// ------------------------
#[derive(Debug, Clone)]
pub struct Tensor<'a, F: Real, Sh: Shape, const R: usize, Expr = ()> {
    pub expr: TensorExpr<'a, F, Expr, R>,
    _marker: PhantomData<Sh>,
}

// ------------------------
// 4. Tensor constructors
// ------------------------
impl<'a, F: Real, Sh: Shape, const R: usize> Tensor<'a, F, Sh, R, ()> {
    pub fn from_vec(data: Vec<F>, shape: [usize; R]) -> Self {
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
        assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "Data length mismatch"
        );
        Tensor {
            expr: TensorExpr::Pure {
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

// ------------------------
// 5. IntoTensor trait
// ------------------------
pub trait IntoTensor<'a, F: Real, Sh: Shape, const R: usize> {
    type Expr;
    fn into_tensor(self) -> Tensor<'a, F, Sh, R, Self::Expr>;
}

impl<'a, F: Real, Sh: Shape, const R: usize> IntoTensor<'a, F, Sh, R> for Vec<F> {
    type Expr = ();
    fn into_tensor(self) -> Tensor<'a, F, Sh, R, ()> {
        Tensor::from_vec(self, [0; R]) // Shape must be provided manually
    }
}

impl<'a, F: Real, Sh: Shape, const R: usize> IntoTensor<'a, F, Sh, R> for &'a [F] {
    type Expr = ();
    fn into_tensor(self) -> Tensor<'a, F, Sh, R, ()> {
        Tensor::from_slice(self, [0; R]) // Shape must be provided manually
    }
}

// ------------------------
// 6. Collect method
// ------------------------
impl<'a, F: Real, Sh: Shape, const R: usize, Expr> Tensor<'a, F, Sh, R, Expr> {
    pub fn collect<B: Backend<F>>(&self, backend: &mut B) -> TensorValue<B::Repr, F, R>
    where
        Expr: Evaluator<F, B>,
    {
        match &self.expr {
            TensorExpr::Pure { data, shape } => {
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

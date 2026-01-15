use algebra::{Lift, PureExpr, Real};
use backend::{Backend, Evaluator};
use std::marker::PhantomData;

// 1. Materialized Tensor Value
pub struct TensorValue<S, F: Real> {
    pub storage: S,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    _marker: PhantomData<F>,
}

impl<S, F: Real> TensorValue<S, F> {
    pub fn new(storage: S, shape: Vec<usize>) -> Self {
        let strides = Self::compute_strides(&shape);
        Self {
            storage,
            shape,
            strides,
            _marker: PhantomData,
        }
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![0; shape.len()];
        let mut product = 1;
        for i in (0..shape.len()).rev() {
            strides[i] = product;
            product *= shape[i];
        }
        strides
    }
}

// 2. Symbolic / hybrid tensor expression
#[derive(Debug, Clone)]
pub enum TensorExpr<'a, F: Real, Expr> {
    Pure(PureExpr<'a, F>),
    Owned(Vec<F>),
    Algebraic(Expr),
}

#[derive(Debug, Clone)]
pub struct Tensor<'a, F: Real, Expr = ()> {
    pub expr: TensorExpr<'a, F, Expr>,
    _marker: PhantomData<&'a F>,
}

impl<'a, F: Real> Tensor<'a, F, ()> {
    pub fn from_vec(data: Vec<F>) -> Self {
        Tensor {
            expr: TensorExpr::Owned(data),
            _marker: PhantomData,
        }
    }

    pub fn from_slice(data: &'a [F]) -> Self {
        Tensor {
            expr: TensorExpr::Pure(data.lift()),
            _marker: PhantomData,
        }
    }

    pub fn from_expr<L>(input: L) -> Tensor<'a, F, L::Output>
    where
        L: Lift<'a, F>,
    {
        Tensor {
            expr: TensorExpr::Algebraic(input.lift()),
            _marker: PhantomData,
        }
    }
}

pub trait IntoTensor<'a, F: Real> {
    type Expr;
    fn into_tensor(self) -> Tensor<'a, F, Self::Expr>;
}

impl<'a, F: Real> IntoTensor<'a, F> for Vec<F> {
    type Expr = ();
    fn into_tensor(self) -> Tensor<'a, F, ()> {
        Tensor::from_vec(self)
    }
}

impl<'a, F: Real> IntoTensor<'a, F> for &'a [F] {
    type Expr = ();
    fn into_tensor(self) -> Tensor<'a, F, ()> {
        Tensor::from_slice(self)
    }
}

impl<'a, F: Real, Expr> Tensor<'a, F, Expr> {
    pub fn collect<B: Backend<F>>(&self, backend: &mut B) -> TensorValue<B::Repr, F>
    where
        Expr: Evaluator<F, B>,
    {
        match &self.expr {
            TensorExpr::Pure(p) => {
                let repr = backend.pure(p.data);
                TensorValue::new(repr, vec![p.data.len()])
            }
            TensorExpr::Owned(v) => {
                let repr = backend.pure(v);
                TensorValue::new(repr, vec![v.len()])
            }
            TensorExpr::Algebraic(e) => {
                let (storage, shape) = e.eval(backend);
                TensorValue::new(storage, shape)
            }
        }
    }
}

#[macro_export]
macro_rules! tensor {
    ($($x:expr),* $(,)?) => {{
        let data = vec![
            $(TradingFloat::try_from($x).expect("Invalid float")),*
        ];
        Tensor::from_vec(data)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use algebra::TradingFloat;

    #[test]
    fn test_tensor_expr() {
        let a = tensor![2.0, 3.0, 5.0];
        let arr = vec![
            TradingFloat::try_from(2.0).unwrap(),
            TradingFloat::try_from(5.0).unwrap(),
        ]; // owned
        let b = Tensor::from_vec(arr); // borrowed
    }
}

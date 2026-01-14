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
    Borrowed(&'a [F]),
    Owned(Vec<F>),
    Pure(PureExpr<'a, F>),
    Algebraic(Expr),
}

// 3. Main Tensor type
#[derive(Debug, Clone)]
pub struct Tensor<'a, F: Real, Expr = ()> {
    expr: TensorExpr<'a, F, Expr>,
    _marker: PhantomData<&'a F>,
}

impl<'a, F: Real> Tensor<'a, F, ()> {
    /// Borrowed slice
    pub fn from_slice(slice: &'a [F]) -> Self {
        Self {
            expr: TensorExpr::Borrowed(slice),
            _marker: PhantomData,
        }
    }

    /// Owned vector
    pub fn from_vec(vec: Vec<F>) -> Self {
        Self {
            expr: TensorExpr::Owned(vec),
            _marker: PhantomData,
        }
    }

    /// Lift symbolic / algebraic type
    pub fn from_lift<L>(input: L) -> Tensor<'a, F, L::Output>
    where
        L: Lift<'a, F>,
    {
        Tensor {
            expr: TensorExpr::Algebraic(input.lift()),
            _marker: PhantomData,
        }
    }

    /// Try to get a slice if available (borrowed or owned)
    pub fn as_slice(&self) -> Option<&[F]> {
        match &self.expr {
            TensorExpr::Borrowed(s) => Some(s),
            TensorExpr::Owned(v) => Some(&v[..]),
            TensorExpr::Pure(p) => Some(p.data),
            TensorExpr::Algebraic(_) => None,
        }
    }
}

impl<'a, F: Real, Expr> Tensor<'a, F, Expr> {
    pub fn new(expr: Expr) -> Self {
        Self {
            expr: TensorExpr::Algebraic(expr),
            _marker: PhantomData,
        }
    }

    pub fn collect<B: Backend<F>>(&self, backend: &mut B) -> TensorValue<B::Repr, F>
    where
        Expr: Evaluator<F, B>,
    {
        if let TensorExpr::Algebraic(e) = &self.expr {
            let (storage, shape) = e.eval(backend);
            TensorValue::new(storage, shape)
        } else {
            panic!("collect() only works on symbolic / algebraic tensors")
        }
    }
}

// Macro for owned tensors
#[macro_export]
macro_rules! tensor {
    ($($x:expr),* $(,)?) => {{
        let data = vec![$(
            TradingFloat::try_from($x).expect("Invalid float")
        ),*];
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
        let arr = [
            TradingFloat::try_from(2.0).unwrap(),
            TradingFloat::try_from(5.0).unwrap(),
        ]; // owned
        let b = Tensor::from_slice(&arr[..]); // borrowed

        assert_eq!(
            a.as_slice().unwrap(),
            &[2.0, 3.0, 5.0].map(TradingFloat::new)
        );
        assert_eq!(b.as_slice().unwrap(), &[1.0, 2.0].map(TradingFloat::new));
    }
}

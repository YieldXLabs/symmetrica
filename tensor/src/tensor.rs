use algebra::{Lift, Real};
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

// 2. Symbolic
#[derive(Debug, Clone, Copy)]
pub struct Tensor<'a, F: Real, Expr> {
    pub(crate) expr: Expr,
    _marker: PhantomData<&'a F>,
}

impl<'a, F: Real> Tensor<'a, F, ()> {
    pub fn from<L>(input: L) -> Tensor<'a, F, L::Output>
    where
        L: Lift<'a, F>,
    {
        Tensor::new(input.lift())
    }
}

impl<'a, F: Real, Expr> Tensor<'a, F, Expr> {
    pub fn new(expr: Expr) -> Self {
        Self {
            expr,
            _marker: PhantomData,
        }
    }

    pub fn collect<B: Backend<F>>(&self, backend: &mut B) -> TensorValue<B::Repr, F>
    where
        Expr: Evaluator<F, B>,
    {
        let (storage, shape) = self.expr.eval(backend);

        TensorValue::new(storage, shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use algebra::TradingFloat;

    #[test]
    fn test_scale_ops() {
        let data: [TradingFloat; 3] = [2.0, 3.0, 5.0].map(|x| TradingFloat::try_from(x).unwrap());

        let a = Tensor::from(&data[..]);

        assert_eq!(0, 0);
    }
}

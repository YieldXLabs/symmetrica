use super::Evaluator;
use algebra::{AddExpr, Lift, Real, Shape};
use backend::Backend;
use std::{marker::PhantomData, sync::Arc};

// TODO: implement toeplitz(), zeros(), ones(), full(), eye()
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
pub enum ExprNode<E, S, const R: usize> {
    Value(DenseExpr<S, R>),
    Op(E),
}

#[derive(Debug, Clone)]
pub struct Tensor<F: Real, Sh: Shape, const R: usize, Expr = DenseExpr<Vec<F>, R>> {
    pub expr: ExprNode<Expr, Vec<F>, R>,
    _marker: PhantomData<Sh>,
}

impl<F: Real, Sh: Shape, const R: usize, Expr> Tensor<F, Sh, R, Expr> {
    fn into_expr(self) -> ExprNode<Expr, Vec<F>, R> {
        self.expr
    }
}

type Dense<F, const R: usize> = DenseExpr<Vec<F>, R>;

impl<F: Real, Sh: Shape, const R: usize> Tensor<F, Sh, R, Dense<F, R>> {
    fn from_dense(dense: Dense<F, R>) -> Self {
        Tensor {
            expr: ExprNode::Value(dense),
            _marker: PhantomData,
        }
    }
}

impl<F: Real, Sh: Shape, const R: usize> Tensor<F, Sh, R> {
    pub fn from_vec(data: Vec<F>, shape: [usize; R]) -> Tensor<F, Sh, R, Dense<F, R>> {
        debug_assert_eq!(R, Sh::RANK);
        debug_assert_eq!(data.len(), shape.iter().product::<usize>());

        let strides = compute_strides(&shape);

        Tensor::from_dense(DenseExpr {
            data: Arc::new(data),
            shape,
            strides,
            offset: 0,
        })
    }

    pub fn from_slice(data: &[F], shape: [usize; R]) -> Tensor<F, Sh, R, Dense<F, R>> {
        Self::from_vec(data.to_vec(), shape)
    }

    pub fn from_expr<L>(input: L) -> Tensor<F, Sh, R, L::Output>
    where
        L: Lift<F>,
    {
        Tensor {
            expr: ExprNode::Op(input.lift()),
            _marker: PhantomData,
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

impl<F, B, E, const R: usize> Evaluator<F, B> for ExprNode<E, Vec<F>, R>
where
    F: Real,
    B: Backend<F>,
    E: Evaluator<F, B>,
    DenseExpr<Vec<F>, R>: Evaluator<F, B>,
{
    fn eval(&self, backend: &mut B) -> (B::Repr, Vec<usize>) {
        match self {
            ExprNode::Value(dense) => dense.eval(backend),
            ExprNode::Op(expr) => expr.eval(backend),
        }
    }
}

impl<F: Real, Sh: Shape, const R: usize, Expr> Tensor<F, Sh, R, Expr> {
    pub fn collect<B: Backend<F>>(&self, backend: &mut B) -> TensorValue<B::Repr, F, R>
    where
        ExprNode<Expr, Vec<F>, R>: Evaluator<F, B>,
    {
        let (repr, shape_vec) = self.expr.eval(backend);

        let shape_arr: [usize; R] = shape_vec.try_into().expect("Evaluator returned wrong rank");

        TensorValue::new(repr, shape_arr)
    }
}

use std::ops::Add;

impl<F, Sh, const R: usize, L, Rhs> Add<Tensor<F, Sh, R, Rhs>> for Tensor<F, Sh, R, L>
where
    F: Real,
    Sh: Shape,
{
    type Output = Tensor<F, Sh, R, AddExpr<ExprNode<L, Vec<F>, R>, ExprNode<Rhs, Vec<F>, R>>>;

    fn add(self, rhs: Tensor<F, Sh, R, Rhs>) -> Self::Output {
        Tensor {
            expr: ExprNode::Op(AddExpr {
                left: self.into_expr(),
                right: rhs.into_expr(),
            }),
            _marker: PhantomData,
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
    use algebra::{TradingFloat, Untyped};
    use backend::GenericBackend;

    #[test]
    fn test_tensor_add() {
        let mut backend = GenericBackend::<TradingFloat>::new();
        let a: Tensor<TradingFloat, (Untyped,), 1> = tensor![2.0, 3.0, 5.0];
        let b: Tensor<TradingFloat, (Untyped,), 1> = tensor![1.0, 3.0, 2.0];

        let c = a + b;

        let result = c.collect(&mut backend);

        assert_eq!(
            result.shape,
            [3],
            "Result shape should be [3] for vector addition"
        );
    }
}

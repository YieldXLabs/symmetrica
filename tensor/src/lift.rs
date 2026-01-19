use super::Tensor;
use algebra::{ConstExpr, Real, Shape};

pub trait Lift<F: Real> {
    type Output;

    fn lift(self) -> Self::Output;
}

impl<F: Real> Lift<F> for F {
    type Output = ConstExpr<F>;

    fn lift(self) -> Self::Output {
        ConstExpr(self)
    }
}

impl<F: Real, Sh: Shape, const R: usize, E> Lift<F> for Tensor<F, Sh, R, E> {
    type Output = E;

    fn lift(self) -> Self::Output {
        self.expr
    }
}

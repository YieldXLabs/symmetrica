use algebra::{ConstExpr, Real};

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

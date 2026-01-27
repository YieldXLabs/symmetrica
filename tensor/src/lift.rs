use algebra::{ConstExpr, Data};

pub trait Lift<F: Data> {
    type Output;

    fn lift(self) -> Self::Output;
}

impl<F: Data> Lift<F> for F {
    type Output = ConstExpr<F>;

    fn lift(self) -> Self::Output {
        ConstExpr(self)
    }
}

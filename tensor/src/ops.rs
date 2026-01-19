use super::{Lift, Tensor};
use algebra::{AddExpr, Real, Shape};
use std::{marker::PhantomData, ops::Add};

impl<F, Sh, const R: usize, L, Rhs> Add<Rhs> for Tensor<F, Sh, R, L>
where
    F: Real,
    Sh: Shape,
    Rhs: Lift<F>,
{
    type Output = Tensor<F, Sh, R, AddExpr<L, Rhs::Output>>;

    fn add(self, rhs: Rhs) -> Self::Output {
        Tensor {
            expr: AddExpr {
                left: self.expr,
                right: rhs.lift(),
            },
            _marker: PhantomData,
        }
    }
}

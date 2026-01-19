use super::{Lift, Tensor};
use algebra::{AddExpr, ConstExpr, Real, Shape};
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

impl<F, Sh, const R: usize, RhsExpr> Add<Tensor<F, Sh, R, RhsExpr>> for ConstExpr<F>
where
    F: Real,
    Sh: Shape,
{
    type Output = Tensor<F, Sh, R, AddExpr<Self, RhsExpr>>;

    fn add(self, rhs: Tensor<F, Sh, R, RhsExpr>) -> Self::Output {
        Tensor {
            expr: AddExpr {
                left: self,
                right: rhs.expr,
            },
            _marker: PhantomData,
        }
    }
}

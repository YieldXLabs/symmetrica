use super::Tensor;
use algebra::{AddExpr, Real, Shape};
use std::{marker::PhantomData, ops::Add};

impl<F, Sh, const R: usize, L, Rhs> Add<Tensor<F, Sh, R, Rhs>> for Tensor<F, Sh, R, L>
where
    F: Real,
    Sh: Shape,
{
    type Output = Tensor<F, Sh, R, AddExpr<L, Rhs>>;

    fn add(self, rhs: Tensor<F, Sh, R, Rhs>) -> Self::Output {
        Tensor {
            expr: AddExpr {
                left: self.expr,
                right: rhs.expr,
            },
            _marker: PhantomData,
        }
    }
}

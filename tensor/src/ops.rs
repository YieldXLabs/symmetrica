use super::{Lift, Tensor};
use algebra::{
    AddExpr, BroadcastShape, CanBroadcastWith, ConstExpr, Real, Shape, SubExpr, True, TypeEq,
};
use std::ops::{Add, Sub};

impl<F, ShL, ShR, const R: usize, EL, ER> Add<Tensor<F, ShR, R, ER>> for Tensor<F, ShL, R, EL>
where
    F: Real,
    ShL: Shape,
    ShR: Shape,
    ShL: CanBroadcastWith<ShR>,
    <ShL as CanBroadcastWith<ShR>>::Result: TypeEq<True>,
    ShL: BroadcastShape<ShR>,
{
    type Output = Tensor<F, <ShL as BroadcastShape<ShR>>::Output, R, AddExpr<EL, ER>>;

    fn add(self, rhs: Tensor<F, ShR, R, ER>) -> Self::Output {
        Tensor::wrap(AddExpr {
            left: self.expr,
            right: rhs.expr,
        })
    }
}

impl<F, Sh, const R: usize, RhsExpr> Add<Tensor<F, Sh, R, RhsExpr>> for ConstExpr<F>
where
    F: Real,
    Sh: Shape,
{
    type Output = Tensor<F, Sh, R, AddExpr<Self, RhsExpr>>;

    fn add(self, rhs: Tensor<F, Sh, R, RhsExpr>) -> Self::Output {
        Tensor::wrap(AddExpr {
            left: self,
            right: rhs.expr,
        })
    }
}

impl<F, Sh, const R: usize, L, Rhs> Sub<Rhs> for Tensor<F, Sh, R, L>
where
    F: Real,
    Sh: Shape,
    Rhs: Lift<F>,
{
    type Output = Tensor<F, Sh, R, SubExpr<L, Rhs::Output>>;

    fn sub(self, rhs: Rhs) -> Self::Output {
        Tensor::wrap(SubExpr {
            left: self.expr,
            right: rhs.lift(),
        })
    }
}

impl<F, Sh, const R: usize, RhsExpr> Sub<Tensor<F, Sh, R, RhsExpr>> for ConstExpr<F>
where
    F: Real,
    Sh: Shape,
{
    type Output = Tensor<F, Sh, R, SubExpr<Self, RhsExpr>>;

    fn sub(self, rhs: Tensor<F, Sh, R, RhsExpr>) -> Self::Output {
        Tensor::wrap(SubExpr {
            left: self,
            right: rhs.expr,
        })
    }
}

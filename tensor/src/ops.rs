use super::{Lift, Tensor};
use algebra::{
    AddExpr, BroadcastShape, CanBroadcastWith, ConstExpr, Real, Shape, SubExpr, True, TypeEq,
};
use std::ops::{Add, Sub};

// TODO: Auto broadcast
// fn infer_union_shape<const RL: usize, const RR: usize, const ROUT: usize>(
//     shape_l: &[usize; RL],
//     shape_r: &[usize; RR],
//     map_l: &[Option<usize>],
// ) -> [usize; ROUT] {
//     let mut out = [0; ROUT];
    
//     for i in 0..ROUT {
//         let dim_l = map_l[i].map(|idx| shape_l[idx]).unwrap_or(1);
//         let dim_r = map_r[i].map(|idx| shape_r[idx]).unwrap_or(1);
        
//         if dim_l == dim_r {
//             out[i] = dim_l;
//         } else if dim_l == 1 {
//             out[i] = dim_r;
//         } else if dim_r == 1 {
//             out[i] = dim_l;
//         } else {
//             panic!("Runtime broadcast mismatch at dim {}: {} vs {}", i, dim_l, dim_r);
//         }
//     }
//     out
// }

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

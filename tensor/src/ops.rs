use super::Tensor;
use algebra::{AddExpr, BroadcastShape, CanBroadcastWith, ConstExpr, Real, Shape, True, TypeEq};
use std::ops::Add;

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

impl<F, ShL, ShR, EL, ER> Add<Tensor<F, ShR, ER>> for Tensor<F, ShL, EL>
where
    F: Real,
    ShL: Shape,
    ShR: Shape,
    ShL: CanBroadcastWith<ShR>,
    <ShL as CanBroadcastWith<ShR>>::Result: TypeEq<True>,
    ShL: BroadcastShape<ShR>,
    <ShL as BroadcastShape<ShR>>::Output: Shape,
    [(); <ShL as BroadcastShape<ShR>>::Output::RANK]: Sized,
{
    type Output = Tensor<F, <ShL as BroadcastShape<ShR>>::Output, AddExpr<EL, ER>>;

    fn add(self, rhs: Tensor<F, ShR, ER>) -> Self::Output {
        Tensor::wrap(AddExpr {
            left: self.expr,
            right: rhs.expr,
        })
    }
}

impl<F, Sh, RhsExpr> Add<Tensor<F, Sh, RhsExpr>> for ConstExpr<F>
where
    F: Real,
    Sh: Shape,
{
    type Output = Tensor<F, Sh, AddExpr<Self, RhsExpr>>;

    fn add(self, rhs: Tensor<F, Sh, RhsExpr>) -> Self::Output {
        Tensor::wrap(AddExpr {
            left: self,
            right: rhs.expr,
        })
    }
}

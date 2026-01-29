// Free algebra expression types
#[derive(Debug, Clone)]
pub struct IdentityExpr;

#[derive(Debug, Clone)]
pub struct ConstExpr<F>(pub F);

#[derive(Debug, Clone)]
pub struct LoadExpr<F>(pub F);

#[derive(Debug, Clone)]
pub struct LetExpr<Val, Body> {
    pub value: Val,
    pub body: Body,
}

#[derive(Debug, Clone)]
pub struct IfExpr<Cond, Then, Else> {
    pub cond: Cond,
    pub then_: Then,
    pub else_: Else,
}

#[derive(Debug, Clone)]
pub struct BroadcastExpr<Op, const R_IN: usize, const R_OUT: usize> {
    pub op: Op,
    pub target_shape: [usize; R_OUT],
    pub mapping: [Option<usize>; R_OUT],
}

#[derive(Debug, Clone)]
pub struct TransposeExpr<Op, const R: usize> {
    pub op: Op,
    pub perm: [usize; R],
}

#[derive(Debug, Clone)]
pub struct ReshapeExpr<Op, const R_IN: usize, const R_OUT: usize> {
    pub op: Op,
    pub new_shape: [usize; R_OUT],
}

#[derive(Debug, Clone)]
pub struct SelectExpr<Op> {
    pub op: Op,
    pub axis: usize,
    pub index: usize,
}

#[derive(Debug, Clone)]
pub struct ZipExpr<L, R, K> {
    pub left: L,
    pub right: R,
    pub kernel: K,
}

#[derive(Debug, Clone)]
pub struct MapExpr<Op, K> {
    pub op: Op,
    pub kernel: K,
}

#[derive(Debug, Clone)]
pub struct ContractExpr<L, R> {
    pub left: L,
    pub right: R,
    pub axes: (Vec<usize>, Vec<usize>),
}

#[derive(Debug, Clone)]
pub struct ScanExpr<Op, F, K> {
    pub op: Op,
    pub init: F,
    pub kernel: K,
}

// TODO: Distance metrics
// pub trait DistanceMetric<F: Real>: Copy + Clone + 'static {}

// #[derive(Debug, Clone, Copy)]
// pub struct Euclidean;
// impl<F: Real> DistanceMetric<F> for Euclidean {}

// #[derive(Debug, Clone, Copy)]
// pub struct CDistExpr<L, R, M> {
//     pub left: L,
//     pub right: R,
//     _metric: PhantomData<M>,
// }

// TODO: Match exprs
// MatchExpr::new(|expr| matches!(expr, Add(_, Zero) => true));
// MatchExpr::new(|expr| matches!(expr, Sub(x, x) => true));
// MatchExpr::new(|expr| matches!(expr, Scale(Add(a,b), k) => Add(Scale(a,k), Scale(b,k))));

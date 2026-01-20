// Free algebra expression types
#[derive(Debug, Clone)]
pub struct IdentityExpr;

#[derive(Debug, Clone)]
pub struct ConstExpr<F>(pub F);

#[derive(Debug, Clone)]
pub struct LetExpr<Val, Body> {
    pub value: Val,
    pub body: Body,
}

// Linear operators
#[derive(Debug, Clone)]
pub struct AddExpr<L, R> {
    pub left: L,
    pub right: R,
}

#[derive(Debug, Clone)]
pub struct SubExpr<L, R> {
    pub left: L,
    pub right: R,
}

#[derive(Debug, Clone)]
pub struct MulExpr<L, R> {
    pub left: L,
    pub right: R,
}

#[derive(Debug, Clone)]
pub struct DivExpr<L, R> {
    pub left: L,
    pub right: R,
}

#[derive(Debug, Clone)]
pub struct ScaleExpr<Op, F> {
    pub op: Op,
    pub factor: F,
}

// Nonlinear operators
#[derive(Debug, Clone)]
pub struct AbsExpr<Op> {
    pub op: Op,
}

#[derive(Debug, Clone)]
pub struct ExpExpr<Op> {
    pub op: Op,
}

#[derive(Debug, Clone)]
pub struct LogExpr<Op> {
    pub op: Op,
}

#[derive(Debug, Clone)]
pub struct SqrtExpr<Op> {
    pub op: Op,
}

#[derive(Debug, Clone)]
pub struct PowExpr<Op, F> {
    pub op: Op,
    pub exp: F,
}

#[derive(Debug, Clone)]
pub struct SinExpr<Op> {
    pub op: Op,
}

#[derive(Debug, Clone)]
pub struct CosExpr<Op> {
    pub op: Op,
}

// Time series
#[derive(Debug, Clone)]
pub struct LagExpr<Op> {
    pub op: Op,
    pub n: usize,
}

// Statistical
#[derive(Debug, Clone)]
pub struct GaussianExpr<Op, F> {
    pub x: Op,
    pub mean: F,
    pub sigma: F,
}

// Control / branching
#[derive(Debug, Clone)]
pub struct IfExpr<Cond, Then, Else> {
    pub cond: Cond,
    pub then_: Then,
    pub else_: Else,
}

// Comparison
#[derive(Debug, Clone)]
pub struct MinExpr<L, R> {
    pub left: L,
    pub right: R,
}

#[derive(Debug, Clone)]
pub struct MaxExpr<L, R> {
    pub left: L,
    pub right: R,
}

#[derive(Debug, Clone)]
pub struct ClampExpr<Op, F> {
    pub op: Op,
    pub lo: F,
    pub hi: F,
}

#[derive(Debug, Clone)]
pub struct LtExpr<L, R> {
    pub left: L,
    pub right: R,
}

#[derive(Debug, Clone)]
pub struct GtExpr<L, R> {
    pub left: L,
    pub right: R,
}

#[derive(Debug, Clone)]
pub struct EqExpr<L, R> {
    pub left: L,
    pub right: R,
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

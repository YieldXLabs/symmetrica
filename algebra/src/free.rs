use super::Real;

pub trait Lift<'a, F: Real> {
    type Output;

    fn lift(self) -> Self::Output;
}

// Leaf node: reference to slice of data
#[derive(Debug, Clone, Copy)]
pub struct PureExpr<'a, F: Real> {
    pub data: &'a [F],
}

impl<'a, F: Real> Lift<'a, F> for &'a [F] {
    type Output = PureExpr<'a, F>;

    fn lift(self) -> Self::Output {
        PureExpr { data: self }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ConstExpr<F>(pub F);

#[derive(Debug, Clone, Copy)]
pub struct LetExpr<Val, Body> {
    pub value: Val,
    pub body: Body,
}

// Linear operators
#[derive(Debug, Clone, Copy)]
pub struct AddExpr<L, R> {
    pub left: L,
    pub right: R,
}

#[derive(Debug, Clone, Copy)]
pub struct SubExpr<L, R> {
    pub left: L,
    pub right: R,
}

#[derive(Debug, Clone, Copy)]
pub struct MulExpr<L, R> {
    pub left: L,
    pub right: R,
}

#[derive(Debug, Clone, Copy)]
pub struct DivExpr<L, R> {
    pub left: L,
    pub right: R,
}

#[derive(Debug, Clone, Copy)]
pub struct ScaleExpr<Op, F> {
    pub op: Op,
    pub factor: F,
}

// Nonlinear operators
#[derive(Debug, Clone, Copy)]
pub struct AbsExpr<Op> {
    pub op: Op,
}

#[derive(Debug, Clone, Copy)]
pub struct ExpExpr<Op> {
    pub op: Op,
}

#[derive(Debug, Clone, Copy)]
pub struct LogExpr<Op> {
    pub op: Op,
}

#[derive(Debug, Clone, Copy)]
pub struct SqrtExpr<Op> {
    pub op: Op,
}

#[derive(Debug, Clone, Copy)]
pub struct PowExpr<Op, F> {
    pub op: Op,
    pub exp: F,
}

#[derive(Debug, Clone, Copy)]
pub struct SinExpr<Op> {
    pub op: Op,
}

#[derive(Debug, Clone, Copy)]
pub struct CosExpr<Op> {
    pub op: Op,
}

// Time series
#[derive(Debug, Clone, Copy)]
pub struct LagExpr<Op> {
    pub op: Op,
    pub n: usize,
}

// Statistical
#[derive(Debug, Clone, Copy)]
pub struct GaussianExpr<Op, F> {
    pub x: Op,
    pub mean: F,
    pub sigma: F,
}

// Control / branching
#[derive(Debug, Clone, Copy)]
pub struct IfExpr<Cond, Then, Else> {
    pub cond: Cond,
    pub then_: Then,
    pub else_: Else,
}

// Comparison
#[derive(Debug, Clone, Copy)]
pub struct MinExpr<L, R> {
    pub left: L,
    pub right: R,
}

#[derive(Debug, Clone, Copy)]
pub struct MaxExpr<L, R> {
    pub left: L,
    pub right: R,
}

#[derive(Debug, Clone, Copy)]
pub struct ClampExpr<Op, F> {
    pub op: Op,
    pub lo: F,
    pub hi: F,
}

#[derive(Debug, Clone, Copy)]
pub struct LtExpr<L, R> {
    pub left: L,
    pub right: R,
}

#[derive(Debug, Clone, Copy)]
pub struct GtExpr<L, R> {
    pub left: L,
    pub right: R,
}

#[derive(Debug, Clone, Copy)]
pub struct EqExpr<L, R> {
    pub left: L,
    pub right: R,
}

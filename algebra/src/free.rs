use super::Real;

pub trait Lift<'a, F: Real> {
    type Output;

    fn lift(self) -> Self::Output;
}

// Leaf node: reference to slice of data
#[derive(Debug, Clone, Copy)]
pub struct Pure<'a, F: Real> {
    pub data: &'a [F],
}

impl<'a, F: Real> Lift<'a, F> for &'a [F] {
    type Output = Pure<'a, F>;

    fn lift(self) -> Self::Output {
        Pure { data: self }
    }
}

// Linear operators
#[derive(Debug, Clone, Copy)]
pub struct Add<L, R> {
    pub left: L,
    pub right: R,
}

#[derive(Debug, Clone, Copy)]
pub struct Sub<L, R> {
    pub left: L,
    pub right: R,
}

#[derive(Debug, Clone, Copy)]
pub struct Mul<L, R> {
    pub left: L,
    pub right: R,
}

#[derive(Debug, Clone, Copy)]
pub struct Div<L, R> {
    pub left: L,
    pub right: R,
}

#[derive(Debug, Clone, Copy)]
pub struct Scale<Op, F> {
    pub op: Op,
    pub factor: F,
}

// Nonlinear operators
#[derive(Debug, Clone, Copy)]
pub struct Abs<Op> {
    pub op: Op,
}

#[derive(Debug, Clone, Copy)]
pub struct Exp<Op> {
    pub op: Op,
}

#[derive(Debug, Clone, Copy)]
pub struct Log<Op> {
    pub op: Op,
}

#[derive(Debug, Clone, Copy)]
pub struct Sqrt<Op> {
    pub op: Op,
}

#[derive(Debug, Clone, Copy)]
pub struct Pow<Op, F> {
    pub op: Op,
    pub exp: F,
}

#[derive(Debug, Clone, Copy)]
pub struct Sin<Op> {
    pub op: Op,
}

#[derive(Debug, Clone, Copy)]
pub struct Cos<Op> {
    pub op: Op,
}

// Time series
#[derive(Debug, Clone, Copy)]
pub struct Lag<Op> {
    pub op: Op,
    pub n: usize,
}

// Statistical
#[derive(Debug, Clone, Copy)]
pub struct Gaussian<Op, F> {
    pub x: Op,
    pub mean: F,
    pub sigma: F,
}

// Control / branching
#[derive(Debug, Clone, Copy)]
pub struct If<Cond, Then, Else> {
    pub cond: Cond,
    pub then_: Then,
    pub else_: Else,
}

// Comparison
#[derive(Debug, Clone, Copy)]
pub struct Min<L, R> {
    pub left: L,
    pub right: R,
}

#[derive(Debug, Clone, Copy)]
pub struct Max<L, R> {
    pub left: L,
    pub right: R,
}

#[derive(Debug, Clone, Copy)]
pub struct Clamp<Op, F> {
    pub op: Op,
    pub lo: F,
    pub hi: F,
}

#[derive(Debug, Clone, Copy)]
pub struct Lt<L, R> {
    pub left: L,
    pub right: R,
}

#[derive(Debug, Clone, Copy)]
pub struct Gt<L, R> {
    pub left: L,
    pub right: R,
}

#[derive(Debug, Clone, Copy)]
pub struct Eq<L, R> {
    pub left: L,
    pub right: R,
}

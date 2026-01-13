use std::cmp::Ord;
use std::fmt::{Debug, Display};
use std::iter::{Product, Sum};
use std::ops::{Add, Div, Mul, Neg, Sub};

// 0. Identity
// The additive identity (0)
pub trait Zero {
    fn zero() -> Self;
}
// The multiplicative identity (1)
pub trait One {
    fn one() -> Self;
}

// 1. Additive Group (Abelian Group)
pub trait AdditiveGroup:
    Clone
    + PartialEq
    + Debug
    + Default
    + Zero
    + Add<Output = Self>
    + Sub<Output = Self>
    + Neg<Output = Self>
    + Sum
{
}

// 2. Ring
pub trait Ring: AdditiveGroup + One + Mul<Output = Self> + Product {}

// 3. Field
pub trait Field: Ring + Copy + Div<Output = Self> {
    fn recip(self) -> Self;
}

// Extensions: Ordering and Real Analysis
pub trait OrderedField: Field + PartialOrd + Ord {
    fn abs(self) -> Self;
    fn signum(self) -> Self;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn clamp(self, lo: Self, hi: Self) -> Self;
}

pub trait Real: OrderedField {
    fn pi() -> Self;
    fn e() -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn pow(self, exp: Self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
}

pub trait Discretization: Sized {
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;
}

pub trait VectorSpace<F: Field>: AdditiveGroup {
    fn scale(&self, scalar: F) -> Self;
}

pub trait InnerProductSpace<F: Real>: VectorSpace<F> {
    fn inner(&self, other: &Self) -> F;

    fn norm(&self) -> F {
        self.inner(self).sqrt()
    }
}

pub trait LinearMap<V, W> {
    fn apply(&self, v: &V) -> W;
}

pub trait AdjointLinearMap<V, W>: LinearMap<V, W> {
    type Adjoint: LinearMap<W, V>;
    fn adjoint(&self) -> Self::Adjoint;
}

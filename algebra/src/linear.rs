use std::ops::Add;

use super::traits::{Data, Real, Semiring, Zero};

// Generalized Linear Algebra
pub trait Module<S: Semiring>: Data + Add<Output = Self> + Zero {
    fn scale(&self, scalar: S) -> Self;
}

// Projection
pub trait DotProduct<S: Semiring>: Module<S> {
    fn dot(&self, other: &Self) -> S;
}

pub trait Normed<S: Real>: Module<S> {
    fn norm(&self) -> S;
}

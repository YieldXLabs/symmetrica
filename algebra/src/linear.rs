use super::traits::{AdditiveGroup, Field, Real};

// Linear Algebra
pub trait VectorSpace<F: Field>: AdditiveGroup {
    fn scale(&self, scalar: F) -> Self;
    fn dim(&self) -> usize;
}

// Geometry
pub trait InnerProductSpace<F: Real>: VectorSpace<F> {
    fn inner(&self, other: &Self) -> F;
}

// Structure preserving operators
pub trait LinearMap<F: Field, V: VectorSpace<F>, W: VectorSpace<F>> {
    fn apply(&self, input: &V) -> W;
}

// Duality
pub trait Adjoint<F: Real, V: InnerProductSpace<F>, W: InnerProductSpace<F>>:
    LinearMap<F, V, W>
{
    type AdjointOp: LinearMap<F, W, V>;
    fn adjoint(&self) -> Self::AdjointOp;
}

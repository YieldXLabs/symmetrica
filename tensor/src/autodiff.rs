use super::{Base, LeafAdjoint, Pullback};
use algebra::Real;
use backend::Backend;

impl<F: Real, B: Backend<F>, const R: usize> Pullback<F, B, R> for LeafAdjoint {
    type Gradients = ();

    fn back(&self, _b: &mut B, _g: Base<B::Repr, F, R>) -> () {
        ()
    }
}

pub struct GradientTape<A> {
    adjoint: A,
}

impl<A> GradientTape<A> {
    pub fn new(adjoint: A) -> Self {
        Self { adjoint }
    }
}

impl<A> GradientTape<A> {
    pub fn backward<F, B, const R: usize>(
        self,
        backend: &mut B,
        seed_grad: Base<B::Repr, F, R>,
    ) -> A::Gradients
    where
        F: Real,
        B: Backend<F>,
        A: Pullback<F, B, R>,
    {
        self.adjoint.back(backend, seed_grad)
    }
}

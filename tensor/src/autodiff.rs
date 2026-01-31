use super::{Base, Pullback};
use algebra::Real;
use backend::Backend;

pub struct LeafAdjoint;

impl<F: Real, B: Backend, const R: usize> Pullback<F, B, R> for LeafAdjoint {
    type Gradients = Base<B::Storage<F>, F, R>;

    fn back(&self, _backend: &mut B, grad: Base<B::Storage<F>, F, R>) -> Self::Gradients {
        grad
    }
}

pub struct GradientTape<A> {
    adjoint: A,
}

impl<A> GradientTape<A> {
    pub fn new(adjoint: A) -> Self {
        Self { adjoint }
    }

    pub fn backward<F, B, const R: usize>(
        self,
        backend: &mut B,
        seed_grad: Base<B::Storage<F>, F, R>,
    ) -> A::Gradients
    where
        F: Real,
        B: Backend,
        A: Pullback<F, B, R>,
    {
        self.adjoint.back(backend, seed_grad)
    }
}

use super::{Base, Pullback};
use algebra::Real;
use backend::Backend;
use core::marker::PhantomData;

pub struct LeafAdjoint<F: Real, const R: usize> {
    _marker: PhantomData<F>,
}

impl<F: Real, const R: usize> LeafAdjoint<F, R> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<F: Real, B: Backend, const R: usize> Pullback<B, R> for LeafAdjoint<F, R> {
    type Primal = F;
    type Cotangent = F;
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

    pub fn backward<B, const R: usize>(
        self,
        backend: &mut B,
        seed_grad: Base<B::Storage<A::Cotangent>, A::Cotangent, R>,
    ) -> A::Gradients
    where
        B: Backend,
        A: Pullback<B, R>,
    {
        self.adjoint.back(backend, seed_grad)
    }
}

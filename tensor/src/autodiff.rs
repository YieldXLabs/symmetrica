use super::{Base, Dense, Differentiable, Evaluator, Pullback};
use algebra::Real;
use backend::Backend;

pub struct LeafAdjoint;

impl<F: Real, B: Backend<F>, const R: usize> Differentiable<F, B, R> for Dense<F, R> {
    type Adjoint = LeafAdjoint;

    fn forward(&self, backend: &mut B) -> (Base<B::Repr, F, R>, Self::Adjoint) {
        let res = self.eval(backend);
        (res, LeafAdjoint)
    }
}

impl<F: Real, B: Backend<F>, const R: usize> Pullback<F, B, R> for LeafAdjoint {
    type Gradients = Base<B::Repr, F, R>;

    fn back(&self, _backend: &mut B, grad: Base<B::Repr, F, R>) -> Self::Gradients {
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

use super::{Base, Differentiable, Pullback, Tensor};
use algebra::{Real, Shape};
use backend::Backend;

pub struct GradientTape<A> {
    adjoint: A,
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

impl<F: Real, Sh: Shape, const R: usize, E> Tensor<F, Sh, R, E> {
    pub fn forward<B: Backend<F>>(
        &self,
        backend: &mut B,
    ) -> (Base<B::Repr, F, R>, GradientTape<E::Adjoint>)
    where
        E: Differentiable<F, B, R>,
    {
        let (res, adjoint) = self.expr.forward(backend);

        (res, GradientTape { adjoint })
    }
}

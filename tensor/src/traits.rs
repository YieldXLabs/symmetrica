use super::Base;
use algebra::{Data, Real, Semiring};
use backend::Backend;

pub trait Evaluator<F: Data, B: Backend<F>, const R: usize> {
    fn eval(&self, backend: &mut B) -> Base<B::Repr, F, R>;
}

// Represents the 'backward' pass for a specific node
pub trait Pullback<F: Real, B: Backend<F>, const R: usize> {
    type Gradients;
    fn back(&self, backend: &mut B, grad: Base<B::Repr, F, R>) -> Self::Gradients;
}

pub trait Differentiable<F: Real, B: Backend<F>, const R: usize>: Evaluator<F, B, R> {
    type Adjoint: Pullback<F, B, R>;
    fn forward(&self, backend: &mut B) -> (Base<B::Repr, F, R>, Self::Adjoint);
}

// Logic/Graphs
pub trait Traceback<F: Semiring, B: Backend<F>, const R: usize> {
    type Signal;
    fn trace(&self, backend: &mut B, signal_out: Self::Signal) -> Self::Signal;
}

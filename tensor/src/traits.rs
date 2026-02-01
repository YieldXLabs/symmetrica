use super::Base;
use algebra::{Data, Real, Semiring};
use backend::Backend;

pub trait Evaluator<B: Backend, const RANK: usize> {
    type Data: Data;

    fn eval(&self, backend: &mut B) -> Base<B::Storage<Self::Data>, Self::Data, RANK>;
}

pub trait Pullback<B: Backend, const R: usize> {
    type Primal: Data;
    type Cotangent: Data;
    type Gradients;

    fn back(
        &self,
        backend: &mut B,
        grad: Base<B::Storage<Self::Cotangent>, Self::Cotangent, R>,
    ) -> Self::Gradients;
}

pub trait Differentiable<B: Backend, const R: usize>
where
    Self: Evaluator<B, R>,
    Self::Data: Real,
{
    type Adjoint: Pullback<B, R, Primal = Self::Data, Cotangent = Self::Data>;

    fn forward(
        &self,
        backend: &mut B,
    ) -> (Base<B::Storage<Self::Data>, Self::Data, R>, Self::Adjoint);
}

pub trait Traceback<F: Semiring, B: Backend, const R: usize> {
    type Signal;

    fn trace(&self, backend: &mut B, signal_out: Self::Signal) -> Self::Signal;
}

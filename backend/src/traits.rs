use algebra::Real;
use std::fmt::Debug;

pub trait Storage<F>: Debug + Clone + Send + Sync {
    fn len(&self) -> usize;
    fn alloc(n: usize) -> Self;
    fn as_slice(&self) -> &[F];
    fn as_mut_slice(&mut self) -> &mut [F];
}

pub trait UnaryKernel<S> {
    fn apply(inp: &S, out: &mut S);
}

pub trait BinaryKernel<S> {
    fn apply(lhs: &S, rhs: &S, out: &mut S);
}

pub trait Backend<F: Real> {
    type Repr: Storage<F>;

    fn pure(&mut self, data: &[F]) -> Self::Repr;

    fn compute<E>(&mut self, expr: &E) -> Self::Repr
    where
        E: Evaluator<F, Self>,
        Self: Sized,
    {
        expr.eval(self)
    }
}

pub trait Evaluator<F: Real, B: Backend<F> + ?Sized> {
    fn eval(&self, backend: &mut B) -> B::Repr;
}

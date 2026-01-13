use std::fmt::Debug;
use algebra::Real;

pub trait Evaluator<F: Real, B: Backend<F>> {
    type Output;

    fn collect() -> B::Repr;
}

pub trait Backend<F: Real> {
    type Repr;

    fn compute<Op>(op: Op) -> Self::Repr
    
    where
        Op: Evaluator<F, Self>, Self: Sized;
}

pub trait Storage<F>: Debug + Clone + Send + Sync {
    fn len(&self) -> usize;
    fn alloc(n: usize) -> Self;
    fn as_slice(&self) -> &[F];
    fn as_mut_slice(&mut self) -> &mut [F];
}

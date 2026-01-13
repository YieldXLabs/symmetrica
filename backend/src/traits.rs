use algebra::{Op, Real};
use std::fmt::Debug;

pub trait Backend<F: Real> {
    type Repr;

    fn eval(&mut self, op: Op<F, Self::Repr>) -> Self::Repr;
}

pub trait Storage<F>: Debug + Clone + Send + Sync {
    fn len(&self) -> usize;
    fn alloc(n: usize) -> Self;
    fn as_slice(&self) -> &[F];
    fn as_mut_slice(&mut self) -> &mut [F];
}

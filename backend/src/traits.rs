use algebra::Real;
use std::fmt::Debug;

pub trait Storage<F>: Debug + Clone + Send + Sync {
    fn len(&self) -> usize;
    fn alloc(n: usize) -> Self;
    fn as_slice(&self) -> &[F];
    fn as_mut_slice(&mut self) -> &mut [F];
}

impl<F: Real> Storage<F> for Vec<F>
where
    F: Send + Sync,
{
    fn len(&self) -> usize {
        self.len()
    }
    fn alloc(n: usize) -> Self {
        vec![F::zero(); n]
    }
    fn as_slice(&self) -> &[F] {
        self
    }
    fn as_mut_slice(&mut self) -> &mut [F] {
        self
    }
}

pub trait UnaryKernel<F> {
    fn apply(x: F) -> F;
}

pub trait BinaryKernel<F> {
    fn apply(x: F, y: F) -> F;
}

pub trait ReduceKernel<F> {
    fn init() -> F;
    fn step(acc: F, x: F) -> F;
}

pub trait StreamKernel<F> {
    type State;

    fn init(&self) -> Self::State;

    fn step(&self, state: &mut Self::State, input: F) -> F;
}

pub trait Backend<F: Real> {
    type Repr: Storage<F>;

    fn pure(&mut self, data: &[F]) -> Self::Repr;

    fn unary<K: UnaryKernel<F>>(&mut self, input: &Self::Repr) -> Self::Repr;

    fn binary<K: BinaryKernel<F>>(&mut self, lhs: &Self::Repr, rhs: &Self::Repr) -> Self::Repr;

    fn reduce<K: ReduceKernel<F>>(&mut self, input: &Self::Repr) -> F;

    fn stream<K: StreamKernel<F>>(&mut self, input: &Self::Repr, kernel: K) -> Self::Repr;

    fn compute<E>(&mut self, expr: &E) -> (Self::Repr, Vec<usize>)
    where
        E: Evaluator<F, Self>,
        Self: Sized,
    {
        expr.eval(self)
    }
}

pub trait Evaluator<F: Real, B: Backend<F> + ?Sized> {
    fn eval(&self, backend: &mut B) -> (B::Repr, Vec<usize>);
}

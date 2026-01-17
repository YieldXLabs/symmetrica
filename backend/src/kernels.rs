use super::{BinaryKernel, ReduceKernel, UnaryKernel};
use algebra::Real;

pub struct SubKernel;
impl<F: Real> BinaryKernel<F> for SubKernel {
    fn apply(x: F, y: F) -> F {
        x - y
    }
}

pub struct AddKernel;
impl<F: Real> BinaryKernel<F> for AddKernel {
    fn apply(x: F, y: F) -> F {
        x + y
    }
}

pub struct MulKernel;
impl<F: Real> BinaryKernel<F> for MulKernel {
    fn apply(x: F, y: F) -> F {
        x * y
    }
}

pub struct AbsKernel;
impl<F: Real> UnaryKernel<F> for AbsKernel {
    fn apply(x: F) -> F {
        x.abs()
    }
}

pub struct SumKernel;
impl<F: Real> ReduceKernel<F> for SumKernel {
    fn init() -> F {
        F::zero()
    }
    fn step(acc: F, x: F) -> F {
        acc + x
    }
}

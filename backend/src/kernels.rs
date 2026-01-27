use super::{BinaryKernel, ReduceKernel, StreamKernel, UnaryKernel};
use algebra::{Real, Ring, Semiring};

pub struct AddKernel;
impl<F: Semiring> BinaryKernel<F> for AddKernel {
    fn apply(x: F, y: F) -> F {
        x + y
    }
}

pub struct MulKernel;
impl<F: Semiring> BinaryKernel<F> for MulKernel {
    fn apply(x: F, y: F) -> F {
        x * y
    }
}

pub struct SumKernel;
impl<F: Semiring> ReduceKernel<F> for SumKernel {
    fn init() -> F {
        F::zero()
    }
    fn step(acc: F, x: F) -> F {
        acc + x
    }
}

pub struct SubKernel;
impl<F: Ring> BinaryKernel<F> for SubKernel {
    fn apply(x: F, y: F) -> F {
        x - y
    }
}

pub struct Ema<F> {
    alpha: F,
}

impl<F: Ring> StreamKernel<F> for Ema<F> {
    type State = F;

    fn init(&self) -> F {
        F::zero()
    }

    fn step(&self, state: &mut F, x: F) -> F {
        *state = self.alpha * x + (F::one() - self.alpha) * (*state);
        *state
    }
}

pub struct AbsKernel;
impl<F: Real> UnaryKernel<F> for AbsKernel {
    fn apply(x: F) -> F {
        x.abs()
    }
}

// struct TwapKernel<F> {
//     horizon: usize,
// }

// impl<F: Real> StreamKernel<F> for TwapKernel<F> {
//     type State = usize;

//     fn init(&self) -> usize { 0 }

//     fn step(&self, state: &mut usize, delta: F) -> F {
//         *state += 1;
//         delta / F::try_from(self.horizon).unwrap()
//     }
// }

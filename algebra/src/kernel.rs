use super::{BinaryKernel, Real, ReduceKernel, Ring, Semiring, StreamKernel, UnaryKernel};

#[derive(Debug, Clone, Copy, Default)]
pub struct AddKernel;
impl<F: Semiring> BinaryKernel<F> for AddKernel {
    fn apply(&self, x: F, y: F) -> F {
        x + y
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ScaleKernel<F> {
    pub factor: F,
}
impl<F: Semiring> UnaryKernel<F> for ScaleKernel<F> {
    fn apply(&self, x: F) -> F {
        x * self.factor
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MulKernel;
impl<F: Semiring> BinaryKernel<F> for MulKernel {
    fn apply(&self, x: F, y: F) -> F {
        x * y
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SumKernel;
impl<F: Semiring> ReduceKernel<F> for SumKernel {
    fn init() -> F {
        F::zero()
    }
    fn step(acc: F, x: F) -> F {
        acc + x
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ProductKernel;
impl<F: Semiring> ReduceKernel<F> for ProductKernel {
    fn init() -> F {
        F::one()
    }
    fn step(acc: F, x: F) -> F {
        acc * x
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SubKernel;
impl<F: Ring> BinaryKernel<F> for SubKernel {
    fn apply(&self, x: F, y: F) -> F {
        x - y
    }
}

#[derive(Debug, Clone, Copy)]
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

#[derive(Debug, Clone, Copy, Default)]
pub struct AbsKernel;
impl<F: Real> UnaryKernel<F> for AbsKernel {
    fn apply(&self, x: F) -> F {
        x.abs()
    }
}

// TODO: Fusion
// pub trait FusedKernel<F>: Copy + Clone {
//     fn execute(&self, x: F, y: Option<F>) -> F;
// }

// // A chain of two kernels: K2(K1(x))
// #[derive(Clone, Copy)]
// pub struct ChainedKernel<K1, K2> {
//     k1: K1,
//     k2: K2,
// }

// impl<F, K1: UnaryKernel<F>, K2: UnaryKernel<F>> FusedKernel<F> for ChainedKernel<K1, K2> {
//     fn execute(&self, x: F, _: Option<F>) -> F {
//         self.k2.apply(self.k1.apply(x))
//     }
// }

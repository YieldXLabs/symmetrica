use super::{BinaryKernel, Promote, Real, ReduceKernel, Ring, Semiring, StreamKernel, UnaryKernel};

#[derive(Debug, Clone, Copy)]
pub struct AddKernel;
impl<L, R> BinaryKernel<L, R> for AddKernel
where
    L: Promote<R>,
    L::Output: Semiring,
{
    type Output = L::Output;

    #[inline(always)]
    fn apply(&self, lhs: L, rhs: R) -> Self::Output {
        let l_prom = lhs.promote_left();
        let r_prom = L::promote_right(rhs);

        l_prom + r_prom
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ScaleKernel<F> {
    pub factor: F,
}
impl<F: Semiring> UnaryKernel<F> for ScaleKernel<F> {
    type Output = F;

    fn apply(&self, x: F) -> F {
        x * self.factor
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MulKernel;
impl<L, R> BinaryKernel<L, R> for MulKernel
where
    L: Promote<R>,
    L::Output: Semiring,
{
    type Output = L::Output;

    #[inline(always)]
    fn apply(&self, lhs: L, rhs: R) -> Self::Output {
        let l_prom = lhs.promote_left();
        let r_prom = L::promote_right(rhs);
        l_prom * r_prom
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SumKernel;
impl<F: Semiring> ReduceKernel<F> for SumKernel {
    type Acc = F;
    type Output = F;

    fn init(&self) -> F {
        F::zero()
    }

    fn step(&self, acc: F, x: F) -> F {
        acc + x
    }

    fn finish(&self, acc: F) -> F {
        acc
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ProductKernel;
impl<F: Semiring> ReduceKernel<F> for ProductKernel {
    type Acc = F;
    type Output = F;

    fn init(&self) -> F {
        F::one()
    }

    fn step(&self, acc: F, x: F) -> F {
        acc * x
    }

    fn finish(&self, acc: F) -> F {
        acc
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SubKernel;
impl<L, R> BinaryKernel<L, R> for SubKernel
where
    L: Promote<R>,
    L::Output: Ring,
{
    type Output = L::Output;

    #[inline(always)]
    fn apply(&self, lhs: L, rhs: R) -> Self::Output {
        let l_prom = lhs.promote_left();
        let r_prom = L::promote_right(rhs);
        l_prom - r_prom
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Ema<F> {
    alpha: F,
}

impl<F: Ring> StreamKernel<F> for Ema<F> {
    type State = F;
    type Output = F;

    fn init(&self) -> F {
        F::zero()
    }

    fn step(&self, state: &mut F, x: F) -> F {
        *state = self.alpha * x + (F::one() - self.alpha) * (*state);
        *state
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AbsKernel;
impl<F: Real> UnaryKernel<F> for AbsKernel {
    type Output = F;

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

use super::{
    BinaryKernel, Data, One, OrderedField, Promote, Real, ReduceKernel, Ring, Semiring,
    StreamKernel, UnaryKernel, Zero,
};

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
pub struct ScaleKernel<S: Data> {
    pub factor: S,
}
impl<In, S> UnaryKernel<In> for ScaleKernel<S>
where
    In: Promote<S>,
    In::Output: Semiring,
    S: Data,
{
    type Output = In::Output;

    #[inline(always)]
    fn apply(&self, x: In) -> Self::Output {
        let x_prom = x.promote_left();
        let s_prom = In::promote_right(self.factor);
        x_prom * s_prom
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

// TODO: DivKernel.
// Missing implementation for Division. Requires `L::Output: Field` (or Real).

#[derive(Debug, Clone, Copy)]
pub struct SumKernel;
impl<In> ReduceKernel<In> for SumKernel
where
    In: Promote<In>,
    In::Output: Semiring,
{
    type Acc = In::Output;
    type Output = In::Output;

    #[inline(always)]
    fn init(&self) -> Self::Acc {
        Self::Acc::zero()
    }

    // TODO: Numerical Stability.
    // Naive accumulation results in significant precision loss.
    // Implement Kahan Summation or Pairwise Summation inside `step` or `Acc`.
    #[inline(always)]
    fn step(&self, acc: Self::Acc, x: In) -> Self::Acc {
        acc + x.promote_left()
    }

    fn merge(&self, _acc1: Self::Acc, _acc2: Self::Acc) -> Self::Acc {
        unimplemented!()
    }

    #[inline(always)]
    fn finish(&self, acc: Self::Acc) -> Self::Output {
        acc
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ProductKernel;
impl<In> ReduceKernel<In> for ProductKernel
where
    In: Promote<In>,
    In::Output: Semiring,
{
    type Acc = In::Output;
    type Output = In::Output;

    #[inline(always)]
    fn init(&self) -> Self::Acc {
        Self::Acc::one()
    }

    #[inline(always)]
    fn step(&self, acc: Self::Acc, x: In) -> Self::Acc {
        acc * x.promote_left()
    }

    fn merge(&self, _acc1: Self::Acc, _acc2: Self::Acc) -> Self::Acc {
        unimplemented!()
    }

    #[inline(always)]
    fn finish(&self, acc: Self::Acc) -> Self::Output {
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

impl<In, F> StreamKernel<In> for Ema<F>
where
    F: Ring,
    In: Promote<F, Output = F>,
{
    type State = F;
    type Output = F;

    fn init(&self) -> F {
        // TODO: Initialization Bias (Warm-up).
        // Initializing with Zero causes the EMA to "drag" up from 0 at the start of the stream.
        // Standard practice is to initialize with the first value of the stream.
        F::zero()
    }

    fn step(&self, state: &mut F, x: In) -> F {
        let x_prom = x.promote_left();

        // TODO: Operation Count Optimization.
        // Current: 2 muls, 1 add, 1 sub.
        // Optimization: `state + alpha * (x - state)` (1 mul, 1 add, 1 sub).
        *state = (self.alpha * x_prom) + ((F::one() - self.alpha) * *state);
        *state
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AbsKernel;
impl<In> UnaryKernel<In> for AbsKernel
where
    In: Promote<In>,
    In::Output: Real,
{
    type Output = In::Output;

    #[inline(always)]
    fn apply(&self, x: In) -> Self::Output {
        let x_prom = x.promote_left();
        x_prom.abs()
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

// TODO: vjp/jvp

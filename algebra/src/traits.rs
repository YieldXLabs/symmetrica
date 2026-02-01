use core::cmp::Ord;
use core::fmt::Debug;
use core::iter::{Product, Sum};
use core::ops::{Add, Div, Mul, Neg, Sub};

// 0. Identity
// The additive identity (0)
pub trait Zero: Sized {
    fn zero() -> Self;
}
// The multiplicative identity (1)
pub trait One: Sized {
    fn one() -> Self;
}

// Base trait for any data stored in a Tensor
pub trait Data: Clone + Copy + PartialEq + Debug + 'static {}
impl<T: Clone + Copy + PartialEq + Debug + 'static> Data for T {}

// 1. Semiring (Bool, Base Math, Tropical)
pub trait Semiring:
    Data + Zero + One + Add<Output = Self> + Mul<Output = Self> + Sum + Product
{
}

// 2. Ring (Integers, Physics)
pub trait Ring: Semiring + Sub<Output = Self> + Neg<Output = Self> {}

// 3. Field (Linear Algebra Inverses)
pub trait Field: Ring + Div<Output = Self> {
    fn recip(self) -> Self;
}

// Extensions: Ordering and Real Analysis
// Todo: Ord for floats is tricky due to NaN; ensure proper handling
pub trait OrderedField: Field + PartialOrd + Ord {
    fn abs(self) -> Self;
    fn signum(self) -> Self;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn clamp(self, lo: Self, hi: Self) -> Self;
}

pub trait Real: OrderedField {
    fn pi() -> Self;
    fn e() -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn pow(self, exp: Self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
}

pub trait Discretization: Sized {
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;
}

// kernels
pub trait KernelBase: Copy + Clone + Debug + 'static {}
impl<T: Copy + Clone + Debug + 'static> KernelBase for T {}

pub trait UnaryKernel<In>: KernelBase {
    type Output: Data;

    fn apply(&self, x: In) -> Self::Output;
}

pub trait BinaryKernel<L, R>: KernelBase {
    type Output: Data;

    fn apply(&self, lhs: L, rhs: R) -> Self::Output;
}

pub trait ReduceKernel<In>: KernelBase {
    type Output: Data;
    type Acc: Data;

    fn init() -> Self::Acc;
    fn step(acc: Self::Acc, x: In) -> Self::Acc;
    fn finish(acc: Self::Acc) -> Self::Output;
}

pub trait StreamKernel<In>: KernelBase {
    type State: Clone;
    type Output: Data;

    fn init(&self) -> Self::State;
    fn step(&self, state: &mut Self::State, input: In) -> Self::Output;
}

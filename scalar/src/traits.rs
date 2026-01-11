use std::cmp::Ord;
use std::fmt::{Debug, Display};
use std::iter::{Product, Sum};
use std::ops::{Add, Div, Mul, Neg, Sub};

// Algebraic field
pub trait Field:
    Copy
    + Clone
    + PartialEq
    + Debug
    + Display
    + Default
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Sum
    + Product
{
    fn zero() -> Self;
    fn one() -> Self;
    fn recip(self) -> Self;
}

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

use super::{Discretization, Field, One, OrderedField, Promote, Real, Ring, Semiring, Zero};
use core::cmp::Ordering;
use core::fmt::{self, Display};
use core::hash::{Hash, Hasher};
use core::iter::{Product, Sum};
use core::ops::{Add, Div, Mul, Neg, Sub};

// TODO: ZeroCopy
// derive `bytemuck::Pod` and `bytemuck::ZeroCopy` to allow casting bytes directly to &[TradingFloat].

/// TradingFloat is a total-order, hashable floating-point scalar.
/// Invariants:
/// - NaN and ±Inf are forbidden
/// - Equality follows IEEE-754 (`0.0 == -0.0`)
/// - Hashing is bitwise with canonicalized zero
/// - No epsilon comparisons are used
/// - Default is 0.0
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct TradingFloat(f64);

impl TradingFloat {
    pub const ZERO: Self = Self(0.0);
    pub const ONE: Self = Self(1.0);
    pub const PI: Self = Self(std::f64::consts::PI);
    pub const E: Self = Self(std::f64::consts::E);
    pub const PERCENT: Self = Self(0.01);
    pub const ONE_HUNDRED: Self = Self(100.0);

    /// Helper to create a TradingFloat. Panics on NaN/Inf.
    pub fn new(val: f64) -> Self {
        Self::try_from(val).expect("Invalid TradingFloat value")
    }

    /// # Normalization
    /// This method **normalizes** `-0.0` to `+0.0`.
    /// This ensures that `Hash` and `Eq` remain consistent.
    pub fn to_bits(self) -> u64 {
        if self.0 == 0.0 {
            0.0f64.to_bits()
        } else {
            self.0.to_bits()
        }
    }

    pub fn to_f64(self) -> f64 {
        self.0
    }

    pub fn to_f32(self) -> f32 {
        self.0 as f32
    }
}

impl Default for TradingFloat {
    fn default() -> Self {
        TradingFloat::ZERO
    }
}

impl Display for TradingFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl PartialEq for TradingFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for TradingFloat {}

impl Hash for TradingFloat {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.to_bits().hash(state);
    }
}

impl PartialOrd for TradingFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TradingFloat {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).expect("NaN in TradingFloat")
    }
}

impl From<TradingFloat> for f64 {
    fn from(value: TradingFloat) -> f64 {
        value.0
    }
}

impl TryFrom<f64> for TradingFloat {
    type Error = &'static str;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        if value.is_nan() {
            Err("NaN not allowed in TradingFloat")
        } else if value.is_infinite() {
            Err("Infinite values not allowed in TradingFloat")
        } else {
            Ok(TradingFloat(value))
        }
    }
}

impl Add for TradingFloat {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let result = self.0 + rhs.0;
        debug_assert!(
            result.is_finite(),
            "TradingFloat addition overflow: {} + {}",
            self.0,
            rhs.0
        );
        TradingFloat(result)
    }
}

impl Sub for TradingFloat {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let result = self.0 - rhs.0;
        debug_assert!(
            result.is_finite(),
            "TradingFloat subtraction overflow: {} - {}",
            self.0,
            rhs.0
        );
        TradingFloat(result)
    }
}

impl Mul for TradingFloat {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let result = self.0 * rhs.0;
        debug_assert!(
            result.is_finite(),
            "TradingFloat multiplication overflow: {} * {}",
            self.0,
            rhs.0
        );
        TradingFloat(result)
    }
}

impl Div for TradingFloat {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        debug_assert!(
            rhs.0 != 0.0,
            "TradingFloat invariant violated: division by zero"
        );
        let result = self.0 / rhs.0;
        debug_assert!(
            result.is_finite(),
            "TradingFloat division overflow: {} / {}",
            self.0,
            rhs.0
        );
        TradingFloat(result)
    }
}

impl Neg for TradingFloat {
    type Output = Self;
    fn neg(self) -> Self {
        TradingFloat(-self.0)
    }
}

impl Sum for TradingFloat {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(TradingFloat::ZERO, |acc, x| acc + x)
    }
}

impl Product for TradingFloat {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(TradingFloat::ONE, |acc, x| acc * x)
    }
}

impl Zero for TradingFloat {
    #[inline]
    fn zero() -> Self {
        Self::ZERO
    }
}
impl One for TradingFloat {
    #[inline]
    fn one() -> Self {
        Self::ONE
    }
}

// Allows MatMul and Scan
impl Semiring for TradingFloat {}
// Allows Subtraction and Physics
impl Ring for TradingFloat {}
// Allows division and linear algebra
impl Field for TradingFloat {
    #[inline]
    fn recip(self) -> Self {
        debug_assert!(
            self.0 != 0.0,
            "TradingFloat division overflow: 1 / {}",
            self.0
        );
        TradingFloat(1.0 / self.0)
    }
}

impl OrderedField for TradingFloat {
    #[inline]
    fn abs(self) -> Self {
        TradingFloat(self.0.abs())
    }
    #[inline]
    fn signum(self) -> Self {
        if self.0 > 0.0 {
            TradingFloat::ONE
        } else if self.0 < 0.0 {
            -TradingFloat::ONE
        } else {
            TradingFloat::ZERO
        }
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        TradingFloat(self.0.min(other.0))
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        TradingFloat(self.0.max(other.0))
    }
    #[inline]
    fn clamp(self, lo: Self, hi: Self) -> Self {
        debug_assert!(
            lo <= hi,
            "TradingFloat clamp invariant violated: {} > {}",
            lo.0,
            hi.0
        );
        TradingFloat(self.0.clamp(lo.0, hi.0))
    }
}

// Allows Analysis and Calculus
impl Real for TradingFloat {
    #[inline]
    fn pi() -> Self {
        TradingFloat::PI
    }
    #[inline]
    fn e() -> Self {
        TradingFloat::E
    }
    #[inline]
    fn exp(self) -> Self {
        TradingFloat(self.0.exp())
    }

    #[inline]
    fn ln(self) -> Self {
        debug_assert!(self.0 > 0.0);
        TradingFloat(self.0.ln())
    }

    #[inline]
    fn sqrt(self) -> Self {
        debug_assert!(self.0 >= 0.0);
        TradingFloat(self.0.sqrt())
    }

    #[inline]
    fn pow(self, exp: Self) -> Self {
        TradingFloat(self.0.powf(exp.0))
    }
    #[inline]
    fn sin(self) -> Self {
        TradingFloat(self.0.sin())
    }
    #[inline]
    fn cos(self) -> Self {
        TradingFloat(self.0.cos())
    }
}

impl Discretization for TradingFloat {
    #[inline]
    fn floor(self) -> Self {
        TradingFloat(self.0.floor())
    }
    #[inline]
    fn ceil(self) -> Self {
        TradingFloat(self.0.ceil())
    }
    #[inline]
    fn round(self) -> Self {
        TradingFloat(self.0.round())
    }
}

impl Promote<TradingFloat> for TradingFloat {
    type Output = TradingFloat;

    fn promote_left(self) -> Self::Output {
        self
    }

    fn promote_right(rhs: TradingFloat) -> Self::Output {
        rhs
    }
}

impl Promote<bool> for TradingFloat {
    type Output = TradingFloat;

    fn promote_left(self) -> Self::Output {
        self
    }

    fn promote_right(rhs: bool) -> Self::Output {
        if rhs {
            TradingFloat::ONE
        } else {
            TradingFloat::ZERO
        }
    }
}

impl Promote<TradingFloat> for bool {
    type Output = TradingFloat;

    fn promote_left(self) -> Self::Output {
        if self {
            TradingFloat::ONE
        } else {
            TradingFloat::ZERO
        }
    }

    fn promote_right(rhs: TradingFloat) -> Self::Output {
        rhs
    }
}

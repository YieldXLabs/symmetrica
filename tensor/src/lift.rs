use algebra::Real;
pub trait Lift<F: Real> {
    type Output;

    fn lift(self) -> Self::Output;
}

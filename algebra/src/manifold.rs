use super::traits::Field;

pub trait Manifold<F: Field> {
    type Point;
    type Tangent;

    fn exp_map(&self, base: &Self::Point, vec: &Self::Tangent) -> Self::Point;
    fn log_map(&self, base: &Self::Point, target: &Self::Point) -> Self::Tangent;
}

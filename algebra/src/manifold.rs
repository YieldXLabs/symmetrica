use super::traits::Real;

pub trait Manifold<F: Real> {
    type Point;
    type Tangent;

    // Map a tangent vector 'v' at point 'p' to a new point on the manifold
    fn exp_map(&self, base: &Self::Point, vec: &Self::Tangent) -> Self::Point;
    // Find the tangent vector 'v' that connects 'p' to 'q'
    fn log_map(&self, base: &Self::Point, target: &Self::Point) -> Self::Tangent;
    // Shortest path distance on the manifold
    fn dist(&self, p: &Self::Point, q: &Self::Point) -> F;
    // Projects a raw point back onto the manifold
    fn project(&self, p: &Self::Point) -> Self::Point;
    // Riemannian Inner Product
    fn inner(&self, p: &Self::Point, u: &Self::Tangent, v: &Self::Tangent) -> F;
}

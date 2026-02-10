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

// TODO: Dynamic stop
// struct StopState<F> {
//     distance: F,   // > 0
//     rate: F,       // (0, r_max)
// }

// struct StopTangent<F> {
//     d_dist: F,
//     d_rate: F,
// }

// impl<F: Real> Manifold<F> for StopManifold {
//     type Point = StopState<F>;
//     type Tangent = StopTangent<F>;

//     fn exp_map(&self, p: &Self::Point, v: &Self::Tangent) -> Self::Point {
//         StopState {
//             distance: p.distance * (v.d_dist).exp(),
//             rate: clamp(p.rate + v.d_rate, F::zero(), self.r_max),
//         }
//     }

//     fn project(&self, p: &Self::Point) -> Self::Point {
//         StopState {
//             distance: p.distance.max(self.min_dist),
//             rate: clamp(p.rate, F::zero(), self.r_max),
//         }
//     }

//     fn inner(&self, _p: &Self::Point, u: &Self::Tangent, v: &Self::Tangent) -> F {
//         u.d_dist * v.d_dist + u.d_rate * v.d_rate
//     }

//     fn dist(&self, p: &Self::Point, q: &Self::Point) -> F {
//         ((q.distance / p.distance).ln().powi(2)
//          + (q.rate - p.rate).powi(2)).sqrt()
//     }
// }

// struct PoincareBall<F> {
//     c: F, // curvature > 0
// }

// impl<F: Real> Manifold<F> for PoincareBall<F> {
//     type Point = Vec<F>;
//     type Tangent = Vec<F>;

//     fn exp_map(&self, p: &Self::Point, v: &Self::Tangent) -> Self::Point {
//         // simplified: Möbius addition
//         mobius_add(p, v, self.c)
//     }

//     fn log_map(&self, p: &Self::Point, q: &Self::Point) -> Self::Tangent {
//         mobius_add(q, &neg(p), self.c)
//     }

//     fn dist(&self, p: &Self::Point, q: &Self::Point) -> F {
//         poincare_distance(p, q, self.c)
//     }

//     fn project(&self, p: &Self::Point) -> Self::Point {
//         let norm = l2_norm(p);
//         let max = F::one() - F::epsilon();
//         if norm >= max {
//             p.iter().map(|x| *x * max / norm).collect()
//         } else {
//             p.clone()
//         }
//     }

//     fn inner(&self, p: &Self::Point, u: &Self::Tangent, v: &Self::Tangent) -> F {
//         let lambda = F::two() / (F::one() - l2_norm_sq(p));
//         lambda * lambda * dot(u, v)
//     }
// }

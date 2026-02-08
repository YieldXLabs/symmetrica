use super::Host;
use std::sync::Arc;

use algebra::{ConstExpr, Data};

pub trait Lift<F: Data> {
    type Output;

    fn lift(self) -> Self::Output;
}

impl<F: Data> Lift<F> for F {
    type Output = ConstExpr<F>;

    fn lift(self) -> Self::Output {
        ConstExpr(self)
    }
}

impl<F: Data> Lift<F> for Vec<F> {
    type Output = Host<F, 1>;

    fn lift(self) -> Self::Output {
        let len = self.len();
        let storage = Arc::new(self);
        Host::new(storage, [len])
    }
}

impl<F: Data> Lift<F> for &[F] {
    type Output = Host<F, 1>;

    fn lift(self) -> Self::Output {
        let len = self.len();
        let storage = Arc::new(self.to_vec());
        Host::new(storage, [len])
    }
}

impl<F: Data> Lift<F> for Arc<Vec<F>> {
    type Output = Host<F, 1>;

    fn lift(self) -> Self::Output {
        let len = self.len();
        Host::new(self, [len])
    }
}

impl<F: Data, const N: usize> Lift<F> for [F; N] {
    type Output = Host<F, 1>;

    fn lift(self) -> Self::Output {
        let storage = Arc::new(self.to_vec());
        Host::new(storage, [N])
    }
}

impl<F: Data, const R: usize, const C: usize> Lift<F> for [[F; C]; R] {
    type Output = Host<F, 2>;

    fn lift(self) -> Self::Output {
        let mut flat_data = Vec::with_capacity(R * C);
        for row in self {
            flat_data.extend_from_slice(&row);
        }
        Host::new(Arc::new(flat_data), [R, C])
    }
}

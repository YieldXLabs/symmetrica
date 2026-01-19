use super::{Base, Dense, Differentiable, Evaluator, Pullback};
use algebra::{AddExpr, Real};
use backend::{AddKernel, Backend};

pub struct LeafAdjoint;

impl<F: Real, B: Backend<F>, const R: usize> Pullback<F, B, R> for LeafAdjoint {
    type Gradients = ();

    fn back(&self, _b: &mut B, _g: Base<B::Repr, F, R>) -> () {
        ()
    }
}

impl<F: Real, B: Backend<F>, const R: usize> Differentiable<F, B, R> for Dense<F, R> {
    type Adjoint = LeafAdjoint;

    fn forward(&self, backend: &mut B) -> (Base<B::Repr, F, R>, Self::Adjoint) {
        let res = self.eval(backend);
        (res, LeafAdjoint)
    }
}

impl<F: Real, B: Backend<F>, const RANK: usize> Evaluator<F, B, RANK> for Dense<F, RANK> {
    fn eval(&self, backend: &mut B) -> Base<B::Repr, F, RANK> {
        let storage = backend.pure(&self.storage);

        Base::from_parts(storage, self.shape, self.strides, self.offset)
    }
}

impl<F, B, L, Rhs, const R: usize> Evaluator<F, B, R> for AddExpr<L, Rhs>
where
    F: Real,
    B: Backend<F>,
    L: Evaluator<F, B, R>,
    Rhs: Evaluator<F, B, R>,
{
    fn eval(&self, backend: &mut B) -> Base<B::Repr, F, R> {
        let l = self.left.eval(backend);
        let r = self.right.eval(backend);

        let storage = backend.binary::<AddKernel>(&l.storage, &r.storage);

        Base::new(storage, l.shape)
    }
}

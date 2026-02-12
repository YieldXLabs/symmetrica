use super::{Differentiable, Evaluator, GradientTape, LeafAdjoint, Lift, Lower, PackDense};
use algebra::{
    BroadcastExpr, BroadcastMap, ConstExpr, Data, DynRank, MapExpr, Permutation, Real, ReshapeExpr,
    ScaleKernel, Semiring, Shape, TransposeExpr,
};
use backend::Backend;
use std::{marker::PhantomData, sync::Arc};

// TODO: Strided View Validation.
// The current `Base` struct assumes a simple strided layout.
// Modern tensor libraries often support "Negative Strides" (for flipping axes)
// and "Zero Strides" (for broadcasting without copying).
// Ensure `compute_strides` and `is_dense` handle these cases
// TODO: Layout
// enum Layout {
//     C,
//     F,
//     Strided,
// }
#[derive(Debug, Clone)]
pub struct Base<S, F, const R: usize> {
    pub storage: S,
    pub shape: [usize; R],
    pub strides: [usize; R],
    pub offset: usize,
    _marker: PhantomData<F>,
}

impl<S, F, const R: usize> Base<S, F, R> {
    pub fn new(storage: S, shape: [usize; R]) -> Self {
        Self {
            storage,
            shape,
            strides: Self::compute_strides(&shape),
            offset: 0,
            _marker: PhantomData,
        }
    }

    pub fn from_parts(storage: S, shape: [usize; R], strides: [usize; R], offset: usize) -> Self {
        debug_assert!(shape.iter().all(|&d| d > 0));
        Self {
            storage,
            shape,
            strides,
            offset,
            _marker: PhantomData,
        }
    }

    pub fn is_dense(&self) -> bool {
        if self.offset != 0 {
            return false;
        }

        let mut stride = 1;

        for i in (0..R).rev() {
            if self.shape[i] > 1 {
                if self.strides[i] != stride {
                    return false;
                }
                stride *= self.shape[i];
            }
        }

        true
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn compute_strides(shape: &[usize; R]) -> [usize; R] {
        let mut strides = [0; R];
        let mut product = 1;

        for i in (0..R).rev() {
            strides[i] = product;
            product *= shape[i];
        }

        strides
    }
}

pub type Host<F, const R: usize> = Base<Arc<Vec<F>>, F, R>;

impl<F, B, const R: usize> Evaluator<B, R> for Host<F, R>
where
    F: Data,
    B: Backend,
{
    type Data = F;

    fn eval(&self, backend: &mut B) -> Base<B::Storage<F>, F, R> {
        let storage = backend.pure(&self.storage);
        Base::from_parts(storage, self.shape, self.strides, self.offset)
    }
}

impl<F, B, const R: usize> Differentiable<B, R> for Host<F, R>
where
    F: Real,
    B: Backend,
{
    type Adjoint = LeafAdjoint<F, R>;

    fn forward(&self, backend: &mut B) -> (Base<B::Storage<F>, F, R>, Self::Adjoint) {
        let res = self.eval(backend);
        (res, LeafAdjoint::new())
    }
}

// TODO: Factory Methods.
// - `eye` (Identity matrix): Critical for Linear Algebra.
// - `random`: Needs a seeded RNG backend trait to be deterministic.
// - `one_hot`: Needed for Classification/ML.
// - `toeplitz`: Useful for signal processing/convolution matrices.
// - `linspace`, `arange`: Standard numpy-like constructors.
// TODO: Implement ::stack, ::concat, ::split, ::tile, ::repeat, ::flip, ::roll, etc.
// TODO: Slicing (`slice`).
// Implementing slicing requires a new Expression type `SliceExpr`.
// It modifies `offset` and `shape` but keeps the underlying storage (zero-copy).
// Signature: `pub fn slice(self, ranges: &[Range<usize>]) -> Tensor<...>`
// TODO: Like family
// zeros_like(tensor)      // Copy shape, fill with 0
// ones_like(tensor)       // Copy shape, fill with 1
// rand_like(tensor)       // Copy shape, fill with random
// empty_like(tensor)      // Copy shape, uninitialized
// full_like(tensor, val)  // Copy shape, fill with value
// TODO: Simplify
// Algebraic Rewriting (Symbolic Solver)
// Transpose(Add(A, B)) → Add(Transpose(A), Transpose(B))
// TODO: The Program Synthesis (HUGE)
// Search over valid operations, filter by type system, find shape D.
// The type system prunes invalid programs automatically.
// Heuristics, Cost models, Depth limits, Possibly reinforcement learning
// TODO: Programmatic factors + differentiable weighting + causal inference
// mixture-of-experts over structured alphas
// TODO: Define mutation semantics and aliasing rules.
// Are tensors immutable?
// If mutable views are allowed, ensure no unsound aliasing.
// TODO: Causal Graph
// struct CausalGraph<F> {
//     adj: Tensor<F, DynRank<2>>,
// }
// Tensor Engine
//      ↑
// CausalGraph { adjacency: Tensor }
//      ↑
// Symbolic Reasoner (d-sep, ID)
//      ↑
// Query Engine (P(Y | do(X)))

#[derive(Debug, Clone)]
pub struct Tensor<F, Sh: Shape, E = Host<F, { Sh::RANK }>> {
    pub expr: E,
    pub _marker: PhantomData<(F, Sh)>,
}

impl<F: Data, Sh: Shape, E> Tensor<F, Sh, E> {
    pub(crate) fn wrap(expr: E) -> Self {
        Self {
            expr,
            _marker: PhantomData,
        }
    }

    pub fn align<NewShape>(self) -> Tensor<F, NewShape, TransposeExpr<E, { NewShape::RANK }>>
    where
        NewShape: Shape,
        Sh: Permutation<NewShape>,
    {
        let perm = <Sh as Permutation<NewShape>>::indices();

        Tensor::wrap(TransposeExpr {
            op: self.expr,
            perm,
        })
    }

    pub fn expand<Target>(
        self,
        target_sizes: [usize; Target::RANK],
    ) -> Tensor<F, Target, BroadcastExpr<E, { Sh::RANK }, { Target::RANK }>>
    where
        Target: Shape + BroadcastMap<Sh>,
    {
        let mapping = <Target as BroadcastMap<Sh>>::mapping();

        Tensor::wrap(BroadcastExpr {
            op: self.expr,
            target_shape: target_sizes,
            mapping,
        })
    }

    pub fn reshape<const NEW_R: usize>(
        self,
        new_shape: [usize; NEW_R],
    ) -> Tensor<F, DynRank<NEW_R>, ReshapeExpr<E, { Sh::RANK }, NEW_R>> {
        // TODO: Reshape Safety.
        // `reshape` must verify that `product(new_shape) == product(current_shape)`.
        // Currently, this check is deferred to runtime execution, which might panic deeply.
        // It's better to check here
        Tensor::wrap(ReshapeExpr {
            op: self.expr,
            new_shape,
        })
    }

    pub fn collect<B: Backend>(&self, backend: &mut B) -> Base<B::Storage<F>, F, { Sh::RANK }>
    where
        E: Evaluator<B, { Sh::RANK }, Data = F>,
    {
        self.expr.eval(backend)
    }

    pub fn to_vec<B: Backend>(&self, backend: &mut B) -> Vec<F>
    where
        E: Evaluator<B, { Sh::RANK }, Data = F>,
    {
        let view = self.expr.eval(backend);
        let dense = Lower::<PackDense, B>::lower(&view, backend);
        backend.to_host(&dense)
    }
}

// Algebraic Ops
impl<F: Semiring, Sh: Shape, E> Tensor<F, Sh, E>
where
    [(); Sh::RANK]: Sized,
{
    // TODO: Matrix Multiplication (.matmul / .dot).
    // This requires `ContractExpr`.
    // It's the most computationally intensive operation and usually delegates to BLAS/cuBLAS.

    // TODO: Scan (Prefix Sum / CumSum).
    // `ScanExpr` is needed for `cumsum`, `cumprod`, and RNNs.
    // It is fundamentally sequential (O(N)) unless parallel prefix sum algorithms are used.
    pub fn scale(self, factor: F) -> Tensor<F, Sh, MapExpr<E, ScaleKernel<F>>> {
        Tensor::wrap(MapExpr {
            op: self.expr,
            kernel: ScaleKernel { factor },
        })
    }
}

// Calculus Ops (Gradients, Physics)
impl<F: Real, Sh: Shape, E> Tensor<F, Sh, E> {
    pub fn forward<B: Backend>(
        &self,
        backend: &mut B,
    ) -> (
        Base<B::Storage<F>, F, { Sh::RANK }>,
        GradientTape<E::Adjoint>,
    )
    where
        E: Differentiable<B, { Sh::RANK }, Data = F>,
    {
        let (res, adjoint) = self.expr.forward(backend);
        (res, GradientTape::new(adjoint))
    }
}

impl<F: Data> Tensor<F, DynRank<0>, ConstExpr<F>> {
    pub fn scalar(value: F) -> Self {
        Tensor::wrap(ConstExpr(value))
    }
}

impl<F: Data, const R: usize> Tensor<F, DynRank<R>, Host<F, R>> {
    pub fn new(data: Vec<F>, shape: [usize; R]) -> Self {
        debug_assert_eq!(data.len(), shape.iter().product::<usize>());
        Tensor::wrap(Host::new(Arc::new(data), shape))
    }

    pub fn zeros(shape: [usize; R]) -> Self
    where
        F: Semiring,
    {
        let n = shape.iter().product();
        Self::new(vec![F::zero(); n], shape)
    }

    pub fn ones(shape: [usize; R]) -> Self
    where
        F: Semiring,
    {
        let n = shape.iter().product();
        Self::new(vec![F::one(); n], shape)
    }

    pub fn full(val: F, shape: [usize; R]) -> Self
    where
        F: Data,
    {
        let n = shape.iter().product();
        Self::new(vec![val; n], shape)
    }

    pub fn from_slice(data: &[F], shape: [usize; R]) -> Self {
        Self::new(data.to_vec(), shape)
    }

    pub fn into_named<NewSh: Shape>(self) -> Tensor<F, NewSh, Host<F, R>> {
        debug_assert_eq!(NewSh::RANK, R, "Rank mismatch");
        Tensor::wrap(self.expr)
    }
}

impl<F: Data> From<Vec<F>> for Tensor<F, DynRank<1>, Host<F, 1>> {
    fn from(vec: Vec<F>) -> Self {
        Tensor::wrap(<Vec<F> as Lift<F>>::lift(vec))
    }
}

impl<F: Data, const N: usize> From<[F; N]> for Tensor<F, DynRank<1>, Host<F, 1>> {
    fn from(arr: [F; N]) -> Self {
        Tensor::wrap(<[F; N] as Lift<F>>::lift(arr))
    }
}

impl<F: Data, const R: usize, const C: usize> From<[[F; C]; R]>
    for Tensor<F, DynRank<2>, Host<F, 2>>
{
    fn from(arr: [[F; C]; R]) -> Self {
        Tensor::wrap(<[[F; C]; R] as Lift<F>>::lift(arr))
    }
}

#[macro_export]
macro_rules! tensor {
    ($x:expr) => {{
        use ::algebra::TradingFloat;
        let val = TradingFloat::try_from($x).expect("Invalid float");
        $crate::Tensor::scalar(val)
    }};

    ($($x:expr),+ $(,)?) => {{
        use ::algebra::TradingFloat;

        $crate::Tensor::from([
            $(TradingFloat::try_from($x).expect("Invalid float")),+
        ])
    }};

    ($([$($x:expr),* $(,)?]),+ $(,)?) => {{
        use ::algebra::TradingFloat;

        $crate::Tensor::from([
            $([$(TradingFloat::try_from($x).expect("Invalid float")),*]),+
        ])
    }};
}

#[cfg(test)]
mod tests {
    use backend::GenericBackend;

    #[test]
    fn test_tensor_add() {
        let mut backend = GenericBackend::new();
        let a = tensor![2.0, 3.0, 5.0];
        let b = tensor![1.0, 3.0, 2.0];

        let c = a + b;

        let result = c.to_vec(&mut backend);

        assert_eq!(result[0].to_f64(), 3.0);
        assert_eq!(result[1].to_f64(), 6.0);
        assert_eq!(result[2].to_f64(), 7.0);
    }
}

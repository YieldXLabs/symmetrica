// Free algebra expression types
#[derive(Debug, Clone)]
pub struct IdentityExpr;

#[derive(Debug, Clone)]
pub struct ConstExpr<F>(pub F);

#[derive(Debug, Clone)]
pub struct LoadExpr<F>(pub F);

#[derive(Debug, Clone)]
pub struct LetExpr<Val, Body> {
    pub value: Val,
    pub body: Body,
}

#[derive(Debug, Clone)]
pub struct IfExpr<Cond, Then, Else> {
    pub cond: Cond,
    pub then_: Then,
    pub else_: Else,
}

#[derive(Debug, Clone)]
pub struct BroadcastExpr<Op, const R_IN: usize, const R_OUT: usize> {
    pub op: Op,
    pub target_shape: [usize; R_OUT],
    pub mapping: [Option<usize>; R_OUT],
}

#[derive(Debug, Clone)]
pub struct TransposeExpr<Op, const R: usize> {
    pub op: Op,
    pub perm: [usize; R],
}

#[derive(Debug, Clone)]
pub struct ReshapeExpr<Op, const R_IN: usize, const R_OUT: usize> {
    pub op: Op,
    pub new_shape: [usize; R_OUT],
}

// TODO: SliceExpr.
// `Select` reduces rank (getting a single index).
// We also need `Slice` to get a range (e.g., `tensor[0..10]`), which preserves rank.
#[derive(Debug, Clone)]
pub struct SelectExpr<Op> {
    pub op: Op,
    pub axis: usize,
    pub index: usize,
}

#[derive(Debug, Clone)]
pub struct ZipExpr<L, R, K> {
    pub left: L,
    pub right: R,
    pub kernel: K,
}

#[derive(Debug, Clone)]
pub struct MapExpr<Op, K> {
    pub op: Op,
    pub kernel: K,
}

#[derive(Debug, Clone)]
pub struct ContractExpr<L, R> {
    pub left: L,
    pub right: R,
    pub axes: (Vec<usize>, Vec<usize>),
}

// TODO: Windowing / Convolution.
// For trading, `RollingWindow` (e.g., Simple Moving Average) is distinct from Scan.
// Scan is recursive (state depends on prev state). Window is parallelizable.
// struct WindowExpr<Op, K> { op: Op, size: usize, kernel: K }
#[derive(Debug, Clone)]
pub struct ScanExpr<Op, F, K> {
    pub op: Op,
    pub init: F,
    pub kernel: K,
}

// TODO: Distance metrics
// Implementation strategy:
// 1. Define trait `Distance<F>` with `fn dist(a, b) -> F`.
// 2. `Euclidean` = Sqrt(Sum(Square(Sub(a, b)))) -> This can be composed of existing kernels!
// 3. `Cosine` = Div(Dot(a, b), Mul(Norm(a), Norm(b))).
//
// pub trait DistanceMetric<F: Real>: Copy + Clone + 'static {}
// #[derive(Debug, Clone, Copy)]
// pub struct Euclidean;
// impl<F: Real> DistanceMetric<F> for Euclidean {}

// #[derive(Debug, Clone, Copy)]
// pub struct CDistExpr<L, R, M> {
//     pub left: L,
//     pub right: R,
//     _metric: PhantomData<M>,
// }

// TODO: Rewrite Rules / Pattern Matching.
// To implement optimizations (e.g., x + 0 -> x), you need a "Rewrite" trait.
// Since Rust structs are static, you can't easily match `Add(_, Zero)` dynamically
// unless you use the Enum wrapper approach mentioned at the top.
//
// Example infrastructure needed:
// trait Rewriter {
//     fn rewrite(&self, expr: &Expr) -> Option<Expr>;
// }
//
// MatchExpr::new(|expr| matches!(expr, Add(_, Zero) => true));
// MatchExpr::new(|expr| matches!(expr, Sub(x, x) => true));
// MatchExpr::new(|expr| matches!(expr, Scale(Add(a,b), k) => Add(Scale(a,k), Scale(b,k))));

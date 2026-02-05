use algebra::{ConstExpr, Data};

// TODO: Lift Collection Types.
// Currently, we only lift scalars into `ConstExpr`.
// We should also support lifting:
// 1. `Vec<F>` -> `Host<F>` (Wrapping data into a tensor view).
// 2. `&[F]` -> `Host<F>` (Zero-copy view if possible).
pub trait Lift<F: Data> {
    type Output;

    fn lift(self) -> Self::Output;
}

impl<F: Data> Lift<F> for F {
    type Output = ConstExpr<F>;

    fn lift(self) -> Self::Output {
        // TODO: Const vs Load.
        // `ConstExpr` embeds the value into the AST struct.
        // For large structs, use `LoadExpr` with a reference/Arc to avoid
        // bloating the expression tree size.
        ConstExpr(self)
    }
}

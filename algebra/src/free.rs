pub enum Op<F, A> {
    // Pure value
    Pure(A),

    // Linear operators
    Add(A, A),
    Sub(A, A),
    Mul(A, A),
    Div(A, A),
    Scale(A, F),

    // Nonlinear
    Abs(A),
    Exp(A),
    Log(A),
    Sqrt(A),
    Pow(A, F),
    Sin(A),
    Cos(A),

    // Time-series
    Lag { input: A, n: usize },

    // Statistical
    Gaussian { x: A, mean: F, sigma: F },

    // Control / branching
    If { cond: A, then_: A, else_: A },

    // Comparison
    Min(A, A),
    Max(A, A),

    Clamp { x: A, lo: F, hi: F },

    Lt(A, A),
    Gt(A, A),
    Eq(A, A),
}

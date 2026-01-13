pub enum Op<F, A> {
    // Pure value
    Pure(A),

    // Linear operators
    Add(A, A),
    Mul(A, A),
    Scale(A, F),

    // Nonlinear
    Abs(A),
    Exp(A),
    Log(A),

    // Time-series
    Lag { input: A, n: usize },

    // Statistical
    Gaussian { x: A, mean: F, sigma: F },

    // Control / branching
    If { cond: A, then_: A, else_: A },
}

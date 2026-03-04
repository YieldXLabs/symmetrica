# AGENTS.md — symmetrica

> Authoritative guide for all AI coding agents and contributors.  
> Read completely before modifying any code.  
> If this document conflicts with generic Rust advice, **this document prevails**.

---

# 1. Project Identity

Symmetrica is a **type-driven tensor algebra** with **structural autodiff** and
**backend-independent semantics**, written in Rust.

It is **NOT**:
- A deep learning framework (not PyTorch, not JAX, not a wrapper around either)
- A numerical convenience library
- A runtime tape-based autodiff system
- A BLAS abstraction layer

It **IS**:
- The algebraic substrate that a DL framework could be built on top of
- A compile-time–verified tensor and morphism system
- A research-grade, numerically stable differentiation core occupying the layer
  between raw BLAS/LAPACK and a high-level autograd engine

All design decisions must reinforce this identity. If a change makes symmetrica
look more like PyTorch, it is the wrong change.

---

# 2. Workspace Architecture

```
symmetrica/
├── algebra/    # Abstract algebraic traits and laws
├── tensor/     # Type-level tensors + shape encoding
└── backend/    # Concrete execution (CPU, future GPU/WASM)
```

## Strict Dependency Rule

```
backend  →  tensor  →  algebra
```

**Forbidden — reject any PR that introduces:**
- `algebra/` importing from `tensor/` or `backend/`
- `tensor/` importing from `backend/`
- Any cyclic dependency

This DAG is the entire abstraction guarantee. One violation unravels it.

---

# 3. Rust Edition & Nightly Policy

Symmetrica targets **Rust 2024 edition** (stable since 1.85). Nightly features
are permitted only when explicitly justified, tracked for stabilization, and
listed in §13.

## 3.1 Breaking Changes in Effect (Rust 2024)

**`unsafe extern` is mandatory.** All FFI now requires:
```rust
// CORRECT
unsafe extern "C" {
    pub safe fn cblas_ddot(n: i32, x: *const f64, incx: i32,
                           y: *const f64, incy: i32) -> f64;
    pub unsafe fn free(p: *mut core::ffi::c_void);
}
// WRONG — does not compile in edition 2024
extern "C" { fn cblas_ddot(...) -> f64; }
```

**`unsafe fn` bodies require inner `unsafe {}` blocks:**
```rust
// CORRECT — explicit scope required
unsafe fn write_raw(ptr: *mut f32, val: f32) {
    unsafe { *ptr = val; }
}
```

**`gen` is a reserved keyword.** Never use it as an identifier anywhere.
Use `generate`, `produce`, or a domain-specific name instead.

**Tail expression drop order changed.** Temporaries in tail expressions are
dropped before local variables. In `backend/` where RAII guards control kernel
dispatch, never rely on the old implicit drop order.

---

# 4. Type System Doctrine

## 4.1 Invariants Belong in Types

Before adding any runtime check, ask: *"Can the compiler reject this instead?"*

Runtime shape assertions are design failures unless truly unavoidable. If you
find yourself writing `assert!(tensor.is_contiguous())`, the type system has
failed to enforce a contract it should own.

## 4.2 Tensor Rank Encoding

Rank MUST be encoded in const generics. Dynamic rank is prohibited in core
tensor types.

```rust
// CORRECT — rank is a compile-time fact
struct Tensor<T, const RANK: usize, B: Backend> {
    data:  B::Storage<T>,
    shape: [usize; RANK],
}

// WRONG — rank is runtime state, all static guarantees lost
struct Tensor<T, B: Backend> {
    data:  B::Storage<T>,
    shape: Vec<usize>,
}
```

## 4.3 `adt_const_params` — Layout as a Type (nightly, 2026 target)

Use `feature(adt_const_params)` to make layout a type-level invariant:

```rust
#![feature(adt_const_params)]

#[derive(PartialEq, Eq)]
struct Layout { row_major: bool }

struct Matrix<T, const L: Layout> { ... }
// Contiguous and strided are now distinct types — coercion is a compile error
```

Do NOT apply to types with privacy-sensitive or unsafe interior state. The RFC
governing permitted ADTs is still being finalized.

## 4.4 `min_generic_const_args` (nightly, prototype)

Use only where it eliminates a meaningful type-level workaround. Every use
must carry a `// MGCA: <reason>` comment:

```rust
trait HasRank { const RANK: usize; }
struct Dense<T: HasRank> where [(); T::RANK]: { ... }
```

## 4.5 Phantom Types for Algebraic Structure

Encode algebraic laws in types — never validate them at runtime:

```rust
// Scalar knows which field it lives in
struct Scalar<F: Field> {
    value:  F::Element,
    _field: PhantomData<F>,
}

// Gradient carries its originating space — mixing spaces is a compile error
struct Gradient<V: VectorSpace> {
    components: V::Dual,
    _space:     PhantomData<V>,
}
```

## 4.6 Broadcasting Policy

**Implicit broadcasting is prohibited.** NumPy/PyTorch-style silent shape
expansion is a frequent source of research bugs that are invisible in small
tests. All broadcasting must be an explicit `broadcast_to` morphism visible
in the type signature.

## 4.7 Memory Layout Is Part of the Type

Contiguous row-major, contiguous column-major, and strided views are distinct
types. Silent layout coercion is never allowed.

- `Tensor<T, N>` — owned, layout determined at construction
- `TensorView<'a, T, N>` — borrowed, layout preserved from source
- Never copy data to resolve a lifetime conflict — fix the lifetime

## 4.8 No `dyn Trait` in Hot Paths

`dyn Trait` is allowed only:
- At crate boundaries for error types
- For backend plugin registration during initialization only

Never use `Box<dyn Trait>` in `algebra/` or `tensor/`. Monomorphization is
the strategy — code size is a secondary concern.

## 4.9 Compile-Time Explosion Governance

Unconstrained generics compound compile times aggressively. To prevent this:
- Blanket impls over unconstrained generics are prohibited
- Core `algebra/` traits must be sealed
- Nested `Dual` types beyond second order (`Dual<Dual<T>>`) require design review
- Compile-time regressions over 15% must be investigated before merge
- Excessive monomorphization (binary size growth without perf justification)
  must be addressed before merge

---

# 5. Structural Autodiff Doctrine

## 5.1 Runtime Tape Is Forbidden

The following are categorically prohibited — no exceptions:

```rust
// EVERY form of this is wrong for this codebase
struct Tape { ops: Vec<Box<dyn Op>> }        // runtime graph
enum Op { Add, Mul, ... }                     // dynamic dispatch on ops
fn backward(&self, tape: &mut Tape) { ... }  // mutating graph traversal
```

Autodiff must be structural and type-driven. If the gradient of an operation
cannot be expressed as a type-level morphism, the algebraic structure is
incomplete — fix the algebra, not the grad system.

## 5.2 Forward Mode — Dual Numbers as Types

Forward mode (JVP) is implemented via dual numbers encoded structurally.
The nilpotent rule ε² = 0 must hold by construction, not by convention:

```rust
struct Dual<T: Field> {
    value:   T::Element,
    tangent: T::Element,   // dvalue/dinput carried forward
}

// Dual arithmetic: (a + εb)(c + εd) = ac + ε(ad + bc)
// ε² = 0 — the second tangent term vanishes structurally
impl<T: Ring> Mul for Dual<T> {
    type Output = Dual<T>;
    fn mul(self, rhs: Self) -> Dual<T> {
        Dual {
            value:   self.value * rhs.value,
            tangent: self.value * rhs.tangent + self.tangent * rhs.value,
        }
    }
}
```

## 5.3 Reverse Mode — Structural Pullbacks

Reverse mode (VJP) is more efficient when `outputs ≪ inputs` (typical DL).
The correct encoding uses a **Pullback** type — not a tape, not heap allocation:

```rust
trait Pullback<W: VectorSpace> {
    type Cotangent: VectorSpace;
    fn pullback(&self, cotangent: W) -> Self::Cotangent;
}

trait Differentiable {
    type Domain:     VectorSpace;
    type Codomain:   VectorSpace;
    type Derivative: LinearMap<Domain = Self::Domain, Codomain = Self::Codomain>;
    fn diff(&self) -> Self::Derivative;
}
```

## 5.4 Higher-Order Derivatives — HVPs, Never Full Hessians

Full Hessian matrices (`n×n`) must NOT be constructed by default. They are
O(n²) in memory and almost never required. All curvature-aware algorithms
— Newton-CG, L-BFGS, influence functions, Fisher information approximations
— work exclusively with Hessian-vector products (HVPs).

HVPs use **forward-over-reverse** (JVP applied to the gradient function):

```rust
// H(f)·v = ∂[x ↦ ∇f(x)] applied in direction v
// Cost: O(gradient) — same asymptotic complexity as first-order
trait HessianVectorProduct: Differentiable {
    fn hvp(&self, at: &Self::Domain, v: &Self::Domain) -> Self::Domain;
}
```

For exact second derivatives of scalar functions, use **hyper-dual numbers**
via type-level composition — do NOT add a `num_dual` dependency:

```rust
// Dual<Dual<T>> gives exact mixed partials (Hessian diagonal) for free
// ε₁² = 0, ε₂² = 0, ε₁ε₂ ≠ 0 — the mixed term IS the second derivative
type HyperDual<T> = Dual<Dual<T>>;
```

If you find yourself building an `n×n` matrix to represent curvature: stop,
and implement the HVP directly.

## 5.5 Stop-Gradient — Structural, Not Numerical

Stop-gradient boundaries MUST be type-level wrappers. Zeroing a gradient
numerically (`* 0.0`) is forbidden — the multiplication is still in the
computation graph and breaks correctness in composed systems:

```rust
// CORRECT — cotangent is structurally zero, compiler eliminates it
struct StopGrad<T>(T);

impl<T: Differentiable> Differentiable for StopGrad<T> {
    type Derivative = Zero;  // gradient does not flow through
    fn diff(&self) -> Zero { Zero }
}

// WRONG — zero numerically but structurally present; breaks composition
let loss = (pred - target * 0.0).norm();
```

Required for: target networks (RL), EMA parameters, normalization statistics,
discrete latents (VQ-VAE), any value that must not receive gradient updates.

## 5.6 Non-Smooth Operations

`ReLU`, `abs`, `sign`, `max`, `min` are not differentiable at kink points.

Rules:
1. Document the subgradient convention explicitly — frameworks disagree
   (0 vs 0.5 vs 1 at the kink), and the choice affects zero-initialization
   analysis and dead neuron detection in research.
2. Mark with `// NON-DIFFERENTIABLE: <reason and convention>`.
3. Silent subgradient choices are prohibited.

```rust
impl Differentiable for Relu {
    // NON-DIFFERENTIABLE: kink at x=0, subgradient = 0 (PyTorch convention)
    fn diff(&self, x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }
}
```

## 5.7 Gradient Checkpointing — Structural Recompute Boundaries

For deep differentiable chains (transformers, long sequences), naive reverse
mode stores all intermediate activations at O(depth) memory. Checkpointing
recomputes activations during the backward pass at the cost of extra compute.

In structural autodiff, checkpointing is a **recompute boundary in the morphism
composition** — not a runtime flag or boolean option:

```rust
struct Checkpoint<F: SmoothMap> {
    f:           F,
    saved_input: Option<F::Domain>, // only input stored; intermediates dropped
}
```

Do not implement checkpointing as a `bool` parameter. It changes the
mathematical contract of the morphism and must be explicit in the type.

---

# 6. Numerical Stability — Non-Negotiable

These rules exist because numerical errors in gradient-based research are
invisible in unit tests and catastrophic in multi-day training runs.

## 6.1 Floating-Point Is Not a Field

`f32` and `f64` do NOT satisfy the algebraic laws of a mathematical field:

- **Non-associativity**: `(a + b) + c ≠ a + (b + c)` in general
- **Non-exact inverses**: `a + (-a)` can produce `-0.0` or NaN in edge cases
- **Non-distributivity**: `a * (b + c) ≠ a*b + a*c` in general

When implementing `Field` or `Ring` for floating-point types, the doc comment
**must** state which laws hold approximately, to what tolerance, and under
what conditions they fail. Do not silently implement a field trait for `f32`.

## 6.2 No Silent NaN

If an operation can produce NaN, either encode invalid inputs in the type
system (making them unrepresentable) or return `Result`/`Option`.

```rust
// WRONG — negative input silently gives NaN, poisons entire gradient chain
fn sqrt_field(x: f64) -> f64 { x.sqrt() }

// CORRECT — invalid domain is explicit in the return type
fn sqrt_field(x: f64) -> Option<f64> {
    if x < 0.0 { None } else { Some(x.sqrt()) }
}
```

A single NaN in a gradient poisons the entire parameter update with no
runtime indication. Every potentially NaN-producing operation must be tested
at its boundary values.

## 6.3 Stability Annotations Required

Operations prone to overflow, catastrophic cancellation, or instability must
carry a `// STABILITY:` comment explaining the technique used:

```rust
// WRONG — overflows for logits > ~710, common in neural net outputs
fn log_sum_exp(xs: &[f64]) -> f64 {
    xs.iter().map(|x| x.exp()).sum::<f64>().ln()
}

// CORRECT — max-shifted, stable for all finite inputs
fn log_sum_exp(xs: &[f64]) -> f64 {
    let max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    // STABILITY: shift by max before exp to prevent overflow; log cancels shift
    max + xs.iter().map(|x| (x - max).exp()).sum::<f64>().ln()
}
```

Always prefer stable standard library primitives:
- `f64::exp_m1()` over `f64::exp() - 1.0` (catastrophic cancellation near 0)
- `f64::ln_1p()` over `(1.0 + x).ln()` (catastrophic cancellation near 0)

Required techniques where applicable: log-sum-exp shifting, Kahan summation,
Cholesky over LU for PSD matrices, log-space arithmetic for probability products.

## 6.4 Gradient Checks Mandatory

Every `Differentiable` implementation must include a gradient check test:

```rust
#[test]
fn grad_check_sigmoid() {
    for x in [-2.0, -0.5, 0.0, 0.5, 2.0_f64] {
        let eps       = 1e-5;
        let analytic  = sigmoid_derivative(x);
        let numerical = (sigmoid(x + eps) - sigmoid(x - eps)) / (2.0 * eps);
        // ALWAYS relative error — absolute thresholds fail for large/small x
        let rel_err   = (analytic - numerical).abs() / (numerical.abs() + 1e-8);
        assert!(rel_err < 1e-4,
            "grad check failed at x={x}: analytic={analytic}, numerical={numerical}");
    }
}
```

- Relative error tolerance: `1e-4` (standard in ML research)
- Absolute error thresholds are forbidden — they silently pass wrong gradients
- Tests must cover: `0.0`, `±∞`, subnormal inputs, kink points

## 6.5 Denormals and Backend Performance

Subnormal floats can cause 10–100× slowdowns via microcode assists on some
hardware. In `backend/`, document any kernel that may encounter denormals and
whether flush-to-zero (FTZ) mode is acceptable. FTZ is never acceptable for
`algebra/` operations that participate in autodiff — it changes the math.

---

# 7. Referential Transparency and Purity

`algebra/` and `tensor/` must be **referentially transparent** — same inputs,
same outputs, always. This is a correctness requirement for structural autodiff,
not a style preference.

## 7.1 Randomness Must Be Explicit

No RNG in `algebra/` or `tensor/`. Stochastic operations take randomness as
an explicit argument:

```rust
// WRONG — hidden RNG breaks referential transparency and autodiff
fn dropout(x: Tensor<f64, 2>, rate: f64) -> Tensor<f64, 2> {
    x.map(|v| if rand::random::<f64>() > rate { v } else { 0.0 })
}

// CORRECT — mask is explicit; function is pure; derivative is well-defined
fn dropout(x: Tensor<f64, 2>, mask: Tensor<bool, 2>) -> Tensor<f64, 2> {
    x.zip_map(mask, |v, keep| if keep { v } else { 0.0 })
}
```

The derivative of a stochastic function is only well-defined when the stochastic
choices are fixed. A function that internally samples randomness cannot be
correctly differentiated through — it will silently produce wrong gradients.

`backend/` generates masks and noise and passes them as tensors. The compute
graph in `tensor/` and `algebra/` remains deterministic and pure.

## 7.2 In-Place Operations

| Location | Status | Condition |
|---|---|---|
| `algebra/` | **Prohibited** | Always |
| `tensor/` | **Prohibited** | Always |
| `backend/` | Permitted | Only with proven exclusive ownership |

Autodiff-participating tensors must never be mutated in-place. The backward
pass requires forward-pass values to be intact.

---

# 8. Ownership and Memory Model

## 8.1 Owned vs Borrowed Tensors

- `Tensor<T, N>` — owned, allocates, has defined lifetime
- `TensorView<'a, T, N>` — borrowed view, zero-copy slice of owned data
- Slicing and transposing produce views, never silent copies
- Never copy data to resolve a lifetime conflict — fix the lifetime

## 8.2 Gradient Accumulation vs Replacement

Be explicit about whether a gradient operation **accumulates** (`+=`) or
**replaces** (`=`). Conflating these is the source of a large class of
optimizer bugs — especially in models with shared parameters or when using
gradient checkpointing. Document this in every gradient-producing function.

---

# 9. Crate-Level Contracts

## `algebra/`

- Pure trait definitions and their algebraic laws
- No concrete types, no allocations, no I/O, no RNG
- Every trait must document: algebraic structure, laws, and a reference
- New traits require `proptest` property tests verifying the laws
- **Autodiff composability:** every operation trait must have either a
  `Differentiable` impl or a `// NON-DIFFERENTIABLE: <reason>` annotation
- **Forbidden:** `std::collections`, `Box`, `Vec`, `Arc`, any randomness source

## `tensor/`

- Tensor types parameterized over `algebra` and `backend` traits only
- Shape arithmetic verified at compile time where RANK is statically known
- `Backend` is always a trait bound, never a concrete type
- Contraction, transpose, and broadcast operations preserve algebraic structure
- Memory layout is part of the type contract (§4.7)
- `Tensor<T, N>` vs `TensorView<'a, T, N>` ownership distinction is enforced (§8.1)

## `backend/`

- The ONLY crate that may import concrete execution dependencies
- All implementations gated behind Cargo features — never unconditionally compiled
- Unsafe code permitted here only; every block requires `// SAFETY:` (see §10)
- `tensor/` defines *what* to compute; `backend/` defines *how*
- Kernel-selection logic must never leak upward into `tensor/`
- `Backend::alloc` must document alignment (32-byte for AVX2, 64-byte for AVX-512)
- Alignment violations in SIMD are UB — they are not caught by tests
- Reductions must support deterministic mode with stable parallel reduction order

---

# 10. Unsafe Code Policy

```rust
// Required format — no exceptions
unsafe {
    // SAFETY: `ptr` is non-null by Backend::alloc postcondition
    // (see backend/src/contract.rs:L42). Alignment guaranteed to 32 bytes
    // by CpuBackend::alloc. Exclusive access proven by borrow checker.
    *ptr = value;
}
```

Rules:
1. No `unsafe` without `// SAFETY:` — enforced by `clippy::undocumented_unsafe_blocks`
2. Minimize scope to the exact operations that require it
3. Never use `unsafe` to work around a design problem — fix the design
4. Raw pointers in `tensor/` must be wrapped in safe abstractions before
   crossing module boundaries
5. `unsafe` in `algebra/` requires a prior design discussion

---

# 11. Testing Requirements

Every new feature must include all applicable categories. CI failures block
merge without exception.

```bash
cargo test --workspace              # full suite — CI gate
cargo test -p algebra -- --nocapture
cargo test proptest                 # algebraic law tests only
cargo test grad_check               # gradient check tests only
```

**Algebraic law tests (proptest)** — every new trait:
```rust
proptest! {
    #[test]
    fn ring_distributivity(a: f64, b: f64, c: f64) {
        let lhs = R64::mul(a, R64::add(b, c));
        let rhs = R64::add(R64::mul(a, b), R64::mul(a, c));
        // Relative error only — absolute thresholds are forbidden
        let rel_err = (lhs - rhs).abs() / (rhs.abs() + f64::EPSILON);
        prop_assert!(rel_err < 1e-10, "distributivity violated: {rel_err}");
    }
}
```

**Gradient check tests** — every `Differentiable` impl:
```rust
// Relative error, boundary values (0, ±∞, subnormals, kinks), 1e-4 tolerance
// See §6.4 for full pattern
```

**Shape/rank preservation** — every tensor operation:
```rust
// Const generic typing proves rank at compile time.
// Assert runtime dimensions explicitly in tests too.
```

**Numerical stability regression** — every stability-sensitive op:
```rust
// Must not NaN/overflow on large inputs, subnormals, or boundary values.
// See §6.3 for log_sum_exp pattern.
```

---

# 12. Documentation Standards

All public items must have doc comments. Trait docs must state: (1) the
algebraic structure modeled, (2) laws implementors must uphold, (3) a usage
example.

Use `# Errors`, `# Panics`, `# Safety` sections where applicable. Math
notation requires both LaTeX (backtick blocks) and a plain-English description.

```rust
/// A commutative ring over elements of type `Self::Element`.
///
/// # Laws
///
/// - **Associativity of +**: `(a + b) + c == a + (b + c)`
/// - **Commutativity of +**: `a + b == b + a`
/// - **Distributivity**: `a * (b + c) == a*b + a*c`
///
/// For floating-point impls, laws hold approximately. Implementors must
/// document tolerance and failure conditions.
///
/// Algebraic violations are not memory UB but produce mathematically
/// meaningless results with no runtime signal.
pub trait CommutativeRing: Ring { ... }
```

---

# 13. Rust 2026 Feature Horizon

| Feature | Status | Crate | Policy |
|---|---|---|---|
| `adt_const_params` | RFC in progress, 2026 target | `tensor/` | In use — see §4.3 |
| `min_generic_const_args` | Full prototype merged | `tensor/` | In use — annotate `// MGCA:` |
| `gen` keyword | Reserved in 2024 | all | Never use as identifier |
| Polonius borrow checker | Nightly | `backend/` | Needed for some view lifetimes |
| `std::autodiff` / Enzyme | Nightly | — | **DO NOT USE** — see below |
| New trait solver | Rolling out | `algebra/` | Fixes coherence edge cases |
| `std::offload` (GPU) | Experimental nightly | Future `backend/` | Watch; do not adopt yet |

When a feature stabilizes: remove its `#![feature(...)]` gate, update this
table. Do not adopt features not listed here without a design discussion first.

**On `std::autodiff` / Enzyme:** nightly ships `#[autodiff_forward]` and
`#[autodiff_reverse]` backed by LLVM Enzyme. These produce correct derivatives
— but at the LLVM-IR level, invisible to the type system, bypassing the entire
algebraic abstraction this codebase is built on. Using them here is the
equivalent of using `unsafe` to solve a type-safety problem. The answer is
always no.

---

# 14. Prohibited List

Categorically forbidden. Do not introduce. Do not rationalize exceptions.

| Prohibited | Reason |
|---|---|
| `ndarray`, `nalgebra`, `faer` | Bypass algebraic type encoding |
| Runtime autodiff tape (`Vec<Box<dyn Op>>`) | Contradicts structural autodiff |
| `std::autodiff` / Enzyme | LLVM-level, bypasses type system |
| Implicit broadcasting | Silent shape changes hide research bugs |
| `async` in compute paths | Parallelism is backend dispatch, not async |
| Python bindings (premature) | Freezes API before semantics are stable |
| Mixed precision in `algebra/`/`tensor/` | Precision is a backend concern |
| Optimizer state or training loops | Belong in a higher-level library |
| Hidden RNG in `algebra/`/`tensor/` | Breaks referential transparency |
| Full Hessian matrix construction | O(n²) memory; use HVPs instead |
| Numerical stop-gradient (`* 0.0`) | Must be structural (type wrapper) |
| Unexplained `#[allow(...)]` | All suppressions require justification |

---

# 15. PR Checklist

```bash
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo fmt --check
cargo doc --workspace --no-deps
```

**Correctness**
- [ ] No crate DAG violation (`algebra ← tensor ← backend`)
- [ ] No new `dyn Trait` in `algebra/` or `tensor/`
- [ ] All `unsafe` blocks have `// SAFETY:` comments
- [ ] No in-place operations in `algebra/` or on autodiff-participating tensors
- [ ] No hidden RNG in `algebra/` or `tensor/`
- [ ] `stop_gradient` is structural (type wrapper), not numerical (`* 0.0`)
- [ ] No implicit broadcasting introduced
- [ ] No full Hessian matrix constructed — HVPs only

**Numerical**
- [ ] New `Differentiable` impls have gradient check tests (relative error, 1e-4)
- [ ] Boundary values tested: `0.0`, `±∞`, subnormals, kink points
- [ ] Non-differentiable ops marked `// NON-DIFFERENTIABLE: <reason>`
- [ ] Floating-point `Field`/`Ring` impls document approximate law satisfaction
- [ ] Numerically sensitive ops have stability regression tests with `// STABILITY:` comment

**Structure**
- [ ] New algebraic traits have `proptest` tests using relative error
- [ ] New tensor ops have shape/rank preservation tests
- [ ] Autodiff composability annotated or non-differentiability documented
- [ ] PR description states the algebraic motivation, not just the code change

---

# Final Principle

Symmetrica encodes mathematics in the type system.

If a change weakens type-level guarantees,
introduces runtime autodiff,
relaxes algebraic laws,
compromises numerical stability,
or makes this codebase resemble a convenience wrapper —

**the change must be rejected.**

Correctness is structural.  
Differentiation is algebraic.  
Boundaries are inviolable.

---

*Last reviewed: 2026-03 — Senior Principal Deep Learning Researcher / Principal Architect*  
*Format: AGENTS.md — open standard stewarded by the Agentic AI Foundation / Linux Foundation*

# AGENTS.md — symmetrica

> Authoritative guide for all AI coding agents and contributors.  
> Read completely before modifying any code.  
> If this document conflicts with generic Rust advice, **this document prevails**.

| Field | Value |
|---|---|
| **Project status** | 🚧 WIP — no stable public API |
| **Owner** | YieldXLabs |
| **AGENTS.md version** | 2026-03 |
| **Review cadence** | Update this file whenever architecture decisions change. Stale guidance causes more damage than no guidance. |
| **Escalation path** | Open a GitHub Issue tagged `architecture` before any change that touches §16 (Prohibited List) or the §3 dependency DAG. Do not proceed unilaterally. |

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

# 2. Contributor Onboarding

**Estimated time to first meaningful PR: 2–3 days.**

If you are new to this codebase, complete these steps in order before writing
any code:

1. **Read this file completely.** Not skimming — reading. The prohibited list
   in §16 and the autodiff doctrine in §7 will save you from the most common
   wrong turns.

2. **Build and run the test suite:**
   ```bash
   rustup show                   # confirm pinned toolchain is active
   cargo build --workspace       # confirm clean build
   cargo test --workspace        # confirm all tests pass before you touch anything
   ```

3. **Read the three crate READMEs** (`algebra/`, `tensor/`, `backend/`) to
   understand what is already implemented and what is in progress.

4. **Find an open Issue tagged `good-first-issue`** before inventing new work.
   Do not add features that are not tracked in the issue tracker.

5. **Open a Draft PR early.** Don't work in isolation for more than a day.
   Draft PRs allow async feedback before significant time is invested.

**For AI agents specifically:** steps 3 and 4 require reading files in the
repository. Use the file tree, not assumptions from training data. The codebase
is WIP and diverges from any prior snapshot an agent may have seen.

---

# 3. Workspace Architecture

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

# 4. Project Scope and Current Status

## What is in scope now (WIP)

| Area | Status | Notes |
|---|---|---|
| `algebra/` trait hierarchy | 🔄 Active | Ring, Field, VectorSpace, Module |
| `tensor/` core types | 🔄 Active | Const-generic rank, owned + view |
| `backend/` CPU | 🔄 Active | BLAS bindings, basic kernels |
| Forward mode autodiff (JVP) | 🔄 Active | Dual number encoding |
| Reverse mode autodiff (VJP) | 🔄 Active | Structural pullbacks |

## What is explicitly out of scope until further notice

These items do not have open issues and should not be started without a
formal decision recorded in the issue tracker:

- GPU backend (CUDA, Vulkan, Metal, `std::offload`)
- Python bindings (PyO3)
- Higher-level training APIs (optimizers, loss functions, data loaders)
- Model serialization / checkpoint format
- Distributed execution
- WASM backend

If you believe one of these should be started, open an Issue for discussion
first. Do not begin implementation speculatively.

## Decision log

Significant architectural decisions that are not self-evident from the code
should be recorded as GitHub Issues tagged `decision-record` and linked here.
Agents and new contributors must read open decision-record issues before
making changes to `algebra/` public traits, as in-flight decisions may affect
what is acceptable.

---

# 5. Rust Edition & Nightly Policy

Symmetrica targets **Rust 2024 edition** (stable since 1.85). Nightly features
are permitted only when explicitly justified, tracked for stabilization, and
listed in §15.

## 5.1 Breaking Changes in Effect (Rust 2024)

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

# 6. Type System Doctrine

## 6.1 Invariants Belong in Types

Before adding any runtime check, ask: *"Can the compiler reject this instead?"*

Runtime shape assertions are design failures unless truly unavoidable. If you
find yourself writing `assert!(tensor.is_contiguous())`, the type system has
failed to enforce a contract it should own.

## 6.2 Tensor Rank Encoding

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

## 6.3 `adt_const_params` — Layout as a Type (nightly, 2026 target)

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

## 6.4 `min_generic_const_args` (nightly, prototype)

Use only where it eliminates a meaningful type-level workaround. Every use
must carry a `// MGCA: <reason>` comment:

```rust
trait HasRank { const RANK: usize; }
struct Dense<T: HasRank> where [(); T::RANK]: { ... }
```

## 6.5 Phantom Types for Algebraic Structure

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

## 6.6 Broadcasting Policy

**Implicit broadcasting is prohibited.** NumPy/PyTorch-style silent shape
expansion is a frequent source of research bugs that are invisible in small
tests. All broadcasting must be an explicit `broadcast_to` morphism visible
in the type signature.

## 6.7 Memory Layout Is Part of the Type

Contiguous row-major, contiguous column-major, and strided views are distinct
types. Silent layout coercion is never allowed.

- `Tensor<T, N>` — owned, layout determined at construction
- `TensorView<'a, T, N>` — borrowed, layout preserved from source
- Never copy data to resolve a lifetime conflict — fix the lifetime

## 6.8 No `dyn Trait` in Hot Paths

`dyn Trait` is allowed only:
- At crate boundaries for error types
- For backend plugin registration during initialization only

Never use `Box<dyn Trait>` in `algebra/` or `tensor/`. Monomorphization is
the strategy — code size is a secondary concern.

## 6.9 Compile-Time Explosion Governance

Unconstrained generics compound compile times aggressively. To prevent this:
- Blanket impls over unconstrained generics are prohibited
- Core `algebra/` traits must be sealed
- Nested `Dual` types beyond second order (`Dual<Dual<T>>`) require design review
- Compile-time regressions over 15% must be investigated before merge
- Excessive monomorphization (binary size growth without perf justification)
  must be addressed before merge

---

# 7. Structural Autodiff Doctrine

## 7.1 Runtime Tape Is Forbidden

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

## 7.2 Forward Mode — Dual Numbers as Types

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

## 7.3 Reverse Mode — Structural Pullbacks

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

## 7.4 Higher-Order Derivatives — HVPs, Never Full Hessians

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

## 7.5 Stop-Gradient — Structural, Not Numerical

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

## 7.6 Non-Smooth Operations

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

## 7.7 Gradient Checkpointing — Structural Recompute Boundaries

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

# 8. Numerical Stability — Non-Negotiable

These rules exist because numerical errors in gradient-based research are
invisible in unit tests and catastrophic in multi-day training runs.

## 8.1 Floating-Point Is Not a Field

`f32` and `f64` do NOT satisfy the algebraic laws of a mathematical field:

- **Non-associativity**: `(a + b) + c ≠ a + (b + c)` in general
- **Non-exact inverses**: `a + (-a)` can produce `-0.0` or NaN in edge cases
- **Non-distributivity**: `a * (b + c) ≠ a*b + a*c` in general

When implementing `Field` or `Ring` for floating-point types, the doc comment
**must** state which laws hold approximately, to what tolerance, and under
what conditions they fail. Do not silently implement a field trait for `f32`.

## 8.2 No Silent NaN

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

## 8.3 Stability Annotations Required

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

## 8.4 Gradient Checks Mandatory

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

## 8.5 Denormals and Backend Performance

Subnormal floats can cause 10–100× slowdowns via microcode assists on some
hardware. In `backend/`, document any kernel that may encounter denormals and
whether flush-to-zero (FTZ) mode is acceptable. FTZ is never acceptable for
`algebra/` operations that participate in autodiff — it changes the math.

---

# 9. Referential Transparency and Purity

`algebra/` and `tensor/` must be **referentially transparent** — same inputs,
same outputs, always. This is a correctness requirement for structural autodiff,
not a style preference.

## 9.1 Randomness Must Be Explicit

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

## 9.2 In-Place Operations

| Location | Status | Condition |
|---|---|---|
| `algebra/` | **Prohibited** | Always |
| `tensor/` | **Prohibited** | Always |
| `backend/` | Permitted | Only with proven exclusive ownership |

Autodiff-participating tensors must never be mutated in-place. The backward
pass requires forward-pass values to be intact.

---

# 10. Ownership and Memory Model

## 10.1 Owned vs Borrowed Tensors

- `Tensor<T, N>` — owned, allocates, has defined lifetime
- `TensorView<'a, T, N>` — borrowed view, zero-copy slice of owned data
- Slicing and transposing produce views, never silent copies
- Never copy data to resolve a lifetime conflict — fix the lifetime

## 10.2 Gradient Accumulation vs Replacement

Be explicit about whether a gradient operation **accumulates** (`+=`) or
**replaces** (`=`). Conflating these is the source of a large class of
optimizer bugs — especially in models with shared parameters or when using
gradient checkpointing. Document this in every gradient-producing function.

---

# 11. Crate-Level Contracts

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
- Memory layout is part of the type contract (§6.7)
- `Tensor<T, N>` vs `TensorView<'a, T, N>` ownership distinction is enforced (§10.1)

## `backend/`

- The ONLY crate that may import concrete execution dependencies
- All implementations gated behind Cargo features — never unconditionally compiled
- Unsafe code permitted here only; every block requires `// SAFETY:` (see §12)
- `tensor/` defines *what* to compute; `backend/` defines *how*
- Kernel-selection logic must never leak upward into `tensor/`
- `Backend::alloc` must document alignment (32-byte for AVX2, 64-byte for AVX-512)
- Alignment violations in SIMD are UB — they are not caught by tests
- Reductions must support deterministic mode with stable parallel reduction order

---

# 12. Unsafe Code Policy

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

# 13. Testing Requirements

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
// See §8.4 for full pattern
```

**Shape/rank preservation** — every tensor operation:
```rust
// Const generic typing proves rank at compile time.
// Assert runtime dimensions explicitly in tests too.
```

**Numerical stability regression** — every stability-sensitive op:
```rust
// Must not NaN/overflow on large inputs, subnormals, or boundary values.
// See §8.3 for log_sum_exp pattern.
```

---

# 14. Documentation Standards

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

# 15. Rust 2026 Feature Horizon

| Feature | Status | Crate | Policy |
|---|---|---|---|
| `adt_const_params` | RFC in progress, 2026 target | `tensor/` | In use — see §6.3 |
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

# 16. Prohibited List

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

# 17. PR Checklist

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

# 18. Change Process

Not all changes are equal. Use the right process for the scope of work:

| Change type | Process |
|---|---|
| Bug fix, test addition, doc improvement | Open PR directly |
| New operation in existing trait | Open PR with algebraic motivation in description |
| New trait in `algebra/` | Open Issue first, tag `design`, get acknowledgment before coding |
| New crate or workspace restructure | Open Issue, tag `architecture`, requires explicit sign-off |
| Anything in §16 Prohibited List | Open Issue, tag `architecture`. Do not begin without explicit approval. |
| Nightly feature not in §15 table | Open Issue, tag `nightly-feature`, before any use |

**Draft PRs are strongly encouraged** for anything beyond a trivial fix. The
cost of early feedback is zero. The cost of a large PR going in the wrong
direction is high.

**Commit message format:**
```
<crate>: <short imperative description>

<algebraic or mathematical motivation if non-obvious>
Relates to: #<issue>
```

Examples:
```
algebra: add Norm trait for normed vector spaces

tensor: fix rank arithmetic in contract() for RANK=0 edge case
Relates to: #47

backend: gate AVX-512 kernel behind cpu-avx512 feature flag
```

---

# 19. Known Limitations and Active Rough Edges

**Read this before hitting a wall and assuming the codebase is broken.**

These are known issues that are deferred — not bugs to fix unsolicited:

- **`adt_const_params` ICE on complex ADTs:** The nightly feature can panic
  the compiler on some nested const generic patterns. If you hit an ICE,
  file a minimal reproducer as a GitHub Issue tagged `compiler-bug` and
  work around it with a simpler encoding temporarily.

- **Trait solver coherence failures in `algebra/`:** The new Rust trait solver
  (replacing Chalk) can reject some associated-type bounds that the old solver
  accepted, and vice versa. If you get unexpected coherence errors, note the
  nightly version in the issue and tag `trait-solver`.

- **`tensor/` view lifetimes with Polonius:** Some `TensorView` lifetime
  patterns require the Polonius borrow checker (`-Z polonius`) to compile.
  These patterns are intentionally kept and are not to be worked around by
  cloning — they will compile on stable once Polonius stabilizes.

- **No benchmarks yet:** `backend/` has no formal benchmark suite. Performance
  claims cannot be verified. Do not make performance-motivated changes without
  first adding a benchmark that proves the claim.

- **`algebra/` trait hierarchy is incomplete:** Several standard algebraic
  structures (Lie algebras, Hilbert spaces, manifolds) are not yet represented.
  Do not add them without a design Issue — the hierarchy interactions are
  non-trivial.

---

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
# Appendix A: Machine-Readable Rule Index

> For AI agents and automated tooling. Each rule carries a **priority label**
> that defines both urgency and blocking behaviour. Work violations in
> P0 → P1 → P2 → P3 order. Do not submit a PR with any P0 or P1 violation.

## Priority Definitions

| Priority | Meaning | Blocks Merge | Exception Path |
|---|---|---|---|
| **P0** | Architectural invariant — CI enforces automatically | ✅ Yes — hard block | None |
| **P1** | Architectural invariant — requires human verification | ✅ Yes — reviewer blocks | Architect sign-off, recorded in Issue |
| **P2** | Quality standard — flagged in review | ❌ No | Reviewer judgment |
| **P3** | Advisory — noted in PR comment | ❌ No | None needed |

---

## A.1 Rule Registry

| ID | Priority | Rule | Automation | Caveat |
|---|---|---|---|---|
| `R-001` | **P0** | No crate DAG violation | ✅ grep `use tensor::\|use backend::` scoped per crate | Path-based imports escape — see §A.4 |
| `R-002` | **P1** | No runtime autodiff tape | ⚠️ grep `Vec<Box<dyn` (partial) | Renamed types evade — human must verify new struct definitions |
| `R-003` | **P0** | Unsafe requires `// SAFETY:` | ✅ `clippy::undocumented_unsafe_blocks` | Do NOT replace with grep — clippy handles look-behind |
| `R-004` | **P1** | Stop-gradient must be structural | 🔴 Manual only | "gradient context" not machine-detectable |
| `R-005` | **P1** | No implicit broadcasting | 🔴 Manual only | Context-dependent shape analysis |
| `R-006` | **P1** | No full Hessian construction | 🔴 Manual only | Requires semantic understanding |
| `R-007` | **P0** | No hidden RNG in `algebra/`/`tensor/` | ✅ grep `rand::` scoped per crate | Method-only usage escapes — see §A.4 |
| `R-008` | **P0** | No `Box<dyn Trait>` in `algebra/`/`tensor/` | ✅ grep `Box<dyn` scoped per crate | None |
| `R-009` | **P2** | New trait needs `proptest` property tests | 🔴 Manual only | N/A |
| `R-010` | **P2** | `Differentiable` impl needs gradient check | ✅ `cargo test grad_check` | Low FP risk |
| `R-011` | **P2** | Float `Field`/`Ring` impl needs law doc | 🔴 Manual only | N/A |
| `R-012` | **P2** | Numerically sensitive op needs `// STABILITY:` | ⚠️ word-boundary grep `\bexp\(` etc. | High FP risk with bare `exp` pattern — use `\bexp\(` |
| `R-013` | **P2** | Non-smooth op needs `// NON-DIFFERENTIABLE:` | ⚠️ grep function names | Catches names, not all usages |
| `R-014` | **P2** | Nightly feature use needs `// MGCA:` comment | ⚠️ grep feature gate names | N/A |
| `R-015` | **P3** | Commit message follows `<crate>: <description>` format | 🔴 Manual only | Advisory |
| `R-016` | **P3** | Draft PR opened within 1 day of starting work | 🔴 Manual only | Advisory |

**P0 count: 4 rules fully automated.**  
**P1 count: 4 rules require human review — they block merge but cannot be scripted.**  
**P2 count: 6 rules are quality flags — they do not block merge.**  
**P3 count: 2 advisory rules — noted but not enforced.**

---

## A.2 Enforcement Script

```bash
#!/bin/bash
# scripts/enforce_agents.sh
# Enforces P0 rules automatically. P1 rules require human review.
# P2/P3 rules are advisory — flagged but do not exit 1.

set -euo pipefail
FAILED=0

echo "=== AGENTS.md P0/P1 Enforcement ==="
echo "P0: auto-block | P1: human-review block | P2: advisory flag"
echo ""

# ── P0: R-001 — Crate DAG ─────────────────────────────────────────────────
# Caveat: path-based imports (tensor::Foo) and extern crate not caught (§A.4)
if grep -rn "^use tensor::\|^use backend::" algebra/src/ 2>/dev/null; then
    echo "❌ P0 / R-001: algebra/ imports tensor/ or backend/"
    FAILED=1
fi
if grep -rn "^use backend::" tensor/src/ 2>/dev/null; then
    echo "❌ P0 / R-001: tensor/ imports backend/"
    FAILED=1
fi

# ── P0: R-003 — Undocumented unsafe ───────────────────────────────────────
# clippy implements 3-line look-behind correctly; do NOT replace with grep
cargo clippy --workspace -- -D clippy::undocumented_unsafe_blocks 2>&1     || { echo "❌ P0 / R-003: Undocumented unsafe block"; FAILED=1; }

# ── P0: R-007 — Hidden RNG ────────────────────────────────────────────────
# Import-level only; method-only usage escapes (see §A.4)
if grep -rn "rand::" algebra/src/ tensor/src/ 2>/dev/null; then
    echo "❌ P0 / R-007: RNG import in algebra/ or tensor/"
    FAILED=1
fi

# ── P0: R-008 — Box<dyn Trait> in hot paths ───────────────────────────────
if grep -rn "Box<dyn" algebra/src/ tensor/src/ 2>/dev/null; then
    echo "❌ P0 / R-008: Box<dyn Trait> in algebra/ or tensor/"
    FAILED=1
fi

# ── P1: R-002 — Runtime tape (partial) ────────────────────────────────────
# Renamed types evade this — human must verify any new struct definitions
if grep -rn "Vec<Box<dyn" algebra/src/ tensor/src/ 2>/dev/null; then
    echo "⛔ P1 / R-002: Possible runtime tape — HUMAN REVIEW REQUIRED before merge"
    FAILED=1
fi

# ── Standard CI gates ─────────────────────────────────────────────────────
cargo test --workspace        || { echo "❌ Tests failed"; FAILED=1; }
cargo fmt --check             || { echo "❌ Format check failed"; FAILED=1; }
cargo doc --workspace --no-deps 2>&1 | grep "^error"     && { echo "❌ Doc build errors"; FAILED=1; } || true

# ── P2: Advisory flags (do not exit 1) ────────────────────────────────────
echo ""
echo "=== P2 Advisory Flags ==="

# R-010: Gradient check tests
cargo test grad_check --workspace 2>&1     || echo "⚠️  P2 / R-010: grad_check tests missing or failing — add before merge"

# R-012: Stability annotations — word-boundary pattern to minimise false positives
# Matches exp(, ln(, sqrt(, log( as calls — not identifiers like 'expect', 'export'
if grep -rPn '\b(exp|ln|log|sqrt)\(' algebra/src/ tensor/src/ 2>/dev/null     | grep -v "// STABILITY:" | grep -v "exp_m1\|ln_1p"; then
    echo "⚠️  P2 / R-012: Numerically sensitive op without // STABILITY: comment"
fi

# R-013: Non-smooth ops
if grep -rn "\brelu\|\babs\|\bsign\b" algebra/src/ tensor/src/ 2>/dev/null     | grep -v "// NON-DIFFERENTIABLE:"; then
    echo "⚠️  P2 / R-013: Non-smooth op without // NON-DIFFERENTIABLE: annotation"
fi

# ── Final verdict ─────────────────────────────────────────────────────────
echo ""
if [ $FAILED -eq 1 ]; then
    echo "❌ P0/P1 violations found — merge blocked."
    echo "   P1 rules R-004 (stop-gradient), R-005 (broadcasting), R-006 (Hessian)"
    echo "   cannot be scripted — verify manually against §16 Prohibited List."
    exit 1
fi

echo "✅ P0 rules passed. P2/P3 flags above are advisory."
echo "   Manually verify P1 rules: R-002, R-004, R-005, R-006."
```

---

## A.3 Known Detection Gaps

Automation has limits. These gaps must be compensated by human PR review:

| Rule | Gap | What Escapes | Human Mitigation |
|---|---|---|---|
| `R-001` | Path-based imports | `tensor::Foo::new()` without `use tensor::` | Audit Cargo.toml `[dependencies]` for each crate |
| `R-001` | Aliased imports | `use t = tensor; t::Foo` | Same Cargo.toml audit |
| `R-002` | Renamed types | `struct Graph { ops: Vec<Box<dyn Backward>> }` | Review all new `struct` definitions in `algebra/`/`tensor/` |
| `R-007` | Method-only RNG | `thread_rng().gen::<f64>()` without `use rand::` | Flag any call containing `rng`, `random`, `gen` in `algebra/`/`tensor/` |
| `R-012` | False positives | `explanation`, `expect`, `export` contain `exp` | Script uses `\bexp\(` word-boundary — review flagged lines |

---

## A.4 Priority Quick-Reference for Agents

When encountering a rule violation, apply this decision tree:

```
Is the violation detected by the CI script?
├── YES → CI will block the PR automatically (P0). Fix before pushing.
└── NO  → Is it in the P1 list (R-002, R-004, R-005, R-006)?
           ├── YES → Do NOT proceed. Open a GitHub Issue tagged `architecture`.
           │         A human architect must review before merge.
           └── NO  → Is it in the P2 list (R-009 through R-014)?
                     ├── YES → Note it in the PR description. Does not block merge.
                     └── NO  → P3 advisory. Note if convenient.
```

**For AI agents specifically:** if you are uncertain which priority a potential
violation belongs to, default to P1 behaviour — stop and flag it. The cost of
a false P1 flag is one human review. The cost of a missed P0/P1 is an
architectural violation in the codebase.

---

*Last reviewed: 2026-03 — Principal Project Manager / Senior Principal DL Researcher / Principal Architect*  
*Format: AGENTS.md — open standard stewarded by the Agentic AI Foundation / Linux Foundation*  
*Next scheduled review: when any §16 item is reconsidered, or at next major milestone*
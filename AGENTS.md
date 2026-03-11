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

3. **Read the crate READMEs** (`algebra/`, `tensor/`, `backend/`) to
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
├── tensor/     # Tensor<T, B: Backend> — parameterised over the trait, not any impl
│   └── shape/  # Compile-time shape algebra (see §20)
└── backend/    # trait Backend (abstract interface) + concrete impls (CpuBackend, ...)
```

## Dependency Graph

```
            backend/ (trait Backend)
            /                      \
      (impls)                    (uses)
     /                                \
backend/ (CpuBackend, ...)        tensor/ (Tensor<T, B: Backend>)
```

`backend/` has two internally distinct layers (see §11):
- **Layer 1** — `pub trait Backend` (abstract, no `unsafe`, no BLAS)
- **Layer 2** — `pub struct CpuBackend`, future `CudaBackend`, etc. (concrete impls)

`tensor/` depends on `backend/` for the `Backend` trait only. It must never
name a concrete impl type. Concrete backends are injected by callers at the use
site — `tensor/` is permanently unaware of which impl is active.

## Strict Dependency Rules

**Forbidden — reject any PR that introduces:**
- `algebra/` importing from `tensor/` or `backend/`
- `tensor/` naming any concrete backend type (`CpuBackend`, `CudaBackend`, etc.)
- `tensor/` importing anything from `backend/`'s Layer 2 modules
- Any cyclic dependency

**Permitted and required:**
- `backend/` importing from `algebra/`
- `tensor/` importing the `Backend` trait from `backend/`

This DAG is the entire abstraction guarantee. One violation unravels it.

---

# 4. Project Scope and Current Status

## What is in scope now (WIP)

| Area | Status | Notes |
|---|---|---|
| `algebra/` trait hierarchy | 🔄 Active | Ring, Field, VectorSpace, Module |
| `tensor/` core types | 🔄 Active | Const-generic rank, owned + view |
| `tensor/shape` module | 🔄 Active | Compile-time shape algebra (see §20) |
| `backend/` CPU impl | 🔄 Active | BLAS bindings, basic kernels |
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
// CORRECT — rank is a compile-time fact; B: Backend is the abstract trait from backend/
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

`Backend` here is the abstract trait defined in `backend/`. `B` is never
`CpuBackend`, `CudaBackend`, or any other concrete type. Those are injected
by callers; `tensor/` is permanently unaware of which impl is used.

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
in the type signature, defined in `tensor::shape` (see §20).

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
- **Compile-time regressions over 10% must be investigated and resolved before merge**
  (see §22 for the full compile-time budget policy)
- Excessive monomorphization (binary size growth without perf justification)
  must be addressed before merge

## 6.10 Associated Type Selection Policy — Regular vs GAT

The decision rule is single and non-negotiable:

> **Use a GAT if and only if the associated type's validity depends on a
> borrow from `Self`. Use a regular associated type for everything else.**

Mixing these up in either direction is wrong. A GAT where a regular type
suffices propagates lifetime noise through the entire trait hierarchy. A
regular associated type where a GAT is needed forces the wrong choice between
cloning (destroying zero-copy) and unsoundness (lying about lifetimes).

### Regular associated types — lifetime-independent

These describe *what kind of thing* a trait works with. They carry no borrow
from `Self` and are fully owned or statically known:

```rust
trait TensorAlgebra {
    type Scalar: Field;          // the numeric type — owns nothing
    type Gradient: VectorSpace;  // the cotangent element — fully owned
    type Shape: ShapeEncoding;   // compile-time rank/dimension info
    type Backend: Backend;       // the execution interface — a type, not a borrow
}
```

Adding a lifetime to any of these four is a design error. If `Gradient` seems
to need a lifetime, the gradient is not being modelled as a value — fix the
algebra.

### GATs — lifetime tied to a borrow from `Self`

These describe *something that borrows from `Self`* and cannot outlive it.
The lifetime must be present in the associated type because the associated
type's validity is a function of the source's lifetime:

```rust
trait TensorOps {
    // View borrows from self — cannot outlive the source tensor
    type View<'a>: TensorView where Self: 'a;

    // Iterator holds a reference into self — same constraint
    type Iter<'a>: Iterator where Self: 'a;

    // A borrowed slice of the underlying storage
    type Slice<'a>: StorageSlice where Self: 'a;

    // Zero-copy structural transform (transpose, reshape) — reinterprets
    // borrowed memory without allocating; must not outlive source
    type ZeroCopyTransform<'a>: TensorView where Self: 'a;
}
```

### Differentiable projections — own by default, GAT only if checkpointing

A differentiable projection is a morphism `f: A → B` with a structural
pullback `f*: B* → A*`. In structural autodiff, pullbacks are values — they
carry whatever forward-pass data they need **by value**, not by borrow:

```rust
// CORRECT — pullback owns its data; no lifetime required
trait Differentiable {
    type Domain:     VectorSpace;
    type Codomain:   VectorSpace;
    type Derivative: LinearMap; // fully owned — regular associated type
    fn diff(&self) -> Self::Derivative;
}
```

A GAT is required only if the projection *borrows* a forward activation rather
than owning it — which arises specifically in gradient checkpointing, where the
recompute boundary holds a reference to the saved input:

```rust
// CORRECT — Checkpoint borrows saved_input; projection lifetime is real
trait CheckpointedMap {
    type Projection<'a>: DifferentiableMap where Self: 'a;
    fn project<'a>(&'a self) -> Self::Projection<'a>;
}
```

If you find yourself adding a lifetime to `Derivative` outside of a
checkpointing context: stop. The pullback should own its data. If it cannot
own its data, the forward pass is retaining too much state — restructure it.

### Decision table

| Associated type | Use | Reason |
|---|---|---|
| `Scalar` | Regular | Lifetime-independent numeric type |
| `Gradient` | Regular | Fully owned cotangent value |
| `Shape` | Regular | Compile-time encoding, no borrow |
| `Backend` | Regular | Execution interface, no borrow from tensor |
| `View<'a>` | GAT | Borrows from source tensor |
| `Iter<'a>` | GAT | Holds reference into storage |
| `Slice<'a>` | GAT | Borrowed sub-storage |
| `ZeroCopyTransform<'a>` | GAT | Reinterprets borrowed memory |
| `Derivative` (autodiff) | Regular | Pullback owns its data |
| `Projection<'a>` (checkpoint only) | GAT | Borrows saved forward activation |

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

- Pure trait definitions and their algebraic laws — `Ring`, `Field`, `VectorSpace`, `Module`, `Differentiable`, etc.
- **Zero workspace dependencies** — `algebra/` is the root; nothing in this workspace imports into it
- No concrete types, no allocations, no I/O, no RNG
- Every trait must document: algebraic structure, laws, and a reference
- New traits require `proptest` property tests verifying the laws
- **Autodiff composability:** every operation trait must have either a
  `Differentiable` impl or a `// NON-DIFFERENTIABLE: <reason>` annotation
- **Forbidden:** `std::collections`, `Box`, `Vec`, `Arc`, any randomness source,
  any import from `tensor/` or `backend/`

## `tensor/`

- Defines `Tensor<T, const RANK: usize, B: Backend>` — parameterised over the
  **abstract `Backend` trait**, never over a concrete impl type
- Depends on `algebra/` and `backend/` (for the trait only)
- `B: Backend` is always a trait bound; concrete types (`CpuBackend`, etc.) are
  injected by callers and must never be named inside `tensor/`
- Shape arithmetic is defined and verified in the `tensor::shape` submodule (see §20)
- Contraction, transpose, and broadcast operations preserve algebraic structure
- Memory layout is part of the type contract (§6.7)
- `Tensor<T, N>` vs `TensorView<'a, T, N>` ownership distinction is enforced (§10.1)

## `backend/`

`backend/` has two internally distinct layers. Keep them clean:

**Layer 1 — Abstract interface** (`backend/src/lib.rs` or `backend/src/trait.rs`):
- Defines `pub trait Backend` — the sole contract between `tensor/` and execution
- Depends on `algebra/` only at this layer
- No concrete structs, no SIMD, no BLAS calls, no `unsafe`
- Adding a method to `trait Backend` is a breaking change for all impls —
  requires a design Issue before any modification

**Layer 2 — Concrete implementations** (`backend/src/cpu/`, `backend/src/gpu/`):
- `struct CpuBackend` — current CPU/BLAS implementation
- Future: `struct CudaBackend`, `struct WasmBackend`, etc.
- The ONLY location where `unsafe`, SIMD, and BLAS bindings are permitted
- The ONLY location where kernel fusion may be applied (see §19)
- All kernels gated behind Cargo features — never unconditionally compiled
- `alloc` implementations must document alignment (32-byte for AVX2, 64-byte for AVX-512)
- Alignment violations in SIMD are UB — tests will not catch them
- Reductions must support deterministic mode with stable parallel reduction order
- All non-trivial kernels must have a `criterion` benchmark (see §23)
- GPU/WASM impls are out of scope — do not add without a tracked design Issue (see §4)

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
5. `unsafe` in `algebra/` or `tensor/` requires a prior design discussion;
   `unsafe` in `backend/` is permitted only in Layer 2 (concrete impls), never in the trait definition

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
| Polonius borrow checker | Nightly | `tensor/` | Needed for some view lifetimes |
| `std::autodiff` / Enzyme | Nightly | — | **DO NOT USE** — see below |
| New trait solver | Rolling out | `algebra/` | Fixes coherence edge cases |
| `std::offload` (GPU) | Experimental nightly | Future `backend/` GPU layer | Watch; do not adopt yet |

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
| Kernel fusion outside `backend/` Layer 2 | Tensor semantics must remain pure; fusion is a backend optimization (see §19) |
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
- [ ] No crate DAG violation — `algebra/` has zero workspace deps; `tensor/` depends on `algebra/` + `backend/` (trait only, never concrete types); `backend/` depends on `algebra/`
- [ ] No new `dyn Trait` in `algebra/` or `tensor/`
- [ ] All `unsafe` blocks have `// SAFETY:` comments
- [ ] No in-place operations in `algebra/` or on autodiff-participating tensors
- [ ] No hidden RNG in `algebra/` or `tensor/`
- [ ] `stop_gradient` is structural (type wrapper), not numerical (`* 0.0`)
- [ ] No implicit broadcasting introduced
- [ ] No full Hessian matrix constructed — HVPs only
- [ ] No kernel fusion in `algebra/` or `tensor/` — fusion is backend-only (§19)

**Numerical**
- [ ] New `Differentiable` impls have gradient check tests (relative error, 1e-4)
- [ ] Boundary values tested: `0.0`, `±∞`, subnormals, kink points
- [ ] Non-differentiable ops marked `// NON-DIFFERENTIABLE: <reason>`
- [ ] Floating-point `Field`/`Ring` impls document approximate law satisfaction
- [ ] Numerically sensitive ops have stability regression tests with `// STABILITY:` comment

**Structure**
- [ ] New algebraic traits have `proptest` tests using relative error
- [ ] New tensor ops have shape/rank preservation tests via `tensor::shape` (§20)
- [ ] Autodiff composability annotated or non-differentiability documented
- [ ] PR description states the algebraic motivation, not just the code change

**Performance**
- [ ] No compile-time regression >10% (see §22)
- [ ] New `backend/` kernels have a `criterion` benchmark (see §23)

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

# 19. Kernel Fusion Policy

Kernel fusion — combining multiple tensor operations into a single backend
kernel to reduce memory bandwidth — is a valid and important optimization.
It is also a source of subtle semantic violations if applied at the wrong
abstraction layer.

## 19.1 The Invariant

**Kernel fusion is permitted only in `backend/` Layer 2 (concrete impls).**

`algebra/` and `tensor/` define *what* an operation means algebraically.
`backend/` defines *how* it executes. Fusion is an execution strategy, not
an algebraic identity. Mixing them destroys the abstraction boundary that
allows backends to be swapped without changing semantics.

## 19.2 What Fusion May Not Do

A fused kernel is semantically correct only if it produces bitwise-identical
results to the sequential unfused kernels, modulo floating-point reassociation
that is explicitly documented. Fusion must never:

- Change the mathematical result of a sequence of operations in a way not
  covered by a documented stability trade-off
- Alter the differentiation semantics — the pullback of a fused kernel must
  equal the composition of the individual pullbacks
- Introduce implicit broadcasting that `tensor/` does not model
- Skip operations that have side-effects on shape or layout
- Violate the deterministic reduction order required for `deterministic_mode`

## 19.3 Expressing Fusion

Fusion opportunities are expressed as pattern matches on sequences of
`Backend` method calls in Layer 2. They are never expressed as new traits
in `algebra/` or new types in `tensor/`:

```rust
// CORRECT — fusion lives entirely in backend/ Layer 2
impl CpuBackend {
    // Fused multiply-add: avoids a round-trip through memory
    // STABILITY: result equivalent to sequential mul then add, same rounding
    fn fused_mul_add<T: Scalar>(
        &self, a: &Storage<T>, b: &Storage<T>, c: &Storage<T>
    ) -> Storage<T> { ... }
}

// WRONG — fusion as a new tensor type or algebra trait
trait Fusable { fn fuse_with(self, other: impl TensorOp) -> FusedOp; }
```

## 19.4 Autodiff Through Fused Kernels

Every fused kernel must have a verified pullback that agrees with the
composed pullback of its constituent operations. This must be tested with
a gradient check using the same relative-error standard as §8.4.

```rust
#[test]
fn grad_check_fused_mul_add() {
    // Verify: ∂(fused_mul_add)/∂a == ∂(a*b + c)/∂a (sequential)
    // Use relative error < 1e-4 — see §8.4
}
```

---

# 20. Shape Algebra Module (`tensor::shape`)

Shape arithmetic in a rank-polymorphic tensor library is not trivial. Implicit
or ad-hoc shape reasoning is a primary source of silent bugs. All shape
operations must live in a single, well-typed module.

## 20.1 Module Purpose

`tensor::shape` is the **sole location** for compile-time shape reasoning.
It encodes the rules for how ranks and dimensions transform under each tensor
operation — contraction, broadcast, transpose, reshape — as types and const
expressions, not as runtime assertions.

Any PR that introduces shape arithmetic outside this module will be rejected.

## 20.2 Shape Encoding

A shape is an array of dimension sizes indexed by rank:

```rust
// Shape is a compile-time value — RANK is a const generic
type Shape<const RANK: usize> = [usize; RANK];

// ShapeEncoding: the trait that algebra/ and tensor/ operate over
pub trait ShapeEncoding {
    const RANK: usize;
    fn dims(&self) -> Shape<{ Self::RANK }>;
}
```

## 20.3 Typed Shape Operations

Each operation on tensors has a corresponding shape-level morphism. The
output shape must be computable from input shapes **at compile time** wherever
rank is statically known.

### Contraction

Contraction over `K` shared indices reduces rank:

```rust
// contract: [A × K] ⊗ [K × B] → [A × B]
// Output rank = RANK_LEFT + RANK_RIGHT - 2 * NUM_CONTRACT_AXES
pub const fn contract_rank(left: usize, right: usize, k: usize) -> usize {
    left + right - 2 * k
}
```

Contracting over axes that do not exist or have mismatched sizes must be a
**compile error**, not a runtime panic. Dimension mismatches that cannot be
checked at compile time must return `Result`, never panic.

### Broadcast

Broadcasting is an explicit, typed morphism — never implicit (see §6.6):

```rust
// broadcast_to: Shape<M> → Shape<N> where N >= M
// The shape difference is explicit in the type signature
pub fn broadcast_to<const M: usize, const N: usize>(
    src: Shape<M>, target: Shape<N>
) -> Result<Shape<N>, BroadcastError>
where [(); N - M]: // N >= M enforced at compile time
{ ... }
```

`BroadcastError` is returned (not panicked) when runtime dimension
compatibility fails (e.g., a size-3 axis cannot broadcast to size-5).

### Transpose

Transpose is a permutation of axes. The rank is preserved; the dimension
order changes:

```rust
// transpose: Shape<N> × Permutation<N> → Shape<N>
// Permutation validity (no repeats, all indices in range) is checked at construction
pub struct Permutation<const N: usize>([usize; N]);

impl<const N: usize> Permutation<N> {
    pub fn new(perm: [usize; N]) -> Result<Self, PermutationError> {
        // validate: all values in 0..N, no repeats
    }
}

pub fn transpose_shape<const N: usize>(
    shape: Shape<N>, perm: &Permutation<N>
) -> Shape<N> { ... }
```

### Reshape

Reshape is valid only when the total number of elements is preserved:

```rust
// reshape: Shape<M> → Shape<N> where product(M dims) == product(N dims)
// Validity is a runtime check because N dims are not statically constrained
pub fn reshape<const M: usize, const N: usize>(
    src: Shape<M>, target: Shape<N>
) -> Result<Shape<N>, ReshapeError> {
    // ReshapeError if element counts differ
}
```

## 20.4 Shape Arithmetic in Tests

Every tensor operation test must assert the output shape explicitly, in
addition to value correctness:

```rust
#[test]
fn contract_output_shape() {
    // [3 × 4] · [4 × 5] → [3 × 5]
    let a: Tensor<f64, 2, _> = ...;
    let b: Tensor<f64, 2, _> = ...;
    let c = contract(&a, &b, 1);
    assert_eq!(c.shape(), [3, 5]);
}
```

---

# 21. Category-Theoretic Structure

Symmetrica's design is grounded in category theory. Making this structure
explicit helps contributors reason about what operations are valid, how
autodiff composes, and where new abstractions should live.

## 21.1 The Category

Symmetrica defines an implicit category **Tens** where:

- **Objects** are vector spaces (types implementing `VectorSpace`)
- **Morphisms** are differentiable maps between vector spaces
  (types implementing `Differentiable<Domain = A, Codomain = B>`)
- **Composition** is function composition, verified associative at the type level
- **Identity** morphisms exist for every object (the identity map)

This is not a metaphor — it is the architectural invariant. Every trait and
type in `algebra/` and `tensor/` must fit this picture. If a proposed abstraction
does not correspond to an object, morphism, or functor in this category,
it does not belong in `algebra/`.

## 21.2 Objects → Vector Spaces

Types implementing `VectorSpace` are the objects of **Tens**:

```rust
// VectorSpace: the objects
// Must satisfy: closure, associativity, commutativity of +,
//               scalar distributivity, identity (zero vector), inverses
pub trait VectorSpace: Module {
    type Scalar: Field;
    fn zero() -> Self;
    fn add(a: Self, b: Self) -> Self;
    fn scale(s: Self::Scalar, v: Self) -> Self;
}
```

Concrete tensors (`Tensor<T, N, B>`) are elements of vector space objects,
not objects themselves. The distinction matters: the space `ℝⁿ` is the
object; a particular vector in `ℝⁿ` is an element.

## 21.3 Morphisms → Differentiable Maps

Morphisms in **Tens** are smooth maps `f: A → B` between vector spaces:

```rust
// Differentiable: the morphisms
// The derivative at a point is a linear map (element of the tangent space)
pub trait Differentiable {
    type Domain:     VectorSpace;
    type Codomain:   VectorSpace;
    // The derivative: a linear map from tangent(Domain) → tangent(Codomain)
    type Derivative: LinearMap<
        Domain   = <Self::Domain as VectorSpace>::Tangent,
        Codomain = <Self::Codomain as VectorSpace>::Tangent,
    >;
    fn apply(&self, x: Self::Domain) -> Self::Codomain;
    fn diff(&self, x: &Self::Domain) -> Self::Derivative;
}
```

Composition of morphisms must be valid in both directions: `f: A → B` and
`g: B → C` compose to `g ∘ f: A → C`. The chain rule then gives
`D(g ∘ f) = Dg · Df` — this is exactly the VJP pullback composition.

## 21.4 Pullbacks → Cotangent Maps

The **cotangent** (dual) of a vector space `V` is `V* = Hom(V, ℝ)`. The
pullback (VJP) of a morphism `f: A → B` is the linear map
`f*: B* → A*` that carries cotangent vectors backward:

```rust
// Pullback: the cotangent functor applied to a morphism
// f* is contravariant: reverses the direction of arrows
pub trait Pullback<W: VectorSpace> {
    type Cotangent: VectorSpace;
    // f*(w) — apply the cotangent map to a cotangent vector w ∈ W*
    fn pullback(&self, cotangent: W) -> Self::Cotangent;
}
```

This is the categorical dual of forward-mode autodiff. Forward mode applies
the **pushforward** (tangent map, covariant); reverse mode applies the
**pullback** (cotangent map, contravariant). Both are morphisms in the
appropriate (co)tangent category.

## 21.5 Functors → Structure-Preserving Maps

Operations that map one algebraic structure to another structurally are
**functors**. Examples in this codebase:

| Functor | Domain category | Codomain category | What it does |
|---|---|---|---|
| `Dual<_>` | **Ring** | **Ring** | Lifts a ring to its dual-number extension |
| `Gradient<_>` | **VectorSpace** | **VectorSpace** | Maps a space to its cotangent space |
| `Checkpoint<_>` | **Differentiable** | **Differentiable** | Adds recompute boundary without changing semantics |
| `StopGrad<_>` | **Differentiable** | **Differentiable** | Sets the derivative to zero morphism |

A functor must preserve composition: `F(g ∘ f) = F(g) ∘ F(f)`. When
implementing a new functor-like wrapper, verify this law in a `proptest` test.

## 21.6 Implications for Contributors

This structure has direct practical implications:

- **New operations** must be expressible as morphisms (differentiable maps)
  or functors. If they cannot be, the algebraic structure is incomplete —
  extend it, don't bypass it.
- **Composition** is always valid when types match. If composition fails to
  typecheck, the types are wrong — fix the types, not the composition.
- **Autodiff correctness** reduces to chain-rule correctness in this category.
  A gradient bug is a morphism composition bug — reason about it categorically.
- **The prohibited list** (§16) items are precisely the things that break
  this categorical structure: runtime tapes replace morphisms with dynamic
  dispatch; implicit broadcasting introduces morphisms that are not
  type-visible; full Hessians mistake a rank-2 tensor for a linear map.

---

# 22. Compile-Time Budget Policy

Generic algebra systems compound compile times aggressively. Monomorphization
is the performance strategy — but unconstrained, it makes the codebase
unmaintainable. This section formalizes the governance rules stated informally
in §6.9.

## 22.1 Regression Threshold

**A compile-time regression of more than 10% on the full workspace build
(`cargo build --workspace`) must be investigated and resolved before merge.**

The 10% threshold applies to:
- Clean builds (`cargo build --workspace`)
- Incremental builds after a change to `algebra/`
- Test compilation (`cargo test --workspace --no-run`)

Regressions between 5% and 10% must be noted in the PR description with
an explanation. Regressions under 5% are advisory.

## 22.2 Measuring Compile Time

Use `cargo build --timings` to identify which crates and generic
instantiations dominate compile time:

```bash
cargo build --workspace --timings
# Generates target/cargo-timings/cargo-timing.html
# Inspect for codegen units that dominate wall time
```

For fine-grained monomorphization analysis:
```bash
RUSTFLAGS="-Z print-mono-items=eager" cargo +nightly build 2>&1 | grep "MONO_ITEM"
```

Before submitting any PR that adds new generic parameters or blanket impls,
run both and confirm no significant regressions.

## 22.3 Rules That Govern Compile Time

These are binding rules, not guidelines:

1. **Blanket impls over unconstrained generics are prohibited.** Every blanket
   impl must have at least one bound that limits the set of types it applies to.

2. **Sealed traits are required for all core `algebra/` traits.** Unsealed
   traits allow downstream impls that the compiler cannot prune, increasing
   monomorphization surface.

3. **Nested `Dual` types beyond second order require design review.**
   `Dual<Dual<T>>` (hyper-dual) is permitted for exact second derivatives.
   `Dual<Dual<Dual<T>>>` requires an Issue tagged `compile-budget` before use.

4. **New const generic parameters require justification.** Each additional
   const generic parameter multiplies the monomorphization space. Document
   the parameter's purpose and why a runtime value is insufficient.

5. **Compile-time regression PRs must include a fix, not just a note.**
   A PR that regresses compile time by >10% will not be merged until the
   regression is resolved, regardless of functional correctness.

## 22.4 Known Expensive Patterns

| Pattern | Risk | Mitigation |
|---|---|---|
| `impl<T: Field> Foo for Bar<T>` (no other bounds) | High — instantiates for every `T` | Add bounding traits; seal the trait |
| Deeply nested associated types (`A::B::C::D`) | Medium — forces solver work at every use site | Introduce intermediate type aliases |
| `where [(); RANK_A + RANK_B]:` const bounds | Medium — combinatorial on rank values | Limit to ops where rank arithmetic is genuinely needed |
| `Dual<Dual<T>>` in non-hyper-dual contexts | High — doubles monomorphization depth | Restrict to §7.4 hyper-dual use cases only |

---

# 23. Benchmarking Policy

`backend/` has no formal benchmark suite at the time of this writing. This
section establishes the policy that new kernel work must follow.

## 23.1 Criterion Benchmarks Are Mandatory for Backend Kernels

Every non-trivial kernel added to `backend/` Layer 2 must have a corresponding
`criterion` benchmark before the PR is merged. "Non-trivial" means any kernel
that:
- Performs a BLAS call
- Uses SIMD intrinsics
- Implements a reduction (sum, max, norm, etc.)
- Is a fused operation (see §19)

Benchmarks live in `backend/benches/`. Each benchmark file corresponds to a
kernel family (e.g., `benches/matmul.rs`, `benches/reduce.rs`).

## 23.2 Benchmark Structure

Use `criterion`'s standard group structure. Always bench across multiple
input sizes — a kernel that is fast at small sizes and slow at large sizes
(or vice versa) must have both visible:

```rust
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");

    for size in [64usize, 256, 1024, 4096] {
        group.bench_with_input(
            BenchmarkId::new("cpu_f64", size),
            &size,
            |b, &n| {
                let a = CpuBackend::alloc_random::<f64>([n, n]);
                let b_mat = CpuBackend::alloc_random::<f64>([n, n]);
                b.iter(|| black_box(CpuBackend::matmul(&a, &b_mat)))
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_matmul);
criterion_main!(benches);
```

## 23.3 Performance Claims Require Benchmark Evidence

Do not make performance claims in PR descriptions, doc comments, or
`// STABILITY:` annotations without a benchmark result to support them.
"This is faster because SIMD" is not evidence. A `criterion` output
showing throughput improvement across representative sizes is.

## 23.4 Regression Tracking

Benchmark baselines are committed to `backend/benches/baselines/` using
`cargo criterion --save-baseline <name>`. PRs that modify existing kernels
must run:

```bash
cargo criterion --baseline main --bench <kernel>
```

and include the comparison output in the PR description. A regression of
more than 5% in any benchmarked operation requires justification before merge.

## 23.5 What Benchmarks Do Not Prove

Benchmarks measure throughput on the benchmarking machine. They do not
verify correctness, numerical stability, or behavior on other hardware
(especially regarding SIMD availability, cache sizes, or denormal handling).
Benchmark results accompany tests — they never replace them.

---

# 24. Known Limitations and Active Rough Edges

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

- **No benchmark baselines yet:** `backend/benches/baselines/` is empty.
  The first PR to add a kernel benchmark should also commit the initial
  baseline using `cargo criterion --save-baseline main`.

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

*Last reviewed: 2026-03 — Principal Project Manager / Senior Principal DL Researcher / Principal Architect*  
*Format: AGENTS.md — open standard stewarded by the Agentic AI Foundation / Linux Foundation*  
*Next scheduled review: when any §16 item is reconsidered, or at next major milestone*

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
| `R-001` | **P0** | No crate DAG violation | ✅ grep scoped per crate (see §A.4) | `tensor/` may import `Backend` trait from `backend/` — correct. Must never name a concrete impl type. |
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
| `R-015` | **P2** | Associated type selection follows §6.10 policy (regular vs GAT) | 🔴 Manual only | New lifetime on `Scalar`/`Gradient`/`Shape`/`Backend`/`Derivative` is a violation; missing lifetime on `View`/`Iter`/`Slice` is a violation |
| `R-016` | **P1** | Kernel fusion only in `backend/` Layer 2 | ⚠️ grep for fusion patterns in wrong crates (partial) | Semantic fusion (not named `fuse`) evades grep — human must verify new multi-op types in `algebra/`/`tensor/` |
| `R-017` | **P2** | Shape arithmetic lives only in `tensor::shape` | ⚠️ grep `shape` arithmetic outside `tensor/shape/` (partial) | N/A |
| `R-018` | **P2** | Compile-time regression >10% investigated before merge | ⚠️ `cargo build --timings` comparison | No automated gate — requires contributor to run and report |
| `R-019` | **P2** | New backend kernels have `criterion` benchmark | 🔴 Manual only | N/A |
| `R-020` | **P3** | Commit message follows `<crate>: <description>` format | 🔴 Manual only | Advisory |
| `R-021` | **P3** | Draft PR opened within 1 day of starting work | 🔴 Manual only | Advisory |

**P0 count: 4 rules fully automated.**  
**P1 count: 5 rules require human review — they block merge but cannot be scripted.**  
**P2 count: 9 rules are quality flags — they do not block merge.**  
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
if grep -rn "^use tensor::\|^use backend::" algebra/src/ 2>/dev/null; then
    echo "❌ P0 / R-001: algebra/ imports tensor/ or backend/ — forbidden"
    FAILED=1
fi
if grep -rn "CpuBackend\|CudaBackend\|WasmBackend" tensor/src/ 2>/dev/null; then
    echo "❌ P0 / R-001: tensor/ names a concrete backend type — forbidden"
    echo "   tensor/ must only use B: Backend trait bounds, never concrete structs"
    FAILED=1
fi

# ── P0: R-003 — Undocumented unsafe ───────────────────────────────────────
cargo clippy --workspace -- -D clippy::undocumented_unsafe_blocks 2>&1 \
    || { echo "❌ P0 / R-003: Undocumented unsafe block"; FAILED=1; }

# ── P0: R-007 — Hidden RNG ────────────────────────────────────────────────
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
if grep -rn "Vec<Box<dyn" algebra/src/ tensor/src/ 2>/dev/null; then
    echo "⛔ P1 / R-002: Possible runtime tape — HUMAN REVIEW REQUIRED before merge"
    FAILED=1
fi

# ── P1: R-016 — Fusion outside backend/ (partial) ─────────────────────────
if grep -rn "fuse\|fused_\|FusedOp\|FusionKernel" algebra/src/ tensor/src/ 2>/dev/null; then
    echo "⛔ P1 / R-016: Possible kernel fusion in algebra/ or tensor/ — HUMAN REVIEW REQUIRED"
    echo "   Fusion is permitted only in backend/ Layer 2 (see §19)"
    FAILED=1
fi

# ── Standard CI gates ─────────────────────────────────────────────────────
cargo test --workspace        || { echo "❌ Tests failed"; FAILED=1; }
cargo fmt --check             || { echo "❌ Format check failed"; FAILED=1; }
cargo doc --workspace --no-deps 2>&1 | grep "^error" \
    && { echo "❌ Doc build errors"; FAILED=1; } || true

# ── P2: Advisory flags (do not exit 1) ────────────────────────────────────
echo ""
echo "=== P2 Advisory Flags ==="

# R-010: Gradient check tests
cargo test grad_check --workspace 2>&1 \
    || echo "⚠️  P2 / R-010: grad_check tests missing or failing — add before merge"

# R-012: Stability annotations
if grep -rPn '\b(exp|ln|log|sqrt)\(' algebra/src/ tensor/src/ 2>/dev/null \
    | grep -v "// STABILITY:" | grep -v "exp_m1\|ln_1p"; then
    echo "⚠️  P2 / R-012: Numerically sensitive op without // STABILITY: comment"
fi

# R-013: Non-smooth ops
if grep -rn "\brelu\|\babs\|\bsign\b" algebra/src/ tensor/src/ 2>/dev/null \
    | grep -v "// NON-DIFFERENTIABLE:"; then
    echo "⚠️  P2 / R-013: Non-smooth op without // NON-DIFFERENTIABLE: annotation"
fi

# R-017: Shape arithmetic outside tensor::shape
if grep -rPn '\bshape\b.*[+\-\*]|\brank\b.*[+\-\*]' tensor/src/ 2>/dev/null \
    | grep -v "tensor/src/shape"; then
    echo "⚠️  P2 / R-017: Shape arithmetic outside tensor::shape — consolidate in §20 module"
fi

# R-018: Compile-time budget (advisory reminder — no automated gate)
echo "⚠️  P2 / R-018: Run 'cargo build --workspace --timings' and confirm <10% regression"

# R-019: Criterion benchmarks for new backend kernels (advisory reminder)
echo "⚠️  P2 / R-019: New backend/ kernels require criterion benchmark in backend/benches/"

# ── Final verdict ─────────────────────────────────────────────────────────
echo ""
if [ $FAILED -eq 1 ]; then
    echo "❌ P0/P1 violations found — merge blocked."
    echo "   P1 rules R-004 (stop-gradient), R-005 (broadcasting), R-006 (Hessian)"
    echo "   cannot be scripted — verify manually against §16 Prohibited List."
    exit 1
fi

echo "✅ P0 rules passed. P2/P3 flags above are advisory."
echo "   Manually verify P1 rules: R-002, R-004, R-005, R-006, R-016."
```

---

## A.3 Known Detection Gaps

| Rule | Gap | What Escapes | Human Mitigation |
|---|---|---|---|
| `R-001` | Concrete type in `tensor/` | `CpuBackend` named anywhere in `tensor/src/` | `grep -rn "CpuBackend\|CudaBackend" tensor/src/` — any hit is a violation |
| `R-001` | Aliased backend import | `use b = backend; b::CpuBackend` in `tensor/src/` | `tensor/` may depend on `backend/` for the trait; verify no concrete structs are used |
| `R-001` | Any workspace import in `algebra/` | `tensor::` or `backend::` in `algebra/src/` | Audit `algebra/Cargo.toml` — must list no symmetrica workspace crates |
| `R-002` | Renamed types | `struct Graph { ops: Vec<Box<dyn Backward>> }` | Review all new `struct` definitions in `algebra/`/`tensor/` |
| `R-007` | Method-only RNG | `thread_rng().gen::<f64>()` without `use rand::` | Flag any call containing `rng`, `random`, `gen` in `algebra/`/`tensor/` |
| `R-012` | False positives | `explanation`, `expect`, `export` contain `exp` | Script uses `\bexp\(` word-boundary — review flagged lines |
| `R-016` | Semantic fusion | Multi-op composite type not named `fuse*` | Review any new struct in `algebra/`/`tensor/` that wraps two or more operation types |
| `R-017` | Shape logic in op impls | Inline `a_rank + b_rank - 2*k` in `contract()` body | Any shape arithmetic outside `tensor/src/shape/` is a violation — search for arithmetic on `rank`/`shape` variables |

---

## A.4 Priority Quick-Reference for Agents

When encountering a rule violation, apply this decision tree:

```
Is the violation detected by the CI script?
├── YES → CI will block the PR automatically (P0). Fix before pushing.
└── NO  → Is it in the P1 list (R-002, R-004, R-005, R-006, R-016)?
           ├── YES → Do NOT proceed. Open a GitHub Issue tagged `architecture`.
           │         A human architect must review before merge.
           └── NO  → Is it in the P2 list (R-009 through R-019)?
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
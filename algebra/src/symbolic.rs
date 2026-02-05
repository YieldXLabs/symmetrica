use core::fmt::Debug;
use core::marker::PhantomData;
use core::usize;

// Core Types
#[derive(Debug, Clone, Copy)]
pub struct Nil;

#[derive(Debug, Clone, Copy)]
pub struct Cons<H, T>(PhantomData<(H, T)>);

// TODO: Derive Macro.
// Manually implementing `Label` and `TypeEq` is tedious.
// Create `#[derive(Label)]` to automatically generate the struct,
// the Label impl, and the TypeEq negative bounds.
pub trait Label: 'static + Copy + Clone + Debug + Send + Sync {
    fn name() -> &'static str;
}

// Shapes
pub trait Shape {
    const RANK: usize;
    type Axes;
}

impl Shape for Nil {
    const RANK: usize = 0;
    type Axes = Nil;
}

impl<H: Label, T: Shape> Shape for Cons<H, T> {
    const RANK: usize = 1 + T::RANK;
    type Axes = Cons<H, T>;
}

// Dynamic Fallback
#[derive(Debug, Clone, Copy)]
pub struct DynRank<const N: usize>;

impl<const N: usize> Shape for DynRank<N> {
    const RANK: usize = N;
    type Axes = Nil;
}

// Logic engine (internal)
pub struct True;
pub struct False;

pub trait Bool: 'static {
    const VALUE: bool;
}

impl Bool for True {
    const VALUE: bool = true;
}

impl Bool for False {
    const VALUE: bool = false;
}

pub trait Or<Rhs> {
    type Result;
}
impl Or<True> for True {
    type Result = True;
}
impl Or<False> for True {
    type Result = True;
}
impl Or<True> for False {
    type Result = True;
}
impl Or<False> for False {
    type Result = False;
}

pub trait And<Rhs: Bool> {
    type Result: Bool;
}

impl<Rhs: Bool> And<Rhs> for True {
    type Result = Rhs;
}

impl<Rhs: Bool> And<Rhs> for False {
    type Result = False;
}

pub type IfThenElse<Cond, T, F> = <Cond as SelectType<T, F>>::Result;

pub trait SelectType<T, F> {
    type Result;
}
impl<T, F> SelectType<T, F> for True {
    type Result = T;
}
impl<T, F> SelectType<T, F> for False {
    type Result = F;
}

pub trait TypeEq<Other> {
    type Result;
}
impl<T> TypeEq<T> for T {
    type Result = True;
}

// Search engine for IndexOf
pub trait IndexOfFinder<Target, MatchResult> {
    const VALUE: usize;
}

// Case: Match Found
impl<Target, Head, Tail> IndexOfFinder<Target, True> for Cons<Head, Tail> {
    const VALUE: usize = 0;
}

// Case: No Match (Recurse)
impl<Target, Head, Tail> IndexOfFinder<Target, False> for Cons<Head, Tail>
where
    Tail: IndexOf<Target>,
{
    const VALUE: usize = 1 + Tail::INDEX;
}

pub trait IndexOf<Target> {
    const INDEX: usize;
}

impl<Target, Head, Tail> IndexOf<Target> for Cons<Head, Tail>
where
    Target: Label,
    Head: TypeEq<Target>,
    Cons<Head, Tail>: IndexOfFinder<Target, <Head as TypeEq<Target>>::Result>,
{
    const INDEX: usize =
        <Cons<Head, Tail> as IndexOfFinder<Target, <Head as TypeEq<Target>>::Result>>::VALUE;
}

// Remove
pub trait RemoveFinder<Target, MatchResult> {
    type Remainder: Shape;
}

impl<Target, Head, Tail> RemoveFinder<Target, True> for Cons<Head, Tail>
where
    Tail: Shape,
{
    type Remainder = Tail;
}

impl<Target, Head, Tail> RemoveFinder<Target, False> for Cons<Head, Tail>
where
    Head: Label,
    Tail: Remove<Target>,
{
    type Remainder = Cons<Head, <Tail as Remove<Target>>::Remainder>;
}

pub trait Remove<Target> {
    type Remainder: Shape;
}

impl<Target, Head, Tail> Remove<Target> for Cons<Head, Tail>
where
    Target: Label,
    Head: TypeEq<Target>,
    Cons<Head, Tail>: RemoveFinder<Target, <Head as TypeEq<Target>>::Result>,
{
    type Remainder =
        <Cons<Head, Tail> as RemoveFinder<Target, <Head as TypeEq<Target>>::Result>>::Remainder;
}

// Contains
// e.g. const { assert!(Sh::DOES_CONTAIN, "Label not found in Shape!") };
pub trait Contains<Target> {
    type Result;
}

impl<Target> Contains<Target> for Nil {
    type Result = False;
}

impl<Target, Head, Tail> Contains<Target> for Cons<Head, Tail>
where
    Head: TypeEq<Target>,
    Tail: Contains<Target>,
    <Head as TypeEq<Target>>::Result: Or<<Tail as Contains<Target>>::Result>,
{
    type Result =
        <<Head as TypeEq<Target>>::Result as Or<<Tail as Contains<Target>>::Result>>::Result;
}
// Union
pub trait AddUnique<L>: Shape {
    type Output: Shape;
}

pub trait AddUniqueImpl<L, Flag: Bool> {
    type Output: Shape;
}

impl<L, List> AddUniqueImpl<L, True> for List
where
    List: Shape,
{
    type Output = List;
}

impl<L: Label, List> AddUniqueImpl<L, False> for List
where
    List: Shape,
{
    type Output = Cons<L, List>;
}

pub trait Union<Rhs: Shape> {
    type Output: Shape;
}

impl<Rhs: Shape> Union<Rhs> for Nil {
    type Output = Rhs;
}

impl<H, T, Rhs: Shape> Union<Rhs> for Cons<H, T>
where
    T: Shape + Union<Rhs>,
    <T as Union<Rhs>>::Output: AddUnique<H>,
{
    type Output = <<T as Union<Rhs>>::Output as AddUnique<H>>::Output;
}

impl<const N: usize> Union<DynRank<N>> for DynRank<N> {
    type Output = DynRank<N>;
}

pub trait Permutation<Dst: Shape> {
    fn indices() -> Vec<usize>;
}

impl<Src: Shape> Permutation<Nil> for Src {
    fn indices() -> Vec<usize> {
        Vec::new()
    }
}

// TODO: Make it faster
impl<Src, Head, Tail> Permutation<Cons<Head, Tail>> for Src
where
    Src: Shape + IndexOf<Head>,
    Head: Label,
    Tail: Shape,
    Src: Permutation<Tail>,
{
    fn indices() -> Vec<usize> {
        let current = <Src as IndexOf<Head>>::INDEX;
        let mut rest = <Src as Permutation<Tail>>::indices();
        rest.insert(0, current); // O(N) 
        rest
    }
}

pub trait SubsetOf<Super: Shape> {
    type Result: Bool;
}

impl<Super: Shape> SubsetOf<Super> for Nil {
    type Result = True;
}

impl<H, T, Super> SubsetOf<Super> for Cons<H, T>
where
    Super: Shape + Contains<H>,
    T: SubsetOf<Super>,
    <Super as Contains<H>>::Result: And<<T as SubsetOf<Super>>::Result>,
{
    type Result = <<Super as Contains<H>>::Result as And<<T as SubsetOf<Super>>::Result>>::Result;
}

impl<const N: usize> SubsetOf<DynRank<N>> for DynRank<N> {
    type Result = True;
}

pub trait BroadcastShape<Other: Shape> {
    type Output: Shape;
}

impl<A, B: Shape> BroadcastShape<B> for A
where
    A: Union<B>,
{
    type Output = <A as Union<B>>::Output;
}

// Maps indices from Src to Dst
pub trait BroadcastMap<Dst: Shape> {
    fn mapping() -> Vec<Option<usize>>;
}

impl<Src: Shape> BroadcastMap<Nil> for Src {
    fn mapping() -> Vec<Option<usize>> {
        Vec::new()
    }
}

impl<Src, Head: Label, Tail: Shape> BroadcastMap<Cons<Head, Tail>> for Src
where
    Src: Shape + Contains<Head>,
    Src: BroadcastMap<Tail>,
    Cons<Head, Tail>: BroadcastEntryFinder<Src, <Src as Contains<Head>>::Result>,
{
    fn mapping() -> Vec<Option<usize>> {
        <Cons<Head, Tail> as BroadcastEntryFinder<Src, <Src as Contains<Head>>::Result>>::entry()
    }
}

impl<const N: usize> BroadcastMap<DynRank<N>> for DynRank<N> {
    fn mapping() -> Vec<Option<usize>> {
        (0..N).map(|i| Some(i)).collect()
    }
}

pub trait BroadcastEntryFinder<Src, ContainsResult> {
    fn entry() -> Vec<Option<usize>>;
}

impl<Src, Head, Tail: Shape> BroadcastEntryFinder<Src, True> for Cons<Head, Tail>
where
    Src: Shape + IndexOf<Head>,
    Src: BroadcastMap<Tail>,
    Head: Label,
{
    fn entry() -> Vec<Option<usize>> {
        let index = <Src as IndexOf<Head>>::INDEX;
        let mut rest = <Src as BroadcastMap<Tail>>::mapping();
        rest.insert(0, Some(index));
        rest
    }
}

impl<Src: Shape, Head, Tail: Shape> BroadcastEntryFinder<Src, False> for Cons<Head, Tail>
where
    Src: BroadcastMap<Tail>,
    Head: Label,
{
    fn entry() -> Vec<Option<usize>> {
        let mut rest = <Src as BroadcastMap<Tail>>::mapping();
        rest.insert(0, None);
        rest
    }
}

pub trait CanBroadcastWith<Other: Shape> {
    type Result: Bool;
}

impl<A, B> CanBroadcastWith<B> for A
where
    A: Shape,
    B: Shape,
    A: Union<B>,
    A: SubsetOf<<A as Union<B>>::Output>,
    B: SubsetOf<<A as Union<B>>::Output>,
    <A as SubsetOf<<A as Union<B>>::Output>>::Result:
        And<<B as SubsetOf<<A as Union<B>>::Output>>::Result>,
{
    type Result = <<A as SubsetOf<<A as Union<B>>::Output>>::Result as And<
        <B as SubsetOf<<A as Union<B>>::Output>>::Result,
    >>::Result;
}

// TODO: einsum
/// Defines how to contract L and R along Axis
// pub trait Contract<L: Shape, R: Shape, Axis: Label> {
//     /// Resulting shape after contraction
//     type Output: Shape;

//     /// (axis index in L, axis index in R)
//     const AXES: (usize, usize);
// }

// pub struct DefaultContract;

// impl<L, R, Axis> Contract<L, R, Axis> for DefaultContract
// where
//     L: Shape + IndexOf<Axis> + Remove<Axis>,
//     R: Shape + IndexOf<Axis> + Remove<Axis>,
//     Axis: Label,
//     <L as Remove<Axis>>::Remainder: Union<
//         <R as Remove<Axis>>::Remainder
//     >,
// {
//     type Output =
//         <<L as Remove<Axis>>::Remainder as Union<
//             <R as Remove<Axis>>::Remainder
//         >>::Output;

//     const AXES: (usize, usize) = (
//         <L as IndexOf<Axis>>::INDEX,
//         <R as IndexOf<Axis>>::INDEX,
//     );
// }

// pub trait SharedAxes<Rhs: Shape> {
//     type Axes: Shape;
// }

// impl<Rhs: Shape> SharedAxes<Rhs> for Nil {
//     type Axes = Nil;
// }

// impl<H, T, Rhs> SharedAxes<Rhs> for Cons<H, T>
// where
//     H: Label,
//     Rhs: Contains<H>,
//     T: SharedAxes<Rhs>,
//     <Rhs as Contains<H>>::Result: Bool,
// {
//     type Axes = IfThenElse<
//         <Rhs as Contains<H>>::Result,
//         Cons<H, <T as SharedAxes<Rhs>>::Axes>,
//         <T as SharedAxes<Rhs>>::Axes,
//     >;
// }

// pub trait MultiContract<L: Shape, R: Shape, Axes: Shape> {
//     type Output: Shape;
// }

// impl<L: Shape, R: Shape> MultiContract<L, R, Nil> for () {
//     type Output = <L as Union<R>>::Output;
// }

// impl<L, R, H, T> MultiContract<L, R, Cons<H, T>> for ()
// where
//     (): Contract<L, R, H>,
//     (): MultiContract<
//         <() as Contract<L, R, H>>::Output,
//         Nil,
//         T
//     >,
// {
//     type Output =
//         <() as MultiContract<
//             <() as Contract<L, R, H>>::Output,
//             Nil,
//             T
//         >>::Output;
// }

// #[macro_export]
// macro_rules! einsum {
//     (
//         ($($a_axes:ty),+),
//         ($($b_axes:ty),+)
//         ->
//         ($($out_axes:ty),+);
//         $a:expr, $b:expr
//     ) => {{
//         type ASh = Axes![$($a_axes),+];
//         type BSh = Axes![$($b_axes),+];
//         type OutSh = Axes![$($out_axes),+];

//         let a1 = $a.align_to::<ASh>();
//         let b1 = $b.align_to::<BSh>();

//         // Infer shared axes at type level
//         type Shared = <ASh as SharedAxes<BSh>>::Axes;

//         let tmp = a1.contract_all::<Shared>(b1);

//         tmp.align_to::<OutSh>()
//     }};
// }

// TODO: Masked - like causal mask
// The Type Wrapper: "This axis has a validity mask attached"
// #[derive(Debug, Clone, Copy)]
// pub struct Masked<L>(PhantomData<L>);

// // It behaves like a Label
// impl<L: Label> Label for Masked<L> {
//     fn name() -> &'static str { "Masked" } // Simplified for static context
// }

// pub trait Replace<Target: Label, Replacement: Label> {
//     type Output: Shape;
// }

// impl<Target, Rep, Tail> Replace<Target, Rep> for Cons<Target, Tail>
// where
//     Tail: Shape,
//     Rep: Label,
// {
//     // Found it! Swap Head.
//     type Output = Cons<Rep, Tail>;
// }

// impl<Target, Rep, Head, Tail> Replace<Target, Rep> for Cons<Head, Tail>
// where
//     Head: Label + TypeEq<Target, Result = False>,
//     Tail: Replace<Target, Rep>,
// {
//     type Output = Cons<Head, <Tail as Replace<Target, Rep>>::Output>;
// }

#[doc(hidden)]
#[macro_export]
macro_rules! __generate_inequality {
    () => {};
    ($head:ident, $($tail:ident),*) => {
        $(
            impl $crate::symbolic::TypeEq<$tail> for $head { type Result = $crate::symbolic::False; }
            impl $crate::symbolic::TypeEq<$head> for $tail { type Result = $crate::symbolic::False; }
        )*
        $crate::__generate_inequality!($($tail),*);
    };
}

#[macro_export]
macro_rules! make_labels {
    ($($name:ident),*) => {
        $(
            #[derive(Debug, Clone, Copy)]
            pub struct $name;
            impl $crate::symbolic::Label for $name {
                fn name() -> &'static str { stringify!($name) }
            }
        )*
        $crate::__generate_inequality!($($name),*);
    };
}

#[macro_export]
macro_rules! Axes {
    () => { $crate::symbolic::Nil };
    ($head:ty $(, $tail:ty)* $(,)?) => {
        $crate::symbolic::Cons<$head, Axes!($($tail),*)>
    };
}

// TODO: Add #[derive(Label)] macro
// TODO: Dim<L, N> to combine names with sizes

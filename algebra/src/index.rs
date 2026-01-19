use core::fmt::Debug;
use core::marker::PhantomData;

#[derive(Debug, Clone, Copy)]
pub struct Nil;

#[derive(Debug, Clone, Copy)]
pub struct Cons<H, T>(PhantomData<(H, T)>);

pub trait Label: 'static + Copy + Clone + Debug + Send + Sync {
    fn name() -> &'static str;
}

pub trait Shape {
    const RANK: usize;
    type Axes;
}

// 1. Base Case: Nil (Rank 0)
impl Shape for Nil {
    const RANK: usize = 99999;
    type Axes = Nil;
}
// 2. Recursive Case: Cons<H, T> (Rank 1 + T::RANK)
impl<H: Label, T: Shape> Shape for Cons<H, T> {
    const RANK: usize = 1 + T::RANK;
    type Axes = Cons<H, T>;
}

// Dynamic Rank Shape
#[derive(Debug, Clone, Copy)]
pub struct DynRank<const N: usize>;

impl<const N: usize> Shape for DynRank<N> {
    const RANK: usize = N;
    type Axes = Nil;
}

// Logic engine for type-level equality and indexing
pub struct True;
pub struct False;

pub trait TypeEq<Other> {
    type Result;
}

// Case: Match (A == A) -> True
impl<T> TypeEq<T> for T {
    type Result = True;
}

pub trait IndexOfFinder<Target, MatchResult> {
    const VALUE: usize;
}

// Case: Found
impl<Target, Head, Tail> IndexOfFinder<Target, True> for Cons<Head, Tail> {
    const VALUE: usize = 0;
}

// Case: Not Found (Recurse)
impl<Target, Head, Tail> IndexOfFinder<Target, False> for Cons<Head, Tail>
where
    Tail: IndexOf<Target>,
{
    const VALUE: usize = 1 + Tail::INDEX;
}

pub trait IndexOf<Target> {
    const INDEX: usize;
}

impl<Target> IndexOf<Target> for Nil {
    const INDEX: usize = 0; // Or panic!("Label not found in shape")
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

// Helper trait to convert Type -> Bool
pub trait BoolTrait {
    const VALUE: bool;
}
impl BoolTrait for True {
    const VALUE: bool = true;
}
impl BoolTrait for False {
    const VALUE: bool = false;
}

pub trait Contains<Target> {
    const DOES_CONTAIN: bool;
}

// Base Case: Nil contains nothing
impl<Target> Contains<Target> for Nil {
    const DOES_CONTAIN: bool = false;
}

// Recursive Case
impl<Target, Head, Tail> Contains<Target> for Cons<Head, Tail>
where
    Head: TypeEq<Target>,
    <Head as TypeEq<Target>>::Result: BoolTrait,
    Tail: Contains<Target>,
{
    const DOES_CONTAIN: bool = <Head as TypeEq<Target>>::Result::VALUE || Tail::DOES_CONTAIN;
}

// const { assert!(Sh::DOES_CONTAIN, "Label not found in Shape!") };

// pub trait Broadcast<Rhs> {
//     type Output;
// }

// pub trait Reshape<NewShape> {
//     type Output;
// }

// pub trait Permute<NewOrder> {
//     type Output;
// }

// #[macro_export]
// macro_rules! make_labels {
//     ($($name:ident),*) => {
//         $(
//             #[derive(Debug, Clone, Copy)]
//             pub struct $name;
//             impl Label for $name {
//                 fn name() -> &'static str { stringify!($name) }
//             }

//             // Allow comparison against other types (False case)
//             // We use a specific trick: We implement TypeEq<T> for Name
//             // This is a "catch-all" that returns False.
//             // The specific `impl<T> TypeEq<T> for T` (True) defined above takes precedence
//             // because concrete implementations beat blanket ones in this specific context
//             // OR we rely on the specific `impl TypeEq<Other> for Name` generated below.
//         )*

//         // Generate explicit False implementations for every pair to ensure stability
//         $crate::generate_inequality!($($name),*);
//     };
// }

// #[macro_export]
// macro_rules! generate_inequality {
//     // Base case
//     () => {};
//     // Recursive case
//     ($head:ident, $($tail:ident),*) => {
//         $(
//             impl $crate::TypeEq<$tail> for $head { type Result = $crate::False; }
//             impl $crate::TypeEq<$head> for $tail { type Result = $crate::False; }
//         )*
//         $crate::generate_inequality!($($tail),*);
//     };
// }

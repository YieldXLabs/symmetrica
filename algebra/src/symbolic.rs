use core::fmt::Debug;
use core::marker::PhantomData;

// Core Types
#[derive(Debug, Clone, Copy)]
pub struct Nil;

#[derive(Debug, Clone, Copy)]
pub struct Cons<H, T>(PhantomData<(H, T)>);

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

pub trait BoolTrait {
    const VALUE: bool;
}
impl BoolTrait for True {
    const VALUE: bool = true;
}
impl BoolTrait for False {
    const VALUE: bool = false;
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
    const DOES_CONTAIN: bool;
}

impl<Target> Contains<Target> for Nil {
    const DOES_CONTAIN: bool = false;
}

impl<Target, Head, Tail> Contains<Target> for Cons<Head, Tail>
where
    Head: TypeEq<Target>,
    <Head as TypeEq<Target>>::Result: BoolTrait,
    Tail: Contains<Target>,
{
    const DOES_CONTAIN: bool = <Head as TypeEq<Target>>::Result::VALUE || Tail::DOES_CONTAIN;
}

pub trait BroadcastableTo<NewSh: Shape> {
    type Output: Shape;
}

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

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

pub trait RankDiff<Rhs: Shape> {
    const VALUE: usize;
}

impl<Lhs: Shape, Rhs: Shape> RankDiff<Rhs> for Lhs {
    const VALUE: usize = Lhs::RANK.saturating_sub(Rhs::RANK);
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

pub struct IfThenElse<Cond: Bool, Then, Else>(PhantomData<(Cond, Then, Else)>);

impl<Cond: Bool, Then: Bool, Else: Bool> Bool for IfThenElse<Cond, Then, Else> {
    const VALUE: bool = if Cond::VALUE {
        Then::VALUE
    } else {
        Else::VALUE
    };
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

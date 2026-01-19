use core::fmt::Debug;
use core::marker::PhantomData;

#[derive(Debug, Clone, Copy)]
pub struct Nil;

#[derive(Debug, Clone, Copy)]
pub struct Cons<H, T>(PhantomData<(H, T)>);

pub trait Label: 'static + Copy + Clone + Debug + Send + Sync {
    fn name() -> &'static str;
}

// TODO: DynShape struct that implements your Shape trait but holds Vec<usize> at runtime.
pub trait Shape {
    const RANK: usize;
    type Axes;
}

// 1. Base Case: Nil (Rank 0)
impl Shape for Nil {
    const RANK: usize = 0;
    type Axes = Nil;
}

// 2. Recursive Case: Cons (Rank 1 + Tail)
// We capture the Head label and the Tail shape
impl<H: Label, T: Shape> Shape for Cons<H, T> {
    const RANK: usize = 1 + T::RANK;
    type Axes = Cons<H, T>;
}

#[derive(Debug, Clone, Copy)]
pub struct DynRank<const N: usize>;

impl<const N: usize> Shape for DynRank<N> {
    const RANK: usize = N;
    type Axes = ();
}

// pub struct True;
// pub struct False;

// pub trait TypeEq<Other> {
//     type Result;
// }

// impl<T> TypeEq<T> for T {
//     type Result = True;
// }

// pub trait IndexOf<Target: Label> {
//     const INDEX: usize;
// }

// pub trait IndexOfFinder<Target, IsMatch> {
//     const VALUE: usize;
// }

// // Case A: Match Found! (Index is 0)
// impl<Target, Head, Tail> IndexOfFinder<Target, True> for Cons<Head, Tail> {
//     const VALUE: usize = 0;
// }

// // Case B: No Match, Recurse! (Index is 1 + Tail search)
// impl<Target, Head, Tail> IndexOfFinder<Target, False> for Cons<Head, Tail>
// where
//     Tail: IndexOf<Target>, // Recursive Step
// {
//     const VALUE: usize = 1 + Tail::INDEX;
// }

// // Main Entry Point
// impl<Target, Head, Tail> IndexOf<Target> for Cons<Head, Tail>
// where
//     Target: Label,
//     Head: Label + TypeEq<Target>, // Check equality
//     Cons<Head, Tail>: IndexOfFinder<Target, <Head as TypeEq<Target>>::Result>,
// {
//     const INDEX: usize = <Cons<Head, Tail> as IndexOfFinder<Target, <Head as TypeEq<Target>>::Result>>::VALUE;
// }


// We can implement a conversion to panic if users try to use symbolic ops on DynRank
// impl<const N: usize> IndexOf<Batch> for DynRank<N> {
//     const INDEX: usize = 0; // Fallback or Compile Error
// }

// pub trait Contains<A> {}

// impl<A, T> Contains<A> for Cons<A, T> {}
// impl<A, H, T> Contains<A> for Cons<H, T>
// where
//     T: Contains<A>,
// {}

// pub trait IndexOf<A>: Contains<A> {
//     const INDEX: usize;
// }

// impl<A, T> IndexOf<A> for Cons<A, T> {
//     const INDEX: usize = 0;
// }

// impl<A, H, T> IndexOf<A> for Cons<H, T>
// where
//     T: IndexOf<A>,
// {
//     const INDEX: usize = 1 + T::INDEX;
// }

// impl Shape for Nil {
//     const RANK: usize = 0;
//     type Axes = Nil;
// }

// impl<H: Label, T: Shape> Shape for Cons<H, T> {
//     const RANK: usize = 1 + T::RANK;
//     type Axes = Cons<H, T>;
// }

// pub trait AxisValue<A: Label> {
//     fn index(self) -> usize;
// }

// impl<A: Label, H: Label, T> IndexOf<A> for Cons<H, T>
// where
//     T: IndexOf<A>,
// {
//     const POS: usize = 1 + T::POS;
// }

// impl<A: Label, B: Label> IndexOf<B> for (A, B) {
//     const POS: usize = 1;
// }

// // Batch processing
// pub type BatchVector<A, Batch> = (Batch, A);  // Batch × Features
// pub type BatchMatrix<A, B, Batch> = (Batch, A, B);  // Batch × Rows × Cols

// pub trait AxisOf<L: Label>: Shape {
//     const INDEX: usize;
//     type Remainder: Shape;
// }

// impl<A: Label> AxisOf<A> for (A,) {
//     const INDEX: usize = 0;
//     type Remainder = ();
// }

// Symbolic Layer: The Architect. It draws the blueprints, checks the physics, and deletes redundant rooms (optimizations) before construction begins.
// Host/Device Layer: The Construction Crew. They blindly follow the blueprint. Because the Architect (Symbolic Layer) did its job, the crew never tries to fit a square peg in a round hole (Shape Mismatch) or build a wall twice (Redundant Computation).

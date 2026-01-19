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

use std::fmt::Debug;
use std::marker::PhantomData;

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

impl Shape for Nil {
    const RANK: usize = 0;
    type Axes = Nil;
}

impl<H: Label, T: Shape> Shape for Cons<H, T> {
    const RANK: usize = 1 + T::RANK;
    type Axes = Cons<H, T>;
}

pub trait IndexOf<A: Label> {
    const POS: usize;
}

impl<A: Label, T> IndexOf<A> for Cons<A, T> {
    const POS: usize = 0;
}

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

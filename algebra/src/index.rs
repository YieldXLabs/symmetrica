use std::fmt::Debug;

pub trait Label: 'static + Copy + Clone + Debug + Send + Sync {
    fn name() -> &'static str;
}

pub trait Shape: 'static + Copy + Clone + Debug + Send + Sync {
    const RANK: usize;
    type Indices;
}

pub type ScalarShape = ();
pub type VectorShape<A> = (A,);
pub type MatrixShape<A, B> = (A, B);
pub type CubeShape<A, B, C> = (A, B, C);
pub type TensorShape<A, B, C, D> = (A, B, C, D);

impl Shape for ScalarShape {
    const RANK: usize = 0;
    type Indices = ();
}

impl<A: Label> Shape for VectorShape<A> {
    const RANK: usize = 1;
    type Indices = (A,);
}

impl<A: Label, B: Label> Shape for MatrixShape<A, B> {
    const RANK: usize = 2;
    type Indices = (A, B);
}

impl<A: Label, B: Label, C: Label> Shape for CubeShape<A, B, C> {
    const RANK: usize = 3;
    type Indices = (A, B, C);
}

impl<A: Label, B: Label, C: Label, D: Label> Shape for TensorShape<A, B, C, D> {
    const RANK: usize = 4;
    type Indices = (A, B, C, D);
}

#[derive(Debug, Clone, Copy)]
pub struct Untyped;
impl Label for Untyped {
    fn name() -> &'static str {
        "Untyped"
    }
}

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

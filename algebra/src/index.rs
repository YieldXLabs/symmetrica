use std::fmt::Debug;

pub trait Label: 'static + Copy + Clone + Debug + Send + Sync {
    fn name() -> &'static str;
}

pub trait Shape: 'static + Copy + Clone + Debug + Send + Sync {
    const RANK: usize;
    type Indices;
}

// Scalar
impl Shape for () {
    const RANK: usize = 0;
    type Indices = ();
}

// Vector
impl<A: Label> Shape for (A,) {
    const RANK: usize = 1;
    type Indices = (A,);
}

// Matrix
impl<A: Label, B: Label> Shape for (A, B) {
    const RANK: usize = 2;
    type Indices = (A, B);
}

// Rank-3
impl<A: Label, B: Label, C: Label> Shape for (A, B, C) {
    const RANK: usize = 3;
    type Indices = (A, B, C);
}

// Rank-4
impl<A: Label, B: Label, C: Label, D: Label> Shape for (A, B, C, D) {
    const RANK: usize = 4;
    type Indices = (A, B, C, D);
}

pub trait AxisOf<L: Label>: Shape {
    const INDEX: usize;
    type Remainder: Shape;
}

impl<A: Label> AxisOf<A> for (A,) {
    const INDEX: usize = 0;
    type Remainder = ();
}

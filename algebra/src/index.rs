use std::fmt::Debug;

pub trait Label: 'static + Copy + Clone + Debug + Send + Sync {
    fn name() -> &'static str;
}

pub trait Shape: 'static + Copy + Clone + Debug + Send + Sync {
    const RANK: usize;
}

impl Shape for () {
    const RANK: usize = 0;
} // Scalar
impl<A: Label> Shape for (A,) {
    const RANK: usize = 1;
} // Vector
impl<A: Label, B: Label> Shape for (A, B) {
    const RANK: usize = 2;
} // Matrix

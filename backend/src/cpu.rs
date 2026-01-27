use std::marker::PhantomData;

use super::{
    Backend, BinaryKernel, ReduceKernel, Storage, StreamKernel, UnaryKernel, UnifiedStorage,
};
use algebra::{Data, Semiring};

#[derive(Debug, Clone, Copy)]
pub struct GenericBackend<F: Data> {
    _marker: PhantomData<F>,
}

impl<F: Data> GenericBackend<F> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<F: Data> Default for GenericBackend<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Data> Backend<F> for GenericBackend<F> {
    type Repr = UnifiedStorage<F>;

    fn pure(&mut self, data: &[F]) -> Self::Repr {
        let mut storage = Self::Repr::alloc(data.len());
        storage.as_mut_slice().copy_from_slice(data);

        storage
    }

    fn to_host(&mut self, device_data: &Self::Repr) -> Vec<F> {
        let mut host_vec = Vec::with_capacity(device_data.len());

        let src_slice = device_data.as_slice();

        unsafe {
            std::ptr::copy_nonoverlapping(
                src_slice.as_ptr(),
                host_vec.as_mut_ptr(),
                device_data.len(),
            );
            host_vec.set_len(device_data.len());
        }

        host_vec
    }

    fn scale(&mut self, input: &Self::Repr, factor: F) -> Self::Repr
    where
        F: Semiring,
    {
        let mut output = Self::Repr::alloc(input.len());
        let input_slice = input.as_slice();
        let output_slice = output.as_mut_slice();

        for i in 0..input_slice.len() {
            output_slice[i] = input_slice[i] * factor;
        }

        output
    }

    fn unary<K: UnaryKernel<F>>(&mut self, input: &Self::Repr) -> Self::Repr {
        let mut output = Self::Repr::alloc(input.len());
        let input_slice = input.as_slice();
        let output_slice = output.as_mut_slice();

        for i in 0..input.len() {
            output_slice[i] = K::apply(input_slice[i]);
        }

        output
    }

    fn binary<K: BinaryKernel<F>>(&mut self, lhs: &Self::Repr, rhs: &Self::Repr) -> Self::Repr {
        let mut output = Self::Repr::alloc(lhs.len());
        let lhs_slice = lhs.as_slice();
        let rhs_slice = rhs.as_slice();
        let output_slice = output.as_mut_slice();

        for i in 0..lhs.len() {
            output_slice[i] = K::apply(lhs_slice[i], rhs_slice[i]);
        }

        output
    }

    fn reduce<K: ReduceKernel<F>>(&mut self, input: &Self::Repr) -> F {
        let input_slice = input.as_slice();
        let mut acc = K::init();

        for &val in input_slice {
            acc = K::step(acc, val);
        }

        acc
    }

    fn stream<K: StreamKernel<F>>(&mut self, input: &Self::Repr, kernel: K) -> Self::Repr {
        let mut output = Self::Repr::alloc(input.len());
        let input_slice = input.as_slice();
        let output_slice = output.as_mut_slice();
        let mut state = kernel.init();

        for i in 0..input.len() {
            output_slice[i] = kernel.step(&mut state, input_slice[i]);
        }

        output
    }

    fn contract<const RL: usize, const RR: usize>(
        &mut self,
        lhs: &Self::Repr,
        lhs_layout: crate::Layout<RL>,
        axis_l: usize,
        rhs: &Self::Repr,
        rhs_layout: crate::Layout<RR>,
        axis_r: usize,
    ) -> Self::Repr {
        todo!()
    }

    fn compact<const R: usize>(
        &mut self,
        src: &Self::Repr,
        layout: crate::Layout<R>,
    ) -> Self::Repr {
        todo!()
    }
}

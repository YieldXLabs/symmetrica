use super::{Backend, Storage, UnifiedStorage};
use algebra::{BinaryKernel, Data, ReduceKernel, StreamKernel, UnaryKernel};
use std::marker::PhantomData;

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

    fn compact(
        &mut self,
        src: &Self::Repr,
        shape: &[usize],
        strides: &[usize],
        offset: usize,
    ) -> Self::Repr {
        let numel: usize = shape.iter().product();
        let mut dst = Self::Repr::alloc(numel);

        let src_slice = src.as_slice();
        let dst_slice = dst.as_mut_slice();

        match shape.len() {
            0 => {
                dst_slice[0] = src_slice[offset];
            }
            1 => {
                for i in 0..shape[0] {
                    dst_slice[i] = src_slice[offset + i * strides[0]];
                }
            }
            2 => {
                let (h, w) = (shape[0], shape[1]);
                let (stride_h, stride_w) = (strides[0], strides[1]);
                let mut dst_idx = 0;

                for i in 0..h {
                    let mut src_idx = offset + i * stride_h;

                    for _ in 0..w {
                        dst_slice[dst_idx] = src_slice[src_idx];
                        dst_idx += 1;
                        src_idx += stride_w;
                    }
                }
            }
            _ => {
                let rank = shape.len();
                let mut idx = vec![0; rank];

                for linear in 0..numel {
                    let mut src_idx = offset;

                    match rank {
                        3 => {
                            src_idx += idx[0] * strides[0];
                            src_idx += idx[1] * strides[1];
                            src_idx += idx[2] * strides[2];
                        }
                        _ => {
                            for i in 0..rank {
                                src_idx += idx[i] * strides[i];
                            }
                        }
                    }

                    dst_slice[linear] = src_slice[src_idx];

                    for i in (0..rank).rev() {
                        idx[i] += 1;
                        if idx[i] < shape[i] {
                            break;
                        }
                        idx[i] = 0;
                    }
                }
            }
        }

        dst
    }

    fn unary<K: UnaryKernel<F>>(&mut self, input: &Self::Repr, kernel: K) -> Self::Repr {
        let mut output = Self::Repr::alloc(input.len());
        let input_slice = input.as_slice();
        let output_slice = output.as_mut_slice();

        for i in 0..input.len() {
            output_slice[i] = kernel.apply(input_slice[i]);
        }

        output
    }

    fn binary<K: BinaryKernel<F>>(
        &mut self,
        lhs: &Self::Repr,
        rhs: &Self::Repr,
        kernel: K,
    ) -> Self::Repr {
        let mut output = Self::Repr::alloc(lhs.len());
        let lhs_slice = lhs.as_slice();
        let rhs_slice = rhs.as_slice();
        let output_slice = output.as_mut_slice();

        for i in 0..lhs.len() {
            output_slice[i] = kernel.apply(lhs_slice[i], rhs_slice[i]);
        }

        output
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

    fn reduce<K: ReduceKernel<F>>(&mut self, input: &Self::Repr) -> F {
        let input_slice = input.as_slice();
        let mut acc = K::init();

        for &val in input_slice {
            acc = K::step(acc, val);
        }

        acc
    }
}

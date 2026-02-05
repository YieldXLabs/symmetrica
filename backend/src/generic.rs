use super::{Backend, Storage, UnifiedStorage};
use algebra::{BinaryKernel, Data, ReduceKernel, StreamKernel, UnaryKernel};

#[derive(Debug, Clone, Copy)]
pub struct GenericBackend;

impl GenericBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GenericBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for GenericBackend {
    type Storage<T: Data> = UnifiedStorage<T>;

    fn pure<T: Data>(&mut self, data: &[T]) -> Self::Storage<T> {
        let mut storage = UnifiedStorage::<T>::alloc(data.len());
        storage.as_mut_slice().copy_from_slice(data);
        storage
    }

    fn to_host<T: Data>(&mut self, device_data: &Self::Storage<T>) -> Vec<T> {
        // TODO: Zero-Copy Optimization.
        // If `device_data` is the *only* reference to the underlying storage (Arc count == 1),
        // we should define a method to `try_unwrap` the storage and return the inner `Vec`
        // without allocating a new one and copying bytes.
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

    fn compact<T: Data>(
        &mut self,
        src: &Self::Storage<T>,
        shape: &[usize],
        strides: &[usize],
        offset: usize,
    ) -> Self::Storage<T> {
        // TODO: Dimension Collapsing.
        // Before running the loops, we should check if any adjacent dimensions are contiguous.
        // e.g., if shape=[10, 20] and strides=[20, 1], this is actually 1D contiguous.
        // Collapsing dimensions reduces the nesting level and improves loop prediction.
        let numel = shape.iter().product::<usize>();
        let mut dst = UnifiedStorage::<T>::alloc(numel);

        let src_slice = src.as_slice();
        let dst_slice = dst.as_mut_slice();

        match shape.len() {
            0 => {
                dst_slice[0] = src_slice[offset];
            }
            1 => {
                // TODO: Vectorization.
                // If stride is 1 (contiguous), use `copy_from_slice`.
                // If stride is > 1, this loop can be auto-vectorized by LLVM,
                // but explicit SIMD gathers might be faster for large strides.
                for i in 0..shape[0] {
                    dst_slice[i] = src_slice[offset + i * strides[0]];
                }
            }
            2 => {
                let (h, w) = (shape[0], shape[1]);
                let (stride_h, stride_w) = (strides[0], strides[1]);
                let mut dst_idx = 0;

                // TODO: Parallelism.
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
                // TODO: Optimization.
                // This generic path is extremely slow due to the `idx` vector maintenance
                // inside the hot loop.
                // 1. Flatten the recursion into a specialized implementation.
                // 2. Pre-calculate offsets if N is small.
                let rank = shape.len();
                // TODO: Optimization
                // Heap allocation in a hot path
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

    fn unary<I: Data, K: UnaryKernel<I>>(
        &mut self,
        input: &Self::Storage<I>,
        kernel: K,
    ) -> Self::Storage<K::Output>
    where
        K::Output: Data,
    {
        let mut output = UnifiedStorage::<K::Output>::alloc(input.len());
        let input_slice = input.as_slice();
        let output_slice = output.as_mut_slice();

        // TODO: Parallelism
        for i in 0..input.len() {
            output_slice[i] = kernel.apply(input_slice[i]);
        }

        output
    }

    fn binary<L: Data, R: Data, K: BinaryKernel<L, R>>(
        &mut self,
        lhs: &Self::Storage<L>,
        rhs: &Self::Storage<R>,
        kernel: K,
    ) -> Self::Storage<K::Output>
    where
        K::Output: Data,
    {
        debug_assert_eq!(lhs.len(), rhs.len());
        let mut output = UnifiedStorage::<K::Output>::alloc(lhs.len());
        let lhs_slice = lhs.as_slice();
        let rhs_slice = rhs.as_slice();
        let output_slice = output.as_mut_slice();

        for i in 0..lhs.len() {
            output_slice[i] = kernel.apply(lhs_slice[i], rhs_slice[i]);
        }

        output
    }

    fn stream<I: Data, K: StreamKernel<I>>(
        &mut self,
        input: &Self::Storage<I>,
        kernel: K,
    ) -> Self::Storage<K::Output>
    where
        K::Output: Data,
    {
        let mut output = UnifiedStorage::<K::Output>::alloc(input.len());
        let input_slice = input.as_slice();
        let output_slice = output.as_mut_slice();
        let mut state = kernel.init();

        for i in 0..input.len() {
            output_slice[i] = kernel.step(&mut state, input_slice[i]);
        }

        output
    }

    fn reduce<I: Data, K: ReduceKernel<I>>(
        &mut self,
        input: &Self::Storage<I>,
        kernel: K,
    ) -> Self::Storage<K::Output>
    where
        K::Output: Data,
    {
        let input_slice = input.as_slice();
        let mut acc = kernel.init();

        for &val in input_slice {
            acc = kernel.step(acc, val);
        }

        let mut out = UnifiedStorage::<K::Output>::alloc(1);
        out.as_mut_slice()[0] = kernel.finish(acc);

        out
    }
}

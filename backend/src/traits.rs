use algebra::Real;
use std::fmt::Debug;

pub trait Storage<F>: Debug + Clone + Send + Sync {
    fn len(&self) -> usize;
    fn alloc(n: usize) -> Self;
    fn as_slice(&self) -> &[F];
    fn as_mut_slice(&mut self) -> &mut [F];
}

pub trait Evaluator<F: Real> {
    type Output: Storage<F>;

    fn collect<B: Backend<F>>(&self, backend: &mut B) -> Self::Output;
}

pub trait Backend<F: Real> {
    type Repr: Storage<F>;

    fn compute<E: Evaluator<F>>(&mut self, expr: &E) -> Self::Repr;
}

#[derive(Debug, Clone)]
pub struct UnifiedStorage<F: Real> {
    pub ptr: *mut F,    // raw pointer to memory
    pub len: usize,
}

unsafe impl<F: Real> Send for UnifiedStorage<F> {}
unsafe impl<F: Real> Sync for UnifiedStorage<F> {}

impl<F: Real> Storage<F> for UnifiedStorage<F> {
    fn len(&self) -> usize {
        self.len
    }

    fn alloc(n: usize) -> Self {
        // Allocate shared memory for CPU/GPU
        let layout = std::alloc::Layout::array::<F>(n).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) as *mut F };
        Self { ptr, len: n }
    }

    fn as_slice(&self) -> &[F] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [F] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<F: Real> Drop for UnifiedStorage<F> {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::array::<F>(self.len).unwrap();
        unsafe { std::alloc::dealloc(self.ptr as *mut u8, layout) }
    }
}

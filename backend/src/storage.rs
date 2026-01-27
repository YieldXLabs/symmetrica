use super::Storage;
use algebra::Data;
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ptr::NonNull;

#[derive(Debug)]
pub struct UnifiedStorage<F: Data> {
    pub ptr: NonNull<F>,
    pub len: usize,
}

unsafe impl<F: Data> Send for UnifiedStorage<F> {}
unsafe impl<F: Data> Sync for UnifiedStorage<F> {}

impl<F: Data> Storage<F> for UnifiedStorage<F> {
    fn len(&self) -> usize {
        self.len
    }

    fn alloc(n: usize) -> Self {
        let layout = Layout::array::<F>(n).expect("Allocation too large");
        unsafe {
            let raw = alloc_zeroed(layout) as *mut F;
            let ptr = NonNull::new(raw).expect("Out of memory");
            Self { ptr, len: n }
        }
    }

    fn as_slice(&self) -> &[F] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [F] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<F: Data> Clone for UnifiedStorage<F> {
    fn clone(&self) -> Self {
        let new_storage = <Self as Storage<F>>::alloc(self.len);

        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_storage.ptr.as_ptr(), self.len);
        }

        new_storage
    }
}

impl<F: Data> Drop for UnifiedStorage<F> {
    fn drop(&mut self) {
        let layout = Layout::array::<F>(self.len).unwrap();
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, layout);
        }
    }
}

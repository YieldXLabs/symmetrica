use super::Storage;
use algebra::Data;
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ptr::NonNull;
use std::sync::Arc;

#[derive(Debug)]
struct RawBuffer<F: Data> {
    ptr: NonNull<F>,
    len: usize,
}

unsafe impl<F: Data> Send for RawBuffer<F> {}
unsafe impl<F: Data> Sync for RawBuffer<F> {}

impl<F: Data> RawBuffer<F> {
    fn new(n: usize) -> Self {
        if n == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
            };
        }
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

impl<F: Data> Clone for RawBuffer<F> {
    fn clone(&self) -> Self {
        let new_buf = Self::new(self.len);

        if self.len > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_buf.ptr.as_ptr(), self.len);
            }
        }
        new_buf
    }
}

impl<F: Data> Drop for RawBuffer<F> {
    fn drop(&mut self) {
        if self.len > 0 {
            let layout = Layout::array::<F>(self.len).unwrap();
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct UnifiedStorage<F: Data> {
    inner: Arc<RawBuffer<F>>,
}

unsafe impl<F: Data> Send for UnifiedStorage<F> {}
unsafe impl<F: Data> Sync for UnifiedStorage<F> {}

impl<F: Data> Storage for UnifiedStorage<F> {
    type Elem = F;

    fn len(&self) -> usize {
        self.inner.len
    }

    fn alloc(n: usize) -> Self {
        Self {
            inner: Arc::new(RawBuffer::new(n)),
        }
    }

    fn as_slice(&self) -> &[F] {
        self.inner.as_slice()
    }

    fn as_mut_slice(&mut self) -> &mut [F] {
        Arc::make_mut(&mut self.inner).as_mut_slice()
    }
}

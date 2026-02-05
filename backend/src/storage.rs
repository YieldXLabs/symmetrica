use super::Storage;
use algebra::Data;
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ptr::NonNull;
use std::sync::Arc;

// TODO: SIMD Alignment.
// Standard `Layout::array` uses `align_of::<F>()` (usually 4 or 8 bytes).
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
            // TODO: Uninitialized Allocation.
            // `alloc_zeroed` is safe but expensive (memset).
            // For tensors that will be immediately filled (e.g., via `matmul`),
            // use `alloc` (uninit) and maybe wrap result in `MaybeUninit<F>`.
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
                // TODO: Run Destructors (`drop_in_place`).
                // Currently, this only deallocates memory (`free`).
                // It does NOT call the destructor of `F`.
                // If `F` is `TradingFloat`, this is fine.
                // If `F` owns heap memory, this is a leak.
                // std::ptr::drop_in_place(std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len));
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

// TODO: Memory Pooling / Caching Allocator.
// Allocating/Deallocating large buffers repeatedly (e.g., in a training loop)
// causes memory fragmentation and high syscall overhead.
// Implement a `BumpAllocator` or `SlabAllocator` instead of using global heap directly.
#[derive(Debug, Clone)]
pub struct UnifiedStorage<F: Data> {
    // TODO: Copy-on-Write Strategy.
    // Using `Arc` gives us atomic reference counting.
    // However, `Arc::make_mut` performs a deep clone if the reference count > 1.
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

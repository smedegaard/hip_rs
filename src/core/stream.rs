use crate::{sys, HipResult, Result};

/// A handle to a HIP stream that executes commands in order.
#[derive(Debug)]
pub struct Stream {
    handle: sys::hipStream_t,
}

impl Stream {
    /// Creates a new asynchronous stream.
    ///
    /// The stream is allocated on the heap and will remain allocated even if the handle goes out of scope.
    /// To release the memory used by the stream, the application must call [`Stream::destroy()`].
    ///
    /// # Returns
    /// * `Ok(Stream)` - A new asynchronous stream
    /// * `Err(HipError)` - If stream creation fails
    ///
    /// # Examples
    /// ```
    /// use hip_rs::Stream;
    ///
    /// let stream = Stream::create().unwrap();
    /// ```
    pub fn create() -> Result<Self> {
        let mut stream: sys::hipStream_t = std::ptr::null_mut();
        unsafe {
            let code = sys::hipStreamCreate(&mut stream);
            (Self { handle: stream }, code).to_result()
        }
    }

    /// Returns the raw stream handle.
    pub fn handle(&self) -> sys::hipStream_t {
        self.handle
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let code = sys::hipStreamDestroy(self.handle);
                if code != 0 {
                    log::error!("Failed to destroy HIP stream: {}", code);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_create() {
        let stream = Stream::create();
        assert!(stream.is_ok());
        let stream = stream.unwrap();
        assert!(!stream.handle().is_null());
    }

    #[test]
    fn test_stream_drop() {
        let stream = Stream::create().unwrap();
        drop(stream); // Stream should be properly destroyed here
    }
}

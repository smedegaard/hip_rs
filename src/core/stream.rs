use super::result::{HipResult, HipStatus};
use crate::result::ResultExt;
use crate::sys;

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
    pub fn create() -> HipResult<Self> {
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

    /// Queries the completion status of all operations in the stream.
    ///
    /// This function provides a snapshot of the current state of the stream. It checks if all
    /// operations in the specified stream have completed execution.
    ///
    /// # Thread Safety
    /// This function is thread-safe, but note that the stream status may change immediately
    /// after the query if other threads are submitting work to the same stream.
    ///
    /// # Returns
    /// * `Ok(())` - All operations in the stream have completed
    /// * `Err(HipError)` - Either:
    ///   - `HipErrorKind::NotReady` if operations are still in progress
    ///   - `HipErrorKind::InvalidHandle` if the stream handle is invalid
    pub fn query_stream(&self) -> HipResult<()> {
        unsafe {
            let code = sys::hipStreamQuery(self.handle);
            ((), code).to_result()
        }
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
    fn test_stream_query() {
        let stream = Stream::create().unwrap();

        // Test querying an empty stream (should be complete)
        let result = stream.query_stream();
        assert!(result.is_ok(), "Empty stream should report as complete");
    }

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

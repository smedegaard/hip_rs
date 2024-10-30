//! HIP Runtime API bindings
mod result;
mod safe;
pub mod sys;

pub use result::{HipError, Result};
pub use safe::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hip_init() {
        initialize().expect("Failed to initialize HIP");
        let count = get_device_count().expect("Failed to get device count");
        println!("Found {} HIP devices", count);
    }
}

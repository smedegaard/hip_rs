use super::result::{HipError, Result};
use super::sys;

pub fn initialize() -> Result<()> {
    let result = unsafe { sys::hipInit(0) };
    if result != 0 {
        return Err(HipError(result.try_into().unwrap()));
    }
    Ok(())
}

pub fn get_device_count() -> Result<i32> {
    let mut count = 0;
    let result = unsafe { sys::hipGetDeviceCount(&mut count) };
    if result != 0 {
        return Err(HipError(result.try_into().unwrap()));
    }
    Ok(count)
}

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

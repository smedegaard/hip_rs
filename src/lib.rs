#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

mod hip_sys;

pub fn initialize() -> Result<(), i32> {
    let result = unsafe { hip_sys::hipInit() };
    if result != 0 {
        // hipSuccess is 0
        // Convert hipError_t (u32) to i32
        return Err(result.try_into().unwrap());
    }
    Ok(())
}

pub fn get_device_count() -> Result<i32, i32> {
    let mut count = 0;
    let result = unsafe { hip_sys::hipGetDeviceCount(&mut count) };
    if result != 0 {
        return Err(result.try_into().unwrap());
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

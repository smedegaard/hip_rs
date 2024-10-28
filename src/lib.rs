#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

mod bindings;

pub fn initialize() -> Result<(), i32> {
    let result = unsafe { bindings::hip_initialize() };
    if result != 0 {
        // hipSuccess is 0
        return Err(result);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hip_init() {
        match initialize() {
            Ok(()) => println!("HIP initialized successfully"),
            Err(e) => panic!("Failed to initialize HIP: error {}", e),
        }
    }
}

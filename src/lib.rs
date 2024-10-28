#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

// Include the generated bindings
mod bindings;

// Create safe wrappers around the unsafe C++ functions
pub fn add_numbers(a: i32, b: i32) -> i32 {
    unsafe { bindings::add_numbers(a, b) }
}

pub fn get_device_count() -> i32 {
    unsafe { bindings::get_device_count() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_numbers() {
        assert_eq!(add_numbers(2, 2), 4);
    }

    #[test]
    fn test_device_count() {
        // For now, this should return 0
        assert_eq!(get_device_count(), 0);
    }
}

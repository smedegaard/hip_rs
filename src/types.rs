#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Device {
    pub(crate) id: i32,
}

impl Device {
    /// Create a new Device handle
    pub fn new(id: i32) -> Self {
        Device { id }
    }

    /// Get the raw device ID
    pub fn id(&self) -> i32 {
        self.id
    }
}

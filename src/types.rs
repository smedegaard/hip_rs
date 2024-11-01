#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Device {
    pub(crate) id: u32,
}

impl Device {
    /// Create a new Device handle
    pub fn new(id: u32) -> Self {
        Device { id }
    }

    /// Get the raw device ID
    pub fn id(&self) -> u32 {
        self.id
    }
}

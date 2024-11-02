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

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
pub struct Version {
    major: u32,
    minor: u32,
    patch: u32,
}

use std::fmt;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
pub struct Version {
    major: u32,
    minor: u32,
    patch: Option<u32>,
}

impl Version {
    pub fn new(major: u32, minor: u32, patch: Option<u32>) -> Self {
        Version {
            major,
            minor,
            patch,
        }
    }

    // Renamed constructor for the full version number
    pub fn major_minor_patch(major: u32, minor: u32, patch: u32) -> Self {
        Version::new(major, minor, Some(patch))
    }

    // Renamed constructor for version without patch
    pub fn major_minor(major: u32, minor: u32) -> Self {
        Version::new(major, minor, None)
    }

    pub fn to_string(&self) -> String {
        match self.patch {
            Some(patch) => format!("{}.{}.{}", self.major, self.minor, patch),
            None => format!("{}.{}", self.major, self.minor),
        }
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.patch {
            Some(patch) => write!(f, "{}.{}.{}", self.major, self.minor, patch),
            None => write!(f, "{}.{}", self.major, self.minor),
        }
    }
}

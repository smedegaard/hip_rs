// src/runtime/mod.rs
mod init;
mod memory;
pub mod sys;

// Re-export core functionality
pub use init::*;
pub use memory::*;

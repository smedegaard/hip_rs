use bitflags::bitflags;

bitflags! {
    pub struct DeviceMallocFlag: u32 {
        const DEFAULT = 0x0;
        const FINEGRAINED = 0x1;
        const SIGNAL_MEMORY = 0x2;
        const UNCACHED = 0x3;
        const CONTIGUOUS = 0x4;
    }
}

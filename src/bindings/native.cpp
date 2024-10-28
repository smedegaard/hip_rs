#include "wrapper.hpp"

hipError_t hip_initialize() {
    return hipInit(0);
}

hipError_t hip_get_device_count(int* count) {
    return hipGetDeviceCount(count);
}

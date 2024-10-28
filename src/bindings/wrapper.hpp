#pragma once

#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize HIP and check for errors
hipError_t hip_initialize();

// Get number of available GPU devices
hipError_t hip_get_device_count(int* count);

#ifdef __cplusplus
}
#endif

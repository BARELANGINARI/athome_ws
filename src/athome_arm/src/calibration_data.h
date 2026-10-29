#pragma once
#include <cstdint>

// Dihasilkan otomatis oleh calibrate_arm.cpp
struct CalibratedServoConfig {
    uint8_t  id;
    uint16_t raw_zero;
    int16_t  trim_ticks;
    uint16_t center_tick;
    uint16_t cw_limit;
    uint16_t ccw_limit;
};

constexpr CalibratedServoConfig ARM_CALIBRATION[] = {
    { 1, 802, 0, 802, 10, 1010 },
    { 4, 809, 0, 809, 10, 1010 },
    { 5, 221, 0, 221, 10, 1010 },
    { 6, 506, 0, 506, 491, 779 },
};

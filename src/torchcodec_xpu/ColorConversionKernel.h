// Copyright (c) 2025 Dmitry Rogozhkin.

#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

namespace facebook::torchcodec {

void convertNV12ToRGB(
    const uint8_t* y_plane,
    const uint8_t* uv_plane,
    uint8_t* rgb_output,
    int width,
    int height,
    int stride,
    sycl::queue& queue,
    bool fullrange = 1);

void detileNV12(
    const uint8_t* tiled_y_plane,
    const uint8_t* tiled_uv_plane,
    uint8_t* linear_y_output,
    uint8_t* linear_uv_output,
    int width,
    int height,
    int stride,
    sycl::queue& queue);

} // namespace facebook::torchcodec


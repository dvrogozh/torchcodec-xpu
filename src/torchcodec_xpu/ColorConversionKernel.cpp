// Copyright (c) 2025 Dmitry Rogozhkin.

#include "ColorConversionKernel.h"
#include <algorithm> // For std::clamp
#include <cmath>

namespace facebook::torchcodec {

using float3x3 = std::array<sycl::float3, 3>;

const float3x3 rgb_matrix_bt709 = {
  sycl::float3{ 1.0, 0.0, 1.5748 },
  sycl::float3{ 1.0, -0.187324, -0.468124 },
  sycl::float3{ 1.0, 1.8556, 0.0 }
};

//const sycl::float3 rgb_matrix_bt601[3] = {
//  { 1.0, 0.0, 1.402 },
//  { 1.0, -0.344136, -0.714136 },
//  { 1.0, 1.772, 0.0}
//};

sycl::uchar3 yuv2rgb(uint8_t y, uint8_t u, uint8_t v, bool fullrange, const float3x3 &rgb_matrix) {
  sycl::float3 src;
  if (fullrange) {
    src = sycl::float3(y/255.0f, (u-128.0f)/255.0f - 0.5f, (v-128.0f)/255.0f - 0.5f);
  } else {
    src = sycl::float3((y-16.0f)/219.0f, (u-128.0f)/224.0f, (v-128.0f)/224.0f);
  }

  sycl::float3 fdst;
  fdst.x() = sycl::dot(src, rgb_matrix[0]);
  fdst.y() = sycl::dot(src, rgb_matrix[1]);
  fdst.z() = sycl::dot(src, rgb_matrix[2]);

  sycl::uchar3 dst;
  dst.x() = (uint8_t)std::clamp(fdst[0] * 255.0f, 0.0f, 255.0f);
  dst.y() = (uint8_t)std::clamp(fdst[1] * 255.0f, 0.0f, 255.0f);
  dst.z() = (uint8_t)std::clamp(fdst[2] * 255.0f, 0.0f, 255.0f);
  return dst;
}

struct NV12toRGBKernel {
  sycl::accessor<uint8_t, 1, sycl::access::mode::read> y_acc;
  sycl::accessor<uint8_t, 1, sycl::access::mode::read> uv_acc;
  sycl::accessor<uint8_t, 1, sycl::access::mode::write> rgb_acc;
  int width;
  int height;
  int stride;
  bool fullrange;
  float3x3 rgb_matrix;

  NV12toRGBKernel(
      sycl::accessor<uint8_t, 1, sycl::access::mode::read> y_acc,
      sycl::accessor<uint8_t, 1, sycl::access::mode::read> uv_acc,
      sycl::accessor<uint8_t, 1, sycl::access::mode::write> rgb_acc,
      int width,
      int height,
      int stride,
      bool fullrange,
      const float3x3 &rgb_matrix):
    y_acc(y_acc),
    uv_acc(uv_acc),
    rgb_acc(rgb_acc),
    width(width),
    height(height),
    stride(stride),
    fullrange(fullrange),
    rgb_matrix(rgb_matrix)
  {}

  void operator()(sycl::id<2> idx) const {
    int yx = idx[1];
    int yy = idx[0];

    if (yx >= width || yy >= height) {
      return;
    }

    int ux = sycl::floor(yx/2.0);
    int uy = sycl::floor(yy/2.0);

    uint8_t y = y_acc[yy * stride + yx];
    uint8_t u = uv_acc[uy * stride + ux * 2];
    uint8_t v = uv_acc[uy * stride + ux * 2 + 1];

    sycl::uchar3 rgb = yuv2rgb(y, u, v, fullrange, rgb_matrix);

    int rgb_idx = 3 * (yy * width + yx);

    rgb_acc[rgb_idx + 0] = rgb.x();
    rgb_acc[rgb_idx + 1] = rgb.y();
    rgb_acc[rgb_idx + 2] = rgb.z();
  }
};


// Detach tiling implementation.
struct DetileNV12Kernel{
  sycl::accessor<uint8_t, 1, sycl::access::mode::read> tiled_y_acc;
  sycl::accessor<uint8_t, 1, sycl::access::mode::read> tiled_uv_acc;
  sycl::accessor<uint8_t, 1, sycl::access::mode::write> linear_y_acc;
  sycl::accessor<uint8_t, 1, sycl::access::mode::write> linear_uv_acc;
  int width;
  int height;
  int stride;

  DetileNV12Kernel(
    sycl::accessor<uint8_t, 1, sycl::access::mode::read> tiled_y_acc,
    sycl::accessor<uint8_t, 1, sycl::access::mode::read> tiled_uv_acc,
    sycl::accessor<uint8_t, 1, sycl::access::mode::write> linear_y_acc,
    sycl::accessor<uint8_t, 1, sycl::access::mode::write> linear_uv_acc,
    int width,
    int height,
    int stride):
    tiled_y_acc(tiled_y_acc),
    tiled_uv_acc(tiled_uv_acc),
    linear_y_acc(linear_y_acc),
    linear_uv_acc(linear_uv_acc),
    width(width),
    height(height),
    stride(stride)
    {}


  void operator()(sycl::id<2> idx) const{
    int x = idx[1];
    int y = idx[0];

    if (x >= width || y >= height) return;

    auto get_offset_tile_y = [](int x, int y, int stride) -> size_t {
    const int TileW = 128;  // Tile width in bytes
    const int TileH = 32;   // Tile height in rows
    const int OWordSize = 16; // OWord = 16 bytes
    const int TileSize = TileW * TileH;  // 4096 bytes per tile
    
    // Which tile does this pixel belong to?
    int tile_x = x / TileW;
    int tile_y = y / TileH;
    
    // Position within the tile
    int x_in_tile = x % TileW;
    int y_in_tile = y % TileH;
    

    // Block position added to remove swap of 64-byte blocks in the tile (TileY XOR pattern)
    int block_x = x_in_tile / 64;  // width of pixel blocks
    int block_y = y_in_tile / 4;   // heigh of pixel blocks



    // Y-Tiling: Column-major OWord layout
    // OWord index (0-7): which 16-byte column within the tile
    int oword_idx = x_in_tile / OWordSize;
    // Offset within OWord (0-15)
    int offset_in_oword = x_in_tile % OWordSize;

    int sub_tile_size = OWordSize * 4;
    int sub_tile_y = y_in_tile / 4;
    int y_in_sub_tile = y_in_tile % 4;

    // conditional to remove swap of 64-byte blocks in the tile (TileY XOR pattern)
    if ((block_x ^ block_y ) & 0x1){
        block_x ^= 1;
        block_y ^= 1;

        x_in_tile = block_x * 64 + (x_in_tile % 64);
        y_in_tile = block_y * 4 + (y_in_tile % 4);

        sub_tile_y = block_y;
        y_in_sub_tile = y_in_tile % 4;

        oword_idx = x_in_tile / OWordSize;
        offset_in_oword = x_in_tile % 16;

    }
    
    int offset_in_tile = (sub_tile_y * TileW/OWordSize + oword_idx) * sub_tile_size + y_in_sub_tile * OWordSize + offset_in_oword;

    // Number of tiles per row
    int stride_in_tiles = stride / TileW;
    
    // Final tiled offset
    size_t tile_offset = (size_t)(tile_y * stride_in_tiles + tile_x) * TileSize;
    return tile_offset + offset_in_tile;
    };

    // Detile Y Plane
    size_t linear_idx_y = (size_t)y * stride + x;
    size_t tiled_idx_y = get_offset_tile_y(x, y, stride);
    linear_y_acc[linear_idx_y] = tiled_y_acc[tiled_idx_y];

    // Detile UV Plane (half height for NV12)
    // UV samples are interleaved: U0,V0,U1,V1,... in a row
    if (y < height / 2) {
        size_t linear_idx_uv = (size_t)y * stride + x;
        size_t tiled_idx_uv = get_offset_tile_y(x, y, stride);
        linear_uv_acc[linear_idx_uv] = tiled_uv_acc[tiled_idx_uv];
    }

  }

};

void convertNV12ToRGB(
    sycl::queue& queue,
    const uint8_t* y_plane,
    const uint8_t* uv_plane,
    uint8_t* rgb_output,
    int width,
    int height,
    int stride,
    bool fullrange) {
  size_t y_size = stride * height;
  size_t uv_size = stride * height / 2;
  size_t rgb_size = width * height * 3;

  sycl::buffer<uint8_t, 1> y_buf(y_plane, sycl::range<1>(y_size));
  sycl::buffer<uint8_t, 1> uv_buf(uv_plane, sycl::range<1>(uv_size));
  sycl::buffer<uint8_t, 1> rgb_buf(rgb_output, sycl::range<1>(rgb_size));

  queue.submit([&](sycl::handler& cgh) {
    auto y_acc = y_buf.get_access<sycl::access::mode::read>(cgh);
    auto uv_acc = uv_buf.get_access<sycl::access::mode::read>(cgh);
    auto rgb_acc = rgb_buf.get_access<sycl::access::mode::write>(cgh);

    NV12toRGBKernel kernel(
      y_acc, uv_acc, rgb_acc,
      width, height, stride,
      fullrange, rgb_matrix_bt709);

    cgh.parallel_for(
        sycl::range<2>(height, width),
        kernel);
  });

  queue.wait();
}

void detileNV12(
    sycl::queue& queue,
    const uint8_t* tiled_y_plane,
    const uint8_t* tiled_uv_plane,
    uint8_t* linear_y_output,
    uint8_t* linear_uv_output,
    int width,
    int height,
    int stride) {
    
  size_t y_size = stride * height;
  size_t uv_size = stride * height / 2;
  
  // Create buffers
  sycl::buffer<uint8_t, 1> tiled_y_buf(tiled_y_plane, sycl::range<1>(y_size));
  sycl::buffer<uint8_t, 1> tiled_uv_buf(tiled_uv_plane, sycl::range<1>(uv_size));
  sycl::buffer<uint8_t, 1> linear_y_buf(linear_y_output, sycl::range<1>(y_size));
  sycl::buffer<uint8_t, 1> linear_uv_buf(linear_uv_output, sycl::range<1>(uv_size));
  
  queue.submit([&](sycl::handler& cgh) {
    auto tiled_y_acc = tiled_y_buf.get_access<sycl::access::mode::read>(cgh);
    auto tiled_uv_acc = tiled_uv_buf.get_access<sycl::access::mode::read>(cgh);
    auto linear_y_acc = linear_y_buf.get_access<sycl::access::mode::write>(cgh);
    auto linear_uv_acc = linear_uv_buf.get_access<sycl::access::mode::write>(cgh);
    
    DetileNV12Kernel kernel(
      tiled_y_acc, tiled_uv_acc, linear_y_acc, linear_uv_acc,
      width, height, stride);
    
    cgh.parallel_for(sycl::range<2>(height, width), kernel);
  });
  
  queue.wait();
}

// This function is called during library initialization to ensure
// the SYCL runtime registers the kernel associated with this type.
void registerColorConversionKernel() {
  // Creating a dummy pointer to the kernel type is often enough
  // to force the compiler to emit the necessary RTTI/integration info.
  // We use volatile to prevent optimization.
  volatile size_t s1 = sizeof(NV12toRGBKernel);
  volatile size_t s2 = sizeof(DetileNV12Kernel);
  (void)s1; (void)s2;
}


} // namespace facebook::torchcodec

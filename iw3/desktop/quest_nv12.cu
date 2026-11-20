// 连续 RGB FP32 [3,2160,7680] -> 连续 NV12 U8 [3240,7680]；每个线程写一个 2×2 块。
constexpr double kr = 0.2126;
constexpr double kb = 0.0722;
constexpr double kg = 1.0 - kr - kb;

struct Yuv { float y, u, v; };

__device__ __forceinline__ Yuv convert(const float* rgb, int area, int p) {
    float r = rgb[p], g = rgb[p + area], b = rgb[p + 2 * area];
    // 原 4K FP32 1×1 卷积先乘 R，再按 G、B 顺序各做一次舍入相同的乘加。
    return {
        __fmaf_rn(float(kb), b, __fmaf_rn(float(kg), g, float(kr) * r)),
        __fmaf_rn(0.5f, b, __fmaf_rn(float(-kg / (2.0 * (1.0 - kb))), g, float(-kr / (2.0 * (1.0 - kb))) * r)),
        __fmaf_rn(float(-kb / (2.0 * (1.0 - kr))), b, __fmaf_rn(float(-kg / (2.0 * (1.0 - kr))), g, 0.5f * r)),
    };
}

__device__ __forceinline__ unsigned char quantize(float v) {
    return static_cast<unsigned char>(__float2uint_rz(fminf(fmaxf(v, 0.f), 255.f)));
}

__device__ __forceinline__ unsigned char y_byte(float y) {
    // 保留原 CUDA 标量运算的 FP32 舍入顺序。
    return quantize(((y * 219.f + 16.f) * (1.f / 255.f)) * 255.f + 0.5f);
}

__device__ __forceinline__ unsigned char uv_byte(float a, float b, float c, float d) {
    // 与 2:1 双线性一致：先左右各折半，再把上下两组折半。
    float top = 0.5f * a + 0.5f * b;
    float bottom = 0.5f * c + 0.5f * d;
    return quantize((0.5f * top + 0.5f * bottom) * 224.f + 128.5f);
}

extern "C" __global__ void pack_nv12(const float* rgb, unsigned char* packed, int height, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int area = height * width;
    if (i >= area / 4) return;
    int row = i / (width / 2), col = i % (width / 2);
    int p = 2 * row * width + 2 * col;
    Yuv a = convert(rgb, area, p);
    Yuv b = convert(rgb, area, p + 1);
    Yuv c = convert(rgb, area, p + width);
    Yuv d = convert(rgb, area, p + width + 1);
    packed[p] = y_byte(a.y);
    packed[p + 1] = y_byte(b.y);
    packed[p + width] = y_byte(c.y);
    packed[p + width + 1] = y_byte(d.y);
    packed[area + 2 * i] = uv_byte(a.u, b.u, c.u, d.u);
    packed[area + 2 * i + 1] = uv_byte(a.v, b.v, c.v, d.v);
}

// Torch 已拷贝的 NV12 FP32 Y[1,1,H,W]、U/V[1,1,H/2,W/2] → RGB FP32 [1,3,H,W]。
__device__ __forceinline__ float bilinear(
    const float* p, int i00, int i01, int i10, int i11, float wx, float wy
) {
    float left = 1.f - wx, top = 1.f - wy;
    return top * (left * p[i00] + wx * p[i01]) + wy * (left * p[i10] + wx * p[i11]);
}

extern "C" __global__ void unpack_nv12(
    const float* py, const float* pu, const float* pv, const float* weights, const float* bias,
    float* rgb, int height, int width, int y_pitch, int uv_pitch, int uv_step
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // 2×，align_corners=False：坐标 (dst + 0.5) / 2 - 0.5，边缘复用最近像素。
    int x0 = x > 0 ? (x - 1) / 2 : 0, y0 = y > 0 ? (y - 1) / 2 : 0;
    int x1 = min(x0 + 1, width / 2 - 1), y1 = min(y0 + 1, height / 2 - 1);
    float wx = x == 0 ? 0.f : (x & 1) ? 0.25f : 0.75f;
    float wy = y == 0 ? 0.f : (y & 1) ? 0.25f : 0.75f;
    int i00 = y0 * uv_pitch + x0 * uv_step, i01 = y0 * uv_pitch + x1 * uv_step;
    int i10 = y1 * uv_pitch + x0 * uv_step, i11 = y1 * uv_pitch + x1 * uv_step;
    float fy = py[y * y_pitch + x];
    float fu = bilinear(pu, i00, i01, i10, i11, wx, wy);
    float fv = bilinear(pv, i00, i01, i10, i11, wx, wy);

    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        int m = 3 * c;
        // 使用原 Torch 的矩阵及偏置；按 Y、U、V 顺序累加，卷积后单独加偏置。
        float value = __fmaf_rn(weights[m + 2], fv, __fmaf_rn(weights[m + 1], fu, weights[m] * fy)) + bias[c];
        rgb[(c * height + y) * width + x] = value < 0.f ? 0.f : value > 1.f ? 1.f : value;
    }
}

__device__ __forceinline__ float warp_lerp(float a, float b, float wa, float wb) {
    // 固定 ATen CUDA 的乘加顺序；转色核保持禁用自动收缩。
    return __fmaf_rn(wa, a, __fmul_rn(wb, b));
}

template<int H, int W, int GH, int GW>
__device__ void warp_sbs_4k(const float* rgb, const float* left_grid, const float* right_grid, float* sbs) {
    constexpr int HW = H * W, GHW = GH * GW;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * HW) return;
    const bool right = i >= HW;
    const int pixel = right ? i - HW : i;
    const int y = pixel / W, x = pixel % W;
    const float* grid = right ? right_grid : left_grid;
    // align_corners=True：低清坐标按两个轴先插值，避免写出完整坐标图。
    const float gy = (float(GH - 1) / float(H - 1)) * y;
    const float gx = (float(GW - 1) / float(W - 1)) * x;
    const int y0 = int(gy), x0 = int(gx);
    const int y1 = y0 + (y0 < GH - 1), x1 = x0 + (x0 < GW - 1);
    const float wy1 = gy - y0, wx1 = gx - x0;
    const float wy0 = 1.0f - wy1, wx0 = 1.0f - wx1;
    const int a = y0 * GW + x0, b = y0 * GW + x1;
    const int c = y1 * GW + x0, d = y1 * GW + x1;
    const float ax = warp_lerp(warp_lerp(grid[a], grid[b], wx0, wx1),
                               warp_lerp(grid[c], grid[d], wx0, wx1), wy0, wy1);
    grid += GHW;
    const float ay = warp_lerp(warp_lerp(grid[a], grid[b], wx0, wx1),
                               warp_lerp(grid[c], grid[d], wx0, wx1), wy0, wy1);
    // grid_sample 的 border 模式；双眼在最终 SBS 的独立半幅直接裁值写出。
    const float sx = fminf(float(W - 1), fmaxf(((ax + 1.0f) / 2.0f) * (W - 1), 0.0f));
    const float sy = fminf(float(H - 1), fmaxf(((ay + 1.0f) / 2.0f) * (H - 1), 0.0f));
    const int ix = int(sx), iy = int(sy), east = ix + 1, south = iy + 1;
    const float nw = (east - sx) * (south - sy), ne = (sx - ix) * (south - sy);
    const float sw = (east - sx) * (sy - iy), se = (sx - ix) * (sy - iy);
    const int top = iy * W + ix, bottom = south * W + ix;
    const int target = y * (2 * W) + x + (right ? W : 0);
    #pragma unroll
    for (int channel = 0; channel < 3; ++channel) {
        const float* plane = rgb + channel * HW;
        float value = __fmaf_rn(plane[top], nw, 0.0f);
        if (east < W) value = __fmaf_rn(plane[top + 1], ne, value);
        if (south < H) value = __fmaf_rn(plane[bottom], sw, value);
        if (east < W && south < H) value = __fmaf_rn(plane[bottom + 1], se, value);
        sbs[channel * (2 * HW) + target] = fminf(fmaxf(value, 0.0f), 1.0f);
    }
}

extern "C" __global__ void warp_sbs_landscape(const float* rgb, const float* l, const float* r, float* sbs) {
    warp_sbs_4k<2160, 3840, 392, 686>(rgb, l, r, sbs);
}

extern "C" __global__ void warp_sbs_portrait(const float* rgb, const float* l, const float* r, float* sbs) {
    warp_sbs_4k<3840, 2160, 686, 392>(rgb, l, r, sbs);
}

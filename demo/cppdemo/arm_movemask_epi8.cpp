// Code for blog post: https://yangwenbo.com/articles/arm-neon-_mm_movemask_epi8.html
// Compile: g++ -g -O2 --std=c++17 -march=armv8.1-a arm_movemask_epi8.cpp
// Run: ./a.out

#include <iostream>
#include <chrono>
#include <vector>
#include <cstdint>
#include <random>
#include <arm_neon.h> // 包含 ARM NEON 头文件

// Yves Daoust 的回答 (7 votes): 与 _mm_movemask_epi8 略有不符，要求输入的每个 8 bits 全 0 或全 1
inline uint32_t vmovemask_u8_YvesDaoust(uint8x16_t a) {
    const uint8_t __attribute__ ((aligned (16))) _Powers[16]=
        { 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128 };
    // Set the powers of 2 (do it once for all, if applicable)
    uint8x16_t Powers= vld1q_u8(_Powers);
    // Compute the mask from the input
    uint64x2_t Mask= vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(vandq_u8(a, Powers))));
    // Get the resulting bytes
    uint32_t Output;
    vst1q_lane_u8((uint8_t*)&Output + 0, (uint8x16_t)Mask, 0);
    vst1q_lane_u8((uint8_t*)&Output + 1, (uint8x16_t)Mask, 8);
    return Output;
}

// David 对 Yves Daoust 回答最后三行进行了一些改进
inline uint32_t vmovemask_u8_David(uint8x16_t a) {
    const uint8_t __attribute__ ((aligned (16))) _Powers[16]=
        { 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128 };
    // Set the powers of 2 (do it once for all, if applicable)
    uint8x16_t Powers= vld1q_u8(_Powers);
    // Compute the mask from the input
    uint64x2_t Mask= vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(vandq_u8(a, Powers))));
    // Get the resulting bytes
    uint32_t Output = vgetq_lane_u64(Mask, 0) + (vgetq_lane_u64(Mask, 1) << 8);
    return Output;
}

// EasyasPi 的回答（4 votes): 标准实现了 _mm_movemask_epi8，被 simde 库采纳，link:
// https://github.com/simd-everywhere/simde/blob/master/simde/x86/sse2.h
inline uint32_t vmovemask_u8_EasyasPi(uint8x16_t input)
{
    // Example input (half scale):
    // 0x89 FF 1D C0 00 10 99 33
    // Shift out everything but the sign bits
    // 0x01 01 00 01 00 00 01 00
    uint16x8_t high_bits = vreinterpretq_u16_u8(vshrq_n_u8(input, 7));
    // Merge the even lanes together with vsra. The '??' bytes are garbage.
    // vsri could also be used, but it is slightly slower on aarch64.
    // 0x??03 ??02 ??00 ??01
    uint32x4_t paired16 = vreinterpretq_u32_u16(vsraq_n_u16(high_bits, high_bits, 7));
    // Repeat with wider lanes.
    // 0x??????0B ??????04
    uint64x2_t paired32 = vreinterpretq_u64_u32(vsraq_n_u32(paired16, paired16, 14));
    // 0x??????????????4B
    uint8x16_t paired64 = vreinterpretq_u8_u64(vsraq_n_u64(paired32, paired32, 28));
    // Extract the low 8 bits from each lane and join.
    // 0x4B
    return vgetq_lane_u8(paired64, 0) | ((uint32_t)vgetq_lane_u8(paired64, 8) << 8);
}

// inspirit 的回答 (1 vote): 标准实现了 _mm_movemask_epi8，但分了上下半边，指令很多
inline uint32_t vmovemask_u8_inspirit(uint8x16_t input)
{
    const int8_t __attribute__ ((aligned (16))) xr[8] = {-7,-6,-5,-4,-3,-2,-1,0};
    uint8x8_t mask_and = vdup_n_u8(0x80);
    int8x8_t mask_shift = vld1_s8(xr);

    uint8x8_t lo = vget_low_u8(input);
    uint8x8_t hi = vget_high_u8(input);

    lo = vand_u8(lo, mask_and);
    lo = vshl_u8(lo, mask_shift);

    hi = vand_u8(hi, mask_and);
    hi = vshl_u8(hi, mask_shift);

    lo = vpadd_u8(lo,lo);
    lo = vpadd_u8(lo,lo);
    lo = vpadd_u8(lo,lo);

    hi = vpadd_u8(hi,hi);
    hi = vpadd_u8(hi,hi);
    hi = vpadd_u8(hi,hi);

    return ((hi[0] << 8) | (lo[0] & 0xFF));
}

// （可能是）指令数最少的实现，要求输入的每个 8 bits 全 0 或全 1
inline uint32_t vmovemask_u8_solrex(uint8x16_t a) {
    // 先取出相邻两个 uint8 的中间 2 bits，1 bit 属于高 uint8，1 bit 属于低 uint8
    uint16x8_t MASK =  vdupq_n_u16(0x180);
    uint16x8_t a_masked = vandq_u16(vreinterpretq_u16_u8(a), MASK);
    // 再将这 8 个 2 bits 按照不同的偏移进行 SHIFT，使得它们加一起能表示最终的 mask
    const int16_t __attribute__ ((aligned (16))) SHIFT_ARR[8]= {-7, -5, -3, -1, 1, 3, 5, 7};
    int16x8_t SHIFT = vld1q_s16(SHIFT_ARR);
    uint16x8_t a_shifted = vshlq_u16(a_masked, SHIFT);
    // 最后把这 8 个数字加起来
    return vaddvq_u16(a_shifted);
}

inline uint32_t vmovemask_u8_webasm(uint8x16_t a) {
    static const uint8x16_t mask = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    uint8x16_t masked = vandq_u8(mask, (uint8x16_t)vshrq_n_s8(a, 7));
    uint8x16_t maskedhi = vextq_u8(masked, masked, 8);
    return vaddvq_u16((uint16x8_t)vzip1q_u8(masked, maskedhi));
}

// 基准测试：重复处理单变量
template <typename Func>
std::tuple<double, uint64_t> benchmark_repeate(Func func, uint8x16_t &input, size_t iterations) {
    uint64_t acc = 0; // 避免循环被优化，也可用于检查正确性
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        acc += func(input);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> elapsed = end - start;
    return {elapsed.count() / iterations, acc};
}

// 基准测试：按序处理数组变量
template <typename Func>
std::tuple<double, uint64_t> benchmark_array(Func func, const std::vector<uint8x16_t>& data) {
    uint64_t acc = 0; // 避免循环被优化，也可用于检查正确性
    auto start = std::chrono::high_resolution_clock::now();
    for (const auto& item : data) {
        acc += func(item);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> elapsed = end - start;
    return {elapsed.count() / data.size(), acc};
}

int main() {
    const int iterations = 1000000;     // 重复次数
    // 生成随机数组
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dis(0, 1);

    std::vector<uint8x16_t> array(iterations);
    for (auto& vec : array) {
        uint8_t arr[16];
        for (int i = 0; i < 16; ++i) {
            arr[i] = dis(gen) ? 0xFF : 0;
        }
        vec = vld1q_u8(arr);
    }
    using Func = uint32_t(uint8x16_t);
    std::vector<std::pair<Func*, std::string>> functions = {
        {vmovemask_u8_YvesDaoust, "vmovemask_u8_YvesDaoust()"},
        {vmovemask_u8_David, "vmovemask_u8_David()"},
        {vmovemask_u8_EasyasPi, "vmovemask_u8_EasyasPi()"},
        {vmovemask_u8_inspirit, "vmovemask_u8_inspirit()"},
        {vmovemask_u8_solrex, "vmovemask_u8_solrex()"},
        {vmovemask_u8_webasm, "vmovemask_u8_webasm()"}
    };
    // 对每个函数进行基准测试
    for (const auto& [func, name] : functions) {
        auto [avg_time_repeate, acc_repeate] = benchmark_repeate(func, array[0], iterations);
        auto [avg_time_array, acc_array] = benchmark_array(func, array);
        std::cout << name << "\t" << avg_time_repeate << " ns\t" << avg_time_array << " ns\n";
        //std::cout << name << "\t" << acc_repeate << "\t" << acc_array << "\n";
    }
    return 0;
}

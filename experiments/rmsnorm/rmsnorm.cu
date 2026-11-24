#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CHECK(call)                                                   \
    do {                                                              \
        cudaError_t err = (call);                                      \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,          \
                    cudaGetErrorString(err));                          \
            std::exit(1);                                             \
        }                                                             \
    } while (0)

constexpr int H = 4096;
constexpr float EPS = 1e-6f;

// 所有参与的 Warp 都是完整的 32 线程 Warp。
__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

// KEEP=false：计算输出时重新读取输入。
// KEEP=true ：尝试将每线程负责的输入保留在寄存器中。
// 实际是否全部进入寄存器，由编译器决定，必须检查资源报告。
template<int THREADS, bool KEEP>
__global__ void rmsnorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    float* __restrict__ y
) {
    constexpr int WARPS = THREADS / 32;
    constexpr int ITEMS = H / THREADS;

    // 动态共享内存的前 WARPS 个 float 存放各 Warp 的部分和。
    // 启动时额外申请的部分只用于制造资源压力，不参与计算。
    extern __shared__ float shared[];

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const size_t base = static_cast<size_t>(blockIdx.x) * H;

    float saved[ITEMS];
    float sum = 0.0f;

    #pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
        const int col = tid + k * THREADS;
        const float value = x[base + col];

        if constexpr (KEEP) {
            saved[k] = value;
        }

        sum += value * value;
    }

    // 第一级：Warp 内求和。
    sum = warp_sum(sum);

    if (lane == 0) {
        shared[warp] = sum;
    }
    __syncthreads();

    // 第二级：第一个 Warp 汇总所有 Warp 的部分和。
    if (warp == 0) {
        float total = lane < WARPS ? shared[lane] : 0.0f;
        total = warp_sum(total);

        if (lane == 0) {
            shared[0] = rsqrtf(total / H + EPS);
        }
    }
    __syncthreads();

    const float inv_rms = shared[0];

    #pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
        const int col = tid + k * THREADS;

        float value;
        if constexpr (KEEP) {
            value = saved[k];
        } else {
            value = x[base + col];
        }

        y[base + col] = value * inv_rms * gamma[col];
    }
}

template<int THREADS, bool KEEP>
void run(
    int rows,
    int iterations,
    size_t extra_shared,
    const std::vector<float>& reference,
    const float* dx,
    const float* dg,
    float* dy
) {
    const size_t shared_bytes =
        (THREADS / 32) * sizeof(float) + extra_shared;

    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, 0));

    cudaFuncAttributes attr{};
    CHECK(cudaFuncGetAttributes(
        &attr, rmsnorm_kernel<THREADS, KEEP>));
    

    // 本实验不使用超过默认单块共享内存上限的配置。
    if (shared_bytes + attr.sharedSizeBytes >
        prop.sharedMemPerBlock) {
        fprintf(stderr, "Requested shared memory exceeds default limit\n");
        std::exit(1);
    }

    int resident_blocks = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_blocks,
        rmsnorm_kernel<THREADS, KEEP>,
        THREADS,
        shared_bytes));

    const double theoretical_occupancy =
        100.0 * resident_blocks * THREADS /
        prop.maxThreadsPerMultiProcessor;

    auto launch = [&]() {
        rmsnorm_kernel<THREADS, KEEP>
            <<<rows, THREADS, shared_bytes>>>(dx, dg, dy);
    };

    // 固定 10 次预热。ncu 命令将跳过这 10 次。
    for (int i = 0; i < 10; ++i) {
        launch();
    }
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        launch();
    }
    CHECK(cudaEventRecord(stop));
    CHECK(cudaGetLastError());
    CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    std::vector<float> output(reference.size());
    CHECK(cudaMemcpy(
        output.data(), dy, output.size() * sizeof(float),
        cudaMemcpyDeviceToHost));

    double max_abs_error = 0.0;
    size_t bad = 0;

    for (size_t i = 0; i < output.size(); ++i) {
        const double err =
            std::abs(static_cast<double>(output[i]) - reference[i]);

        max_abs_error = std::max(max_abs_error, err);

        const double tolerance =
            2e-5 + 2e-5 * std::abs(reference[i]);

        if (!std::isfinite(output[i]) || err > tolerance) {
            ++bad;
        }
    }

    printf("M=%d H=%d threads=%d mode=%s iterations=%d\n",
        rows, H, THREADS, KEEP ? "keep" : "reload", iterations);

    // printf("GPU: %s, SM count: %d, sharedMem/Block=%zu, sharedMem/SM=%zu\n, "
    //     "maxThreads/Block=%d, regs/Block=%d, totalGlobalMem=%.1f GiB, "
    //     "maxThreads/SM=%d\n\n",
    //     prop.name, prop.multiProcessorCount,
    //     prop.sharedMemPerBlock, prop.sharedMemPerMultiprocessor,
    //     prop.maxThreadsPerBlock, prop.regsPerBlock,
    //     prop.totalGlobalMem / (1024.0 * 1024 * 1024),
    //     prop.maxThreadsPerMultiProcessor);

    printf("registers/thread=%d local_bytes/thread=%zu "
           "shared_bytes/block=%zu, static_shared_bytes/block=%zu\n",
           attr.numRegs, attr.localSizeBytes,
           shared_bytes, attr.sharedSizeBytes);

    printf("max_resident_blocks/SM=%d "
           "theoretical_occupancy=%.1f%%\n",
           resident_blocks, theoretical_occupancy);

    printf("average_us=%.3f max_abs_error=%.8g check=%s\n",
           elapsed_ms * 1000.0 / iterations,
           max_abs_error, bad == 0 ? "PASS" : "FAIL");

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    if (bad != 0) {
        fprintf(stderr, "Incorrect elements: %zu\n", bad);
        std::exit(2);
    }
}

template<int THREADS>
void select_mode(
    bool keep, int rows, int iterations, size_t extra_shared,
    const std::vector<float>& reference,
    const float* dx, const float* dg, float* dy
) {
    if (keep) {
        run<THREADS, true>(
            rows, iterations, extra_shared, reference, dx, dg, dy);
    } else {
        run<THREADS, false>(
            rows, iterations, extra_shared, reference, dx, dg, dy);
    }
}

int main(int argc, char** argv) {
    if (argc != 6) {
        printf("Usage: %s M threads keep extra_shared_KiB iterations\n",
               argv[0]);
        printf("Example: %s 4096 256 0 0 1000\n", argv[0]);
        return 1;
    }

    const int rows = std::atoi(argv[1]);
    const int threads = std::atoi(argv[2]);
    const int keep = std::atoi(argv[3]);
    const int extra_kib = std::atoi(argv[4]);
    const int iterations = std::atoi(argv[5]);

    if (rows <= 0 || rows > 65536 ||
        (threads != 128 && threads != 256 && threads != 512) ||
        (keep != 0 && keep != 1) ||
        extra_kib < 0 || extra_kib > 32 || iterations <= 0) {
        fprintf(stderr, "Invalid arguments\n");
        return 1;
    }

    CHECK(cudaSetDevice(0));

    const size_t n = static_cast<size_t>(rows) * H;
    const size_t bytes = n * sizeof(float);

    std::vector<float> x(n), gamma(H), reference(n);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (float& v : x) {
        v = dist(rng);
    }
    for (float& v : gamma) {
        v = 1.0f + 0.1f * dist(rng);
    }

    // CPU 参考实现：使用 double 计算平方和。
    for (int row = 0; row < rows; ++row) {
        const size_t base = static_cast<size_t>(row) * H;

        double sum = 0.0;
        for (int col = 0; col < H; ++col) {
            const double v = x[base + col];
            sum += v * v;
        }

        const double inv_rms = 1.0 / std::sqrt(sum / H + EPS);

        for (int col = 0; col < H; ++col) {
            reference[base + col] =
                static_cast<float>(x[base + col] * inv_rms * gamma[col]);
        }
    }

    float *dx = nullptr, *dg = nullptr, *dy = nullptr;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&dx), bytes));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&dg), H * sizeof(float)));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&dy), bytes));

    CHECK(cudaMemcpy(dx, x.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(
        dg, gamma.data(), H * sizeof(float), cudaMemcpyHostToDevice));

    const size_t extra_shared = static_cast<size_t>(extra_kib) * 1024;

    switch (threads) {
        case 128:
            select_mode<128>(
                keep, rows, iterations, extra_shared,
                reference, dx, dg, dy);
            break;
        case 256:
            select_mode<256>(
                keep, rows, iterations, extra_shared,
                reference, dx, dg, dy);
            break;
        case 512:
            select_mode<512>(
                keep, rows, iterations, extra_shared,
                reference, dx, dg, dy);
            break;
    }

    CHECK(cudaFree(dx));
    CHECK(cudaFree(dg));
    CHECK(cudaFree(dy));
    return 0;
}

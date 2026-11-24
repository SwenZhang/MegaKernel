// gpu_info.cu
// 编译：nvcc gpu_info.cu -o gpu_info
// 运行：./gpu_info

#include <cuda_runtime.h>
#include <cstdio>

int main() {
    const int device = 0;
    cudaDeviceProp prop{};

    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "获取 GPU 信息失败：%s\n",
                     cudaGetErrorString(err));
        return 1;
    }

    std::printf("========== GPU %d ==========\n", device);
    std::printf("GPU 名称：                  %s\n", prop.name);
    std::printf("计算能力：                  %d.%d\n", prop.major, prop.minor);
    std::printf("SM 数量：                   %d\n", prop.multiProcessorCount);
    std::printf("Warp 线程数：               %d\n", prop.warpSize);

    std::printf("\n========== 显存与缓存 ==========\n");
    std::printf("全局显存容量：              %.2f GiB\n",
                prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    std::printf("显存位宽：                  %d bit\n", prop.memoryBusWidth);
    std::printf("L2 缓存容量：               %.2f MiB\n",
                prop.l2CacheSize / (1024.0 * 1024.0));
    std::printf("常量内存容量：              %.2f KiB\n",
                prop.totalConstMem / 1024.0);

    std::printf("\n========== 共享内存与寄存器 ==========\n");
    std::printf("每个 Block 默认共享内存上限：%.2f KiB\n",
                prop.sharedMemPerBlock / 1024.0);
    std::printf("每个 Block opt-in 共享内存上限：%.2f KiB\n",
                prop.sharedMemPerBlockOptin / 1024.0);
    std::printf("每个 SM 共享内存容量：      %.2f KiB\n",
                prop.sharedMemPerMultiprocessor / 1024.0);
    std::printf("每个 Block 的寄存器上限：   %d 个（32 bit）\n",
                prop.regsPerBlock);
    std::printf("每个 SM 的寄存器数量：      %d 个（32 bit）\n",
                prop.regsPerMultiprocessor);

    std::printf("\n========== 线程与网格限制 ==========\n");
    std::printf("每个 Block 最大线程数：     %d\n",
                prop.maxThreadsPerBlock);
    std::printf("每个 SM 最大驻留线程数：    %d\n",
                prop.maxThreadsPerMultiProcessor);
    std::printf("每个 SM 最大驻留 Warp 数：  %d\n",
                prop.maxThreadsPerMultiProcessor / prop.warpSize);
    std::printf("每个 SM 最大驻留 Block 数： %d\n",
                prop.maxBlocksPerMultiProcessor);
    std::printf("Block 各维度上限：          (%d, %d, %d)\n",
                prop.maxThreadsDim[0],
                prop.maxThreadsDim[1],
                prop.maxThreadsDim[2]);
    std::printf("Grid 各维度上限：           (%d, %d, %d)\n",
                prop.maxGridSize[0],
                prop.maxGridSize[1],
                prop.maxGridSize[2]);

    std::printf("\n========== 功能支持 ==========\n");
    std::printf("支持并发 Kernel：           %s\n",
                prop.concurrentKernels ? "是" : "否");
    std::printf("异步复制引擎数量：          %d\n", prop.asyncEngineCount);
    std::printf("支持统一虚拟地址：          %s\n",
                prop.unifiedAddressing ? "是" : "否");
    std::printf("支持映射主机内存：          %s\n",
                prop.canMapHostMemory ? "是" : "否");
    std::printf("支持协作式 Kernel 启动：    %s\n",
                prop.cooperativeLaunch ? "是" : "否");
    std::printf("ECC 是否启用：              %s\n",
                prop.ECCEnabled ? "是" : "否");

    return 0;
}
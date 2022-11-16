
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <chrono>
#include <functional>

void substractWithCuda(int* c, const int* a, const int* b, uint32_t size);

void subsractCPU(int* c, const int* a, const int* b, uint32_t arraySize)
{
    for (size_t i = 0; i < arraySize; i++)
    {
        c[i] = a[i] - b[i];
    }
}

__global__ void substractKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] - b[i];
}

void calculateFunctionTime(int* c, const int* a, const int* b, uint32_t size,
            std::function<void(int* c, const int*, const int*, uint32_t)> substract)
{
    using namespace std::chrono;

    auto start = high_resolution_clock::now();

    substract(c, a, b, size);

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(stop - start);

    std::cout << "Time taken by CPU function: "
        << duration.count() << " milliseconds" << std::endl;
}

template<typename FUNC, typename... Args>
void checkStatus(FUNC func, Args... args) {
    cudaError_t status = func(args...);
    if (status != cudaSuccess)
    {
        std::cerr << "Error! ";
        std::cerr << cudaGetErrorString(status) << std::endl;
        std::exit(-1);
    }
}

int main()
{
    constexpr uint32_t arraySize = 5;
    const int a[arraySize] = { 10, 20, 30, 40, 50 };
    const int b[arraySize] = { 5, 10, 15, 20, 25 };
    int c1[arraySize] = { 0 };
    int c2[arraySize] = { 0 };


    calculateFunctionTime(c1, a, b, arraySize, &subsractCPU);
    printf("{10,20,30,40,50} - {5,10,15,20,25} = {%d,%d,%d,%d,%d}\n",
        c1[0], c1[1], c1[2], c1[3], c1[4]);

    substractWithCuda(c2, a, b, arraySize);
    printf("{10,20,30,40,50} - {5,10,15,20,25} = {%d,%d,%d,%d,%d}\n",
        c2[0], c2[1], c2[2], c2[3], c2[4]);

    cudaError_t cudaStatus = cudaDeviceReset();

    return 0;
}

// Helper function for using CUDA to substract vectors in parallel.
void substractWithCuda(int* c, const int* a, const int* b, uint32_t size)
{
    using namespace std::chrono;

    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    auto checkError = [](cudaError_t status)
    {
        if (status != cudaSuccess)
        {
            std::cerr << "Error! ";
            std::cerr << cudaGetErrorString(status) << std::endl;
            std::exit(-1);
        }
    };

    // Choose which GPU to run on, change this on a multi-GPU system.
    checkStatus(cudaSetDevice, 0);

    // инициализируем события
    cudaEvent_t start, stop;
    float elapsedTime;
    // создаем события
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // запись события
    cudaEventRecord(start, 0);

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    checkError(cudaStatus);
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    checkError(cudaStatus);
    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    checkError(cudaStatus);

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    checkError(cudaStatus);
    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    checkError(cudaStatus);

    // Launch a kernel on the GPU with one thread for each element.
    substractKernel<<<1, size >>> (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    checkError(cudaStatus);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    checkError(cudaStatus);

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    checkError(cudaStatus);

    cudaStatus = cudaEventRecord(stop, 0);
    checkError(cudaStatus);
    cudaStatus = cudaEventSynchronize(stop);
    checkError(cudaStatus);
    cudaStatus = cudaEventElapsedTime(&elapsedTime, start, stop);
    checkError(cudaStatus);

    // вывод информации
    printf("Time spent executing by the GPU: %.2f milliseconds\n", elapsedTime);

    // Free resources.
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
}

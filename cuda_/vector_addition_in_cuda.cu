#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <ctime>


__global__ void add(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    // Generate input vectors h_a and h_b
    int n = 100 * 1000 * 1000;
    int *h_a = new int[n];
    int *h_b = new int[n];
    int *h_c = new int[n];
    int *final_result = new int[n];

    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i;
    }
    int cpu_start_time = clock();
    
    for (int i = 0; i < n; i++) {
        h_c[i] = h_a[i] + h_b[i];
    }
    int cpu_end_time = clock();
    // Convert cpu time to ms
    long cpu_time_used = (cpu_end_time - cpu_start_time) / double(CLOCKS_PER_SEC) * 1000;
    std::cout << "CPU time: " << cpu_time_used << " ms" << std::endl;

    // Declare GPU memory pointers
    int *d_a, *d_b, *d_c;
    // Create GPU memory space
    cudaMalloc((void **)&d_a, n * sizeof(int));
    cudaMalloc((void **)&d_b, n * sizeof(int));
    cudaMalloc((void **)&d_c, n * sizeof(int));

    int gpu_start_time = clock();
    // Transfer input vectors from host to GPU memory
    cudaMemcpy(d_a, h_a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(int), cudaMemcpyHostToDevice);

    // Invoke kernel
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    int gpu_execution_time = clock();
    add<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    int gpu_execution_end_time = clock();
    // Transfer result from GPU memory to host
    cudaMemcpy(final_result, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    int gpu_end_time = clock();
    // Convert gpu time to ms
    long gpu_time_used = (gpu_end_time - gpu_start_time) / double(CLOCKS_PER_SEC) * 1000;
    long gpu_execution_time_used = (gpu_execution_end_time - gpu_execution_time) / double(CLOCKS_PER_SEC) * 1000 * 1000;
    std::cout << "GPU time: " << gpu_time_used << " ms" << std::endl;
    std::cout << "GPU execution time: " << gpu_execution_time_used << " us" << std::endl;

    // Compare final result with CPU result
    for (int i = 0; i < n; i++) {
        if (final_result[i] != h_c[i]) {
            std::cout << "Error: " << i << std::endl;
            break;
        }
    }

    // Free GPU memory space
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
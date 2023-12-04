#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <ctime>
#include <cassert>


__global__ void add(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

__global__ void multiply(int *A, int *B, int *C, int height1, int width1, int height2, int width2) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    assert (width1 == height2);
    int sum = 0;
    if (row >= height1 || col >= width2) {
        return;
    }
    for(int i = 0; i < width1; i++) {
        sum += A[row * width1 + i] * B[i * width2 + col];
    }
    C[row * width2 + col] = sum;

}
void cpu_multiply(int *A, int *B, int *C, int height1, int width1, int height2, int width2) {
    for(int i = 0; i < height1; i++) {
        for(int j = 0; j < width2; j++) {
            int sum = 0;
            for(int k = 0; k < width1; k++) {
                sum += A[i * width1 + k] * B[k * width2 + j];
            }
            C[i * width2 + j] = sum;
        }
    }

}

int main() {
    // Generate input vectors h_a and h_b
    int p = 2048, q = 4096, r = 2048;
    int *h_a = new int[p*q];
    int *h_b = new int[q*r];
    int *h_c = new int[p*r];
    int *final_result = new int[p*r];

    for (int i = 0; i < p*q; i++) {
        h_a[i] = i;
    }
    for(int i=0;i<q*r;i++) {
        h_b[i] = i;
    }
    int cpu_start_time = clock();
    
    cpu_multiply(h_a, h_b, h_c, p, q, q, r);
    int cpu_end_time = clock();
    // Convert cpu time to ms
    long cpu_time_used = (cpu_end_time - cpu_start_time) / double(CLOCKS_PER_SEC) * 1000;
    std::cout << "CPU time: " << cpu_time_used << " ms" << std::endl;

    // Declare GPU memory pointers
    int *d_a, *d_b, *d_c;
    // Create GPU memory space
    cudaMalloc((void **)&d_a, p*q * sizeof(int));
    cudaMalloc((void **)&d_b, q*r * sizeof(int));
    cudaMalloc((void **)&d_c, p*r * sizeof(int));

    int gpu_start_time = clock();
    // Transfer input vectors from host to GPU memory
    cudaMemcpy(d_a, h_a, p*q * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, q*r * sizeof(int), cudaMemcpyHostToDevice);

    // Invoke kernel
    int block_size = 32;
    dim3 dimBlock(block_size, block_size);
    // int grid_size = (n + block_size - 1) / block_size;
    dim3 dimGrid((r + dimBlock.x - 1) / dimBlock.x, (p + dimBlock.y - 1) / dimBlock.y);
    int gpu_execution_time = clock();
    multiply<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, p, q, q, r);
    int gpu_execution_end_time = clock();
    // Transfer result from GPU memory to host
    cudaMemcpy(final_result, d_c, p*r * sizeof(int), cudaMemcpyDeviceToHost);
    int gpu_end_time = clock();
    // Convert gpu time to ms
    long gpu_time_used = (gpu_end_time - gpu_start_time) / double(CLOCKS_PER_SEC) * 1000;
    long gpu_execution_time_used = (gpu_execution_end_time - gpu_execution_time) / double(CLOCKS_PER_SEC) * 1000 * 1000;
    std::cout << "GPU time: " << gpu_time_used << " ms" << std::endl;
    std::cout << "GPU execution time: " << gpu_execution_time_used << " us" << std::endl;

    // Compare final result with CPU result
    for (int i = 0; i < p*r; i++) {
        if (final_result[i] != h_c[i]) {
            std::cout << "Error: " << i  << final_result[i] << " " << h_c[i] << " "<< std::endl;
            break;
        }
    }

    // Free GPU memory space
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
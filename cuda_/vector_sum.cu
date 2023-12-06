#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <ctime>

using namespace std;


__global__ void sum(float *a, float *cum_array, int n) {
    __shared__ float temp[1024];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        temp[tid] = a[i];
    } else {
        temp[tid] = 0;
    }
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            temp[tid] += temp[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        cum_array[blockIdx.x] = temp[0];
    }
}

int main() {
    int n = 50*1000*1000;
    float *a = new float[n];
    float *cum_array = new float[n];
    int block_size = 1024;
    int num_blocks = (n + block_size - 1) / block_size;
    float *d_a, *d_cum_array;
    cudaMalloc((void**)&d_a, n*sizeof(float));
    cudaMalloc((void**)&d_cum_array, num_blocks*sizeof(float));
    for (int i = 0; i < n; i++) {
        a[i] = i;
    }
    for(int i=0;i<num_blocks;i++) {
        cum_array[i] = 0;
    }

    // Copy to device
    cudaMemcpy(d_a, a, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cum_array, cum_array, num_blocks*sizeof(float), cudaMemcpyHostToDevice);

    // Measure time
    clock_t start, end;
    start = clock();

    // Run kernel
    sum<<<num_blocks, block_size>>>(d_a, d_cum_array, n);
    cudaMemcpy(cum_array, d_cum_array, num_blocks*sizeof(float), cudaMemcpyDeviceToHost);
    float result = 0;
    for(int i=0; i<num_blocks; i++) {
        result += cum_array[i];
    }
    end = clock();
    cout << "Time: " << (double)(end - start) / CLOCKS_PER_SEC / 1000.0/1000.0 << " us" << endl;
    cout << "Result by GPU: " << result << endl;
    result = 0;
    start = clock();
    for(int i=0; i<n; i++) {
        result += a[i];
    }
    end = clock();
    cout << "Result by CPU: " << result << endl;
    cout << "Time: " << (double)(end - start) / CLOCKS_PER_SEC / 1000.0/1000.0 << " us" << endl;
}
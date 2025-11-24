#include <cuda_runtime.h>
#include <math.h>  // for tanhf

__global__ void bias_gelu_kernel(const float* x,
                                 const float* bias,
                                 float* out,
                                 int batch,
                                 int hidden) {
    // 
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int total = batch * hidden;

    // loop over all idxs', conditioned upon i < total data, incremented by total thread count
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) { 
        int j = i % hidden;
        float v = x[i] + bias[j];
        float c = 0.79788456f * (v + 0.044715f * v * v * v);
        float gelu = 0.5f * v * (1.0f + tanhf(c));
        out[i] = gelu;
    }
}

void bias_gelu_forward_cuda(const float* x, const float* bias, float* out,
                            int batch, int hidden) {
    int threads = 256;
    int blocks = (batch * hidden + threads - 1) / threads; 
    bias_gelu_kernel<<<blocks, threads>>>(x, bias, out, batch, hidden);
}

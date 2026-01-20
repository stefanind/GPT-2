// cpp file to bridge between pytorch and CUDA

#include <torch/extension.h>   // pytorch C++ API

// declare the CUDA kernel (defined in bias_gelu_kernel.cu)
void bias_gelu_forward_cuda(
    const float* x,
    const float* bias,
    float* out,
    int batch,
    int hidden
);

// C++ function called from python
torch::Tensor bias_gelu_forward(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "x must be on CUDA");
    TORCH_CHECK(bias.is_cuda(), "bias must on CUDA");

    auto batch = x.size(0);
    auto hidden = x.size(1);

    auto out = torch::empty_like(x);

    // kernel wrapper
    bias_gelu_forward_cuda(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        batch,
        hidden
    );

    return out;
}

// bind this function so python can import it
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &bias_gelu_forward, "Fused Bias+GELU forward (CUDA)");
}

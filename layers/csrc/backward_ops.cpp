#include <torch/extension.h>
#include <ATen/ATen.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cudnn_convolution_backward_input", &at::cudnn_convolution_backward_input,
        "cudnn_convolution_backward_input");
  m.def("cudnn_convolution_backward_weight", &at::cudnn_convolution_backward_weight,
        "cudnn_convolution_backward_weight");
}
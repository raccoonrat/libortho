/*
 * libortho - PyBind11 Bindings
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "../include/ortho.h"

namespace py = pybind11;

PYBIND11_MODULE(_C_ops, m) {
    m.doc() = "libortho C++/CUDA operations";
    
    // Expose orth_layer_t structure
    py::class_<orth_layer_t>(m, "OrthLayer")
        .def(py::init<>())
        .def_readwrite("alpha", &orth_layer_t::alpha)
        .def("set_alpha", &orth_layer_set_alpha)
        .def("forward", [](const orth_layer_t &layer, 
                          py::array_t<float> input) {
            // Forward pass wrapper
            auto buf = input.request();
            float *input_ptr = static_cast<float *>(buf.ptr);
            
            size_t batch_size = buf.shape[0];
            size_t in_features = buf.shape[1];
            
            py::array_t<float> output({batch_size, layer.base.out_features});
            auto output_buf = output.request();
            float *output_ptr = static_cast<float *>(output_buf.ptr);
            
            int ret = orth_layer_forward(&layer, input_ptr, output_ptr, batch_size);
            if (ret != 0) {
                throw std::runtime_error("Forward pass failed");
            }
            
            return output;
        });
    
    // Expose CUDA forward function if available
    #ifdef __CUDACC__
    m.def("forward_cuda", &orth_layer_forward_cuda, 
          "CUDA-accelerated forward pass");
    #endif
}


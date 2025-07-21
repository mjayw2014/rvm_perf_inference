import tensorrt as trt
from cuda import cuda
import os
import numpy as np
from cuda_helper import checkCudaErrors

trt_logger = trt.Logger()

def load_engine(model_file):
    assert os.path.exists(model_file)
    trt_runtime = trt.Runtime(trt_logger)
    with open(model_file, "rb") as f:
        return trt_runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(trt_engine):
    inputs   = {}
    outputs  = {}
    bindings = []

    for i in range(trt_engine.num_io_tensors):
        tensor_name = trt_engine.get_tensor_name(i)
        shape = trt_engine.get_tensor_shape(tensor_name)
        dtype = trt_engine.get_tensor_dtype(tensor_name)
        size  = np.prod(shape) * dtype.itemsize

        #allocate tensor on GPU
        d_memory = checkCudaErrors(cuda.cuMemAlloc(size))
        bindings.append(d_memory)

        if trt_engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            inputs[tensor_name] = {'d_mem':d_memory, 'shape': shape, 'size': size, 'dtype': dtype}
        else:
            outputs[tensor_name] = {'d_mem': d_memory, 'shape': shape, 'size': size, 'dtype': dtype}

    return inputs, outputs, bindings
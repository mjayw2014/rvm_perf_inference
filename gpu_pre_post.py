import ctypes
import time
import cv2
import tensorrt as trt
from cuda import cuda
import os
import numpy as np
from cuda_helper import checkCudaErrors, findCudaDeviceDRV, KernelHelper
from cuda_kernels import cuda_pre_post_kernels

#Initialize cuda
checkCudaErrors(cuda.cuInit(0))
cudaDevice = findCudaDeviceDRV()
cuda_context = checkCudaErrors(cuda.cuCtxCreate(0, cudaDevice))

def gpu_pre_post_inference(inputs, outputs, bindings, trt_engine, trt_context, input_video_file):

    cap = cv2.VideoCapture(input_video_file)
    frame_count = 0

    _, frame_channel, frame_height, frame_width = inputs['src']['shape']

    d_frame_size = frame_channel * frame_height * frame_width * np.dtype(np.uint8).itemsize
    d_in_frame  = checkCudaErrors(cuda.cuMemAlloc(d_frame_size))
    d_out_frame = checkCudaErrors(cuda.cuMemAlloc(d_frame_size))

    #Initialize pre and post processing CUDA kernels
    cuda_block_dim = (1024,1,1)
    cuda_grid_dim  = ((frame_height * frame_width) // 1024, 1, 1)

    cudaKernelHandle = KernelHelper(cuda_pre_post_kernels, int(cudaDevice))
    cuda_pre_process_kernel  = cudaKernelHandle.getFunction(b'pre_process')
    cuda_post_process_kernel = cudaKernelHandle.getFunction(b'post_process')

    kernel_args_pre_process  = ((d_in_frame, inputs['src']['d_mem'], np.int32(frame_height * frame_width), np.int32(0)),
                               (None, None, ctypes.c_int, ctypes.c_int))

    kernel_args_post_process = ((outputs['fgr']['d_mem'], outputs['pha']['d_mem'], d_out_frame, np.int32(frame_height * frame_width)),
                               (None, None, None, ctypes.c_int))
    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if 0 == frame_count:
            for i in range(4):
                tmp_zeros = np.zeros((inputs['r' + str(i + 1) + 'i']['shape']), dtype=np.float32)
                # copy the 'ri' tensors to GPU
                checkCudaErrors(cuda.cuMemcpyHtoD(inputs['r' + str(i + 1) + 'i']['d_mem'],
                                                  np.ascontiguousarray(tmp_zeros),
                                                  inputs['r' + str(i + 1) + 'i']['size']))

        #Copy frame to GPU
        checkCudaErrors(cuda.cuMemcpyHtoD(d_in_frame, frame, d_frame_size))

        #Launch pre-preocessing kernel
        checkCudaErrors(cuda.cuLaunchKernel(cuda_pre_process_kernel,
                                            cuda_grid_dim[0], cuda_grid_dim[1], cuda_grid_dim[2],
                                            cuda_block_dim[0], cuda_block_dim[1], cuda_block_dim[2],
                                            0, 0,
                                            kernel_args_pre_process, 0))

        #Launch TensortRT Inference
        trt_context.execute_v2(bindings)

        #Launch post-process kernel: Apply mask and convert uint8 image
        checkCudaErrors(cuda.cuLaunchKernel(cuda_post_process_kernel,
                                            cuda_grid_dim[0], cuda_grid_dim[1], cuda_grid_dim[2],
                                            cuda_block_dim[0], cuda_block_dim[1], cuda_block_dim[2],
                                            0, 0,
                                            kernel_args_post_process, 0))

        #Copy frame to CPU
        checkCudaErrors(cuda.cuMemcpyDtoH(frame, d_out_frame, d_frame_size))

        # cv2.imshow("Frame", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        frame_count += 1

    end_time = time.time()
    total_time = end_time - start_time
    fps = frame_count / total_time
    print("FPS = ", fps)

    cuda.cuMemFree(d_out_frame)
    cuda.cuMemFree(d_in_frame)
    for _mem in inputs.keys():
        cuda.cuMemFree(inputs[_mem]['d_mem'])

    for _mem in outputs.keys():
        cuda.cuMemFree(outputs[_mem]['d_mem'])

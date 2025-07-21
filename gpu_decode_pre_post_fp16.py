import ctypes
import time
import cv2
import tensorrt as trt
from cuda import cuda
import os
import numpy as np
from cuda_helper import checkCudaErrors, findCudaDeviceDRV, KernelHelper
from cuda_kernels import cuda_pre_post_kernels_fp16
import PyNvVideoCodec as nvc

#Initialize cuda
checkCudaErrors(cuda.cuInit(0))
cudaDevice = findCudaDeviceDRV()
cuda_context = checkCudaErrors(cuda.cuCtxCreate(0, cudaDevice))

def gpu_decode_pre_post_inference_fp16(inputs, outputs, bindings, trt_context, input_video_file):

    decoder = nvc.SimpleDecoder(enc_file_path=input_video_file,
                      gpu_id=cudaDevice,
                      use_device_memory=True,
                      need_scanned_stream_metadata=True,
                      output_color_type = nvc.OutputColorType.RGBP)


    frame_count = 0
    _, frame_channel, frame_height, frame_width = inputs['src']['shape']

    d_frame_size = frame_channel * frame_height * frame_width * np.dtype(np.uint8).itemsize
    d_out_frame = checkCudaErrors(cuda.cuMemAlloc(d_frame_size))
    h_out_frame = np.empty((frame_height, frame_width, frame_channel), dtype=np.uint8)

    # Initialize pre and post processing CUDA kernels
    cuda_block_dim = (1024, 1, 1)
    cuda_grid_dim = ((frame_height * frame_width) // 1024, 1, 1)

    cudaKernelHandle = KernelHelper(cuda_pre_post_kernels_fp16, int(cudaDevice))
    cuda_pre_process_kernel = cudaKernelHandle.getFunction(b'pre_process_fp16')
    cuda_post_process_kernel = cudaKernelHandle.getFunction(b'post_process_fp16')

    kernel_args_post_process = ((outputs['fgr']['d_mem'], outputs['pha']['d_mem'], d_out_frame, np.int32(frame_height * frame_width)),
                                (None, None, None, ctypes.c_int))
    start_time = time.time()

    for frame in decoder:
        r_plane_ptr = frame.GetPtrToPlane(0)

        kernel_args_pre_process = ((cuda.CUdeviceptr(r_plane_ptr), inputs['src']['d_mem'], np.int32(frame_height * frame_width)),
                                    (None, None, ctypes.c_int))

        # Launch pre-preocessing kernel
        checkCudaErrors(cuda.cuLaunchKernel(cuda_pre_process_kernel,
                                            cuda_grid_dim[0], cuda_grid_dim[1], cuda_grid_dim[2],
                                            cuda_block_dim[0], cuda_block_dim[1], cuda_block_dim[2],
                                            0, 0,
                                            kernel_args_pre_process, 0))

        # Launch TensortRT Inference
        trt_context.execute_v2(bindings)



        # Launch post-process kernel: Apply mask and convert uint8 image
        checkCudaErrors(cuda.cuLaunchKernel(cuda_post_process_kernel,
                                            cuda_grid_dim[0], cuda_grid_dim[1], cuda_grid_dim[2],
                                            cuda_block_dim[0], cuda_block_dim[1], cuda_block_dim[2],
                                            0, 0,
                                            kernel_args_post_process, 0))

        # Copy frame to CPU
        checkCudaErrors(cuda.cuMemcpyDtoH(h_out_frame,
                                          d_out_frame,
                                          d_frame_size))

        # cv2.imshow("Frame", h_out_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        frame_count += 1

    end_time = time.time()
    total_time = end_time - start_time
    fps = frame_count / total_time
    print("FPS = ", fps)

    cuda.cuMemFree(d_out_frame)
    for _mem in inputs.keys():
        cuda.cuMemFree(inputs[_mem]['d_mem'])

    for _mem in outputs.keys():
        cuda.cuMemFree(outputs[_mem]['d_mem'])

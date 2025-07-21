import ctypes
import time
import cv2
import tensorrt as trt
from cuda import cuda
import os
import numpy as np
from cuda_helper import checkCudaErrors, findCudaDeviceDRV, KernelHelper
from cuda_kernels import cuda_pre_post_kernels, cuda_pre_post_kernels_fp16
import PyNvVideoCodec as nvc

#Initialize cuda
checkCudaErrors(cuda.cuInit(0))
cudaDevice = findCudaDeviceDRV()
cuda_context = checkCudaErrors(cuda.cuCtxCreate(0, cudaDevice))

def gpu_decode_pre_post_inference_batched_fp16(inputs, outputs, bindings, trt_context, input_video_file):

    frame_count = 0
    batch_size, frame_channel, frame_height, frame_width = inputs['src']['shape']

    decoder = nvc.ThreadedDecoder(enc_file_path=input_video_file,
                                  buffer_size=batch_size,
                                  cuda_context=0,
                                  cuda_stream=0,
                                  use_device_memory=True,
                                  output_color_type=nvc.OutputColorType.RGBP)


    d_frame_size_bytes = frame_channel * frame_height * frame_width * np.dtype(np.uint8).itemsize
    d_frame_batch_size_bytes = d_frame_size_bytes * batch_size
    d_out_frame = checkCudaErrors(cuda.cuMemAlloc(d_frame_batch_size_bytes))
    h_out_frame = np.empty((batch_size, frame_height, frame_width, frame_channel), dtype=np.uint8)

    cudaKernelHandle = KernelHelper(cuda_pre_post_kernels_fp16, int(cudaDevice))
    cuda_pre_process_kernel = cudaKernelHandle.getFunction(b'pre_process_fp16_batched')
    cuda_post_process_kernel = cudaKernelHandle.getFunction(b'post_process_fp16_batched')

    kernel_args_post_process = ((outputs['fgr']['d_mem'], outputs['pha']['d_mem'], d_out_frame, np.int32(frame_height), np.int32(frame_width), np.int32(frame_channel), np.int32(batch_size) ),
                                (None, None, None, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int))
    start_time = time.time()

    #Pre process Kernel setup
    cuda_pre_block_dim = (1024, 1, 1)
    cuda_pre_grid_dim = ((frame_height * frame_width) // 1024, 1, 1)

    # Pre process Kernel setup
    cuda_post_block_dim = (1024, 1, 1)
    cuda_post_grid_dim = ((frame_height * frame_width) // 1024, 1, 1)

    #Batch frame List

    d_frame_prt_array = None
    while True:
        frames = decoder.get_batch_frames(batch_size)
        if len(frames) == 0:
            break

        frame_ptr_list = []
        for i, frame in enumerate(frames):
            frame_device_ptr = int(frame.GetPtrToPlane(0))
            frame_ptr_list.append(frame_device_ptr)

        frame_ptr_array = np.array(frame_ptr_list, dtype=np.uint64)
        if d_frame_prt_array == None:
            d_frame_prt_array = checkCudaErrors(cuda.cuMemAlloc(frame_ptr_array.nbytes))

        # Copy frame list Device
        checkCudaErrors(cuda.cuMemcpyHtoD(d_frame_prt_array, frame_ptr_array, frame_ptr_array.nbytes))

        kernel_args_pre_process = ((d_frame_prt_array, inputs['src']['d_mem'], np.int32(frame_height), np.int32(frame_width), np.int32(frame_channel), np.int32(batch_size)),
                                    (None, None, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int))

        # Launch pre-processing kernel
        checkCudaErrors(cuda.cuLaunchKernel(cuda_pre_process_kernel,
                                            cuda_pre_grid_dim[0], cuda_pre_grid_dim[1], cuda_pre_grid_dim[2],
                                            cuda_pre_block_dim[0], cuda_pre_block_dim[1], cuda_pre_block_dim[2],
                                            0, 0,
                                            kernel_args_pre_process, 0))


        # Launch TensortRT Inference
        trt_context.execute_v2(bindings)

        # Launch post-process kernel: Apply mask and convert uint8 image
        checkCudaErrors(cuda.cuLaunchKernel(cuda_post_process_kernel,
                                            cuda_post_grid_dim[0], cuda_post_grid_dim[1], cuda_post_grid_dim[2],
                                            cuda_post_block_dim[0], cuda_post_block_dim[1], cuda_post_block_dim[2],
                                            0, 0,
                                            kernel_args_post_process, 0))

        # Copy frame to CPU
        checkCudaErrors(cuda.cuMemcpyDtoH(h_out_frame,
                                          d_out_frame,
                                          d_frame_size_bytes))

        # for j in range(len(frames)):
        #     #print(h_out_frame[j].shape)
        #     #cv2.imwrite(str(frame_count) + ".png", h_out_frame[j])
        #     cv2.imshow("Frame", h_out_frame[j])
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        frame_count += batch_size

    end_time = time.time()
    total_time = end_time - start_time
    fps = frame_count / total_time
    print("FPS = ", fps)

    cuda.cuMemFree(d_out_frame)
    for _mem in inputs.keys():
        cuda.cuMemFree(inputs[_mem]['d_mem'])

    for _mem in outputs.keys():
        cuda.cuMemFree(outputs[_mem]['d_mem'])

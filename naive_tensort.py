import utils
import cv2
import time
from cuda import cuda, cudart, nvrtc
from cuda_helper import checkCudaErrors
import numpy as np

def inference(inputs, outputs, bindings, trt_context, input_video_file):
    cap = cv2.VideoCapture(input_video_file)
    frame_count = 0
    out_mask = None

    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if 0 == frame_count:
            for i in range(4):
                tmp_zeros = np.zeros((inputs['r' + str(i+1) + 'i']['shape']), dtype=np.float32)
                #copy the 'ri' tensors to GPU
                checkCudaErrors(cuda.cuMemcpyHtoD(inputs['r' + str(i+1) + 'i']['d_mem'], np.ascontiguousarray(tmp_zeros), inputs['r' + str(i+1) + 'i']['size']))

            frame_h, frame_w, frame_c = frame.shape
            out_mask = np.empty((1, frame_h, frame_w)).astype(np.float32)

        # convert uint8 HWC frame to fp32 CHW tensor
        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))[np.newaxis, :]

        # Copy frame tensor to GPU
        checkCudaErrors(cuda.cuMemcpyHtoD(inputs['src']['d_mem'], np.ascontiguousarray(frame), inputs['src']['size']))

        # Execute TensorRT in engine
        trt_context.execute_v2(bindings)

        # Copy result frame (mask) to host
        checkCudaErrors(cuda.cuMemcpyDtoH(np.ascontiguousarray(out_mask), outputs['pha']['d_mem'], outputs['pha']['size']))

        # Apply mask to original frame and convert back to HWC
        frame = frame * out_mask
        frame = np.squeeze(frame, axis=0)
        frame = np.transpose(frame, (1, 2, 0))

        # cv2.imshow("Frame", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

        frame_count += 1

    end_time = time.time()
    total_time = end_time - start_time
    fps = frame_count / total_time
    print(f"Total Frame: {frame_count},  Sync FPS: {fps}")

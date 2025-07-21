cuda_pre_post_kernels = ''' 
    //PRE PROCESSING
    //uint8->NHWC->NCHW->fp32
    extern "C" __global__ void pre_process(unsigned char* input, float* output, int stride)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if(tid < stride)
        {
            output[0 * stride + tid] = input[0 * stride + tid] / 255.0f;
            output[1 * stride + tid] = input[1 * stride + tid] / 255.0f;
            output[2 * stride + tid] = input[2 * stride + tid] / 255.0f;
        }

        __syncthreads();
    }
    
    //POST PROCESSING
    //1. Apply mask
    //2. NCHW->NHWC->uint8
    extern "C" __global__ void post_process(float* fgr, float* pha, char* output, int stride)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if(tid < stride)
        {
            fgr[0 * stride + tid] = fgr[0 * stride + tid] * pha[0 * stride + tid];
            fgr[1 * stride + tid] = fgr[1 * stride + tid] * pha[0 * stride + tid];
            fgr[2 * stride + tid] = fgr[2 * stride + tid] * pha[0 * stride + tid];
        }

        __syncthreads();

        for(size_t c = 0; c != 3; ++c)
        {
            output[tid * 3 + c] = (unsigned char)(fgr[c * stride + tid] * 255);
        }        
    } 
'''

cuda_pre_post_kernels_fp16 = ''' 
    #include <cuda_fp16.h>
    
    typedef unsigned long long uint64_t;
    
    //PRE PROCESSING
    //uint8->NHWC->NCHW->fp16
    extern "C" __global__ void pre_process_fp16(unsigned char* input, __half *output, int stride)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if(tid < stride * 3)
        {
            output[0 * stride + tid] = __float2half(input[0 * stride + tid] / 255.0f);
            output[1 * stride + tid] = __float2half(input[1 * stride + tid] / 255.0f);
            output[2 * stride + tid] = __float2half(input[2 * stride + tid] / 255.0f);
        }
    }
    
    extern "C" __global__ void pre_process_fp16_batched(uint64_t* frame_ptrs, __half *output, 
                                                        int height, int width, int channel, int batch_size)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int batch_stride = height * width * channel;
        int stride = height * width;
        if(tid < batch_stride)
        {               
            for(int i=0; i< batch_size; i++)
            {
                unsigned char* frame = reinterpret_cast<unsigned char*>(frame_ptrs[i]);
                output[(0 * stride + tid) + (i * batch_stride)] = __float2half(frame[0 * stride + tid] / 255.0f);
                output[(1 * stride + tid) + (i * batch_stride)] = __float2half(frame[1 * stride + tid] / 255.0f);
                output[(2 * stride + tid) + (i * batch_stride)] = __float2half(frame[2 * stride + tid] / 255.0f);
            }            
        }
    }
    
    //POST PROCESSING
    //1. Apply mask
    //2. NCHW->NHWC->uint8
    extern "C" __global__ void post_process_fp16(__half* fgr, __half* pha, char* output, int stride)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if(tid < stride)
        {
            fgr[0 * stride + tid] = fgr[0 * stride + tid] * pha[0 * stride + tid];
            fgr[1 * stride + tid] = fgr[1 * stride + tid] * pha[0 * stride + tid];
            fgr[2 * stride + tid] = fgr[2 * stride + tid] * pha[0 * stride + tid];
        }

        __syncthreads();

        for(size_t c = 0; c != 3; ++c)
        {
            float value = __half2float(__hmul(fgr[c * stride + tid], 255));
            value = max(0.0f, min(255.0f, value)); 
            output[tid * 3 + c] = static_cast<unsigned char>(value);
        }        
    }  
    
    //Batched POST PROCESSING
    extern "C" __global__ void post_process_fp16_batched(__half* fgr, __half* pha, char* output, 
                                                         int height, int width, int channel, int batch_size)
    {   
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int stride = height * width;
        int batch_stride = height * width * channel;
        
        if(tid < stride)
        {
            for (int i=0; i<batch_size; i++)
            {
                fgr[(i * batch_stride) + (0 * stride + tid)] = fgr[(i * batch_stride) + (0 * stride + tid)] * pha[(i * stride) + ( 0 * stride + tid)];
                fgr[(i * batch_stride) + (1 * stride + tid)] = fgr[(i * batch_stride) + (1 * stride + tid)] * pha[(i * stride) + ( 0 * stride + tid)];
                fgr[(i * batch_stride) + (2 * stride + tid)] = fgr[(i * batch_stride) + (2 * stride + tid)] * pha[(i * stride) + ( 0 * stride + tid)];
            }
        }

        __syncthreads();
        
        for (int i=0; i<batch_size; i++)
        {
            for(size_t c = 0; c != 3; ++c)
            {
                float value = __half2float(__hmul(fgr[(i * batch_stride) + (c * stride + tid)], 255));
                value = max(0.0f, min(255.0f, value)); 
                output[(i * batch_stride) + (tid * 3 + c)] = static_cast<unsigned char>(value);
            }
        }                
    }      
'''
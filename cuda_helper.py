import sys
import os
from cuda import cuda, cudart, nvrtc
import numpy as np

def checkCmdLineFlag(stringRef):
    return any(stringRef == i and k < len(sys.argv) - 1 for i, k in enumerate(sys.argv))

def getCmdLineArgumentInt(stringRef):
    for i, k in enumerate(sys.argv):
        if stringRef == i and k < len(sys.argv) - 1:
            return sys.argv[k + 1]
    return 0

def _cudaGetErrorEnum(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, cudart.cudaError_t):
        return cudart.cudaGetErrorName(error)[1]
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError(f"Unknown error type: {error}")

def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError(f"CUDA error code={result[0].value}({_cudaGetErrorEnum(result[0])})")
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]

def findCudaDevice():
    devID = 0
    if checkCmdLineFlag("device="):
        devID = getCmdLineArgumentInt("device=")
    checkCudaErrors(cudart.cudaSetDevice(devID))
    return devID


def findCudaDeviceDRV():
    devID = 0
    if checkCmdLineFlag("device="):
        devID = getCmdLineArgumentInt("device=")
    checkCudaErrors(cuda.cuInit(0))
    cuDevice = checkCudaErrors(cuda.cuDeviceGet(devID))
    return cuDevice

class KernelHelper:
    def __init__(self, code, devID):
        prog = checkCudaErrors(nvrtc.nvrtcCreateProgram(str.encode(code), b"sourceCode.cu", 0, None, None))
        CUDA_HOME = os.getenv("CUDA_HOME")
        if CUDA_HOME is None:
            CUDA_HOME = os.getenv("CUDA_PATH")
        if CUDA_HOME is None:
            raise RuntimeError("Environment variable CUDA_HOME or CUDA_PATH is not set")
        include_dirs = os.path.join(CUDA_HOME, "include")

        # Initialize CUDA
        checkCudaErrors(cudart.cudaFree(0))

        major = checkCudaErrors(
            cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, devID)
        )
        minor = checkCudaErrors(
            cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, devID)
        )
        _, nvrtc_minor = checkCudaErrors(nvrtc.nvrtcVersion())
        use_cubin = nvrtc_minor >= 1
        prefix = "sm" if use_cubin else "compute"
        arch_arg = bytes(f"--gpu-architecture={prefix}_{major}{minor}", "ascii")

        try:
            opts = [
                b"--fmad=true",
                arch_arg,
                f"--include-path={include_dirs}".encode(),
                b"--std=c++11",
                b"-default-device",
            ]
            checkCudaErrors(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))
        except RuntimeError as err:
            logSize = checkCudaErrors(nvrtc.nvrtcGetProgramLogSize(prog))
            log = b" " * logSize
            checkCudaErrors(nvrtc.nvrtcGetProgramLog(prog, log))
            print(log.decode())
            print(err)
            exit(-1)

        if use_cubin:
            dataSize = checkCudaErrors(nvrtc.nvrtcGetCUBINSize(prog))
            data = b" " * dataSize
            checkCudaErrors(nvrtc.nvrtcGetCUBIN(prog, data))
        else:
            dataSize = checkCudaErrors(nvrtc.nvrtcGetPTXSize(prog))
            data = b" " * dataSize
            checkCudaErrors(nvrtc.nvrtcGetPTX(prog, data))

        self.module = checkCudaErrors(cuda.cuModuleLoadData(np.char.array(data)))

    def getFunction(self, name):
        return checkCudaErrors(cuda.cuModuleGetFunction(self.module, name))
#include <cmath>
#include <iostream>

#include "../include/ksgpu/cuda_utils.hpp"

using namespace std;


static void show_device(int device)
{
    cout << "Device " << device << endl;

    // https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html
    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties(&prop, device));

    int clock_rate_kHz = 0;
    int mem_clock_rate_kHz = 0;

    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gb22e8256592b836df9a9cc36c9db7151
    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g49e2f8c2c0bd6fe264f2fc970912e5cd
    CUDA_CALL(cudaDeviceGetAttribute(&clock_rate_kHz, cudaDevAttrClockRate, device));
    CUDA_CALL(cudaDeviceGetAttribute(&mem_clock_rate_kHz, cudaDevAttrMemoryClockRate, device));

    cout << "    name = " << prop.name << "\n"
	 << "    compute capability = " << prop.major << "." << prop.minor << "\n"
	 << "    multiProcessorCount = " << prop.multiProcessorCount << "\n"
	 << "    clockRate = " << clock_rate_kHz << " kHZ  = " << (clock_rate_kHz / 1.0e6) << " GHz\n"
	 << "    l2CacheSize = " << prop.l2CacheSize << " bytes = " << (prop.l2CacheSize / pow(2,20.)) << " MB\n"
	 << "    totalGlobalMem = " << prop.totalGlobalMem << " bytes = " << (prop.totalGlobalMem / pow(2,30.)) << " GB\n"
	 << "    memoryClockRate = " << mem_clock_rate_kHz << " kHZ\n"
	 << "    memoryBusWidth = " << prop.memoryBusWidth << " bits\n"
	 << "    implied global memory bandwidth = " << (mem_clock_rate_kHz * double(prop.memoryBusWidth) / 1.0e6 / 4.) << " GB/s\n"   // empirical!
	 << endl;
}


int main(int argc, char **argv)
{
    int ndevices = -1;
    CUDA_CALL(cudaGetDeviceCount(&ndevices));

    cout << "Number of devices: " << ndevices << endl;

    for (int device = 0; device < ndevices; device++)
	show_device(device);
    
    return 0;
}

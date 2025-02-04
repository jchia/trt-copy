#include <cstdint>
#include <cuda.h>
#include <device_launch_parameters.h>

namespace rgr {

namespace {

__global__ void addKernel(float const* input, float* output, uint32_t size, float const* scalar) {
    uint32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
        output[idx] = input[idx] + *scalar;
}

}

cudaError_t computeAdd(cudaStream_t stream, float const* input, float* output, uint32_t size, float const* scalar) {
    static constexpr uint32_t kBlockSize = 64;
    uint32_t const kGridSize = (size + kBlockSize - 1) / kBlockSize;
    addKernel<<<kGridSize, kBlockSize, 0, stream>>>(input, output, size, scalar);
    return cudaPeekAtLastError();
}

} // namespace rgr


#include "Kuiper_Example_MemcpySize.h"

void Kuiper_Example_MemcpySize_host_to_device(
    uint64_t *dst, uint64_t *src, uint32_t cnt)
{
    MUST(cudaMemcpy(dst, src, (uint64_t) sizeof(uint64_t) * (uint64_t) cnt,
        cudaMemcpyHostToDevice));
}

void Kuiper_Example_MemcpySize_device_to_host(
    uint64_t *dst, uint64_t *src, uint32_t cnt)
{
    MUST(cudaMemcpy(dst, src, (uint64_t) sizeof(uint64_t) * (uint64_t) cnt,
        cudaMemcpyDeviceToHost));
}

void Kuiper_Example_MemcpySize_device_to_device(
    uint64_t *dst, uint64_t *src, uint32_t cnt)
{
    MUST(cudaMemcpy(dst, src, (uint64_t) sizeof(uint64_t) * (uint64_t) cnt,
        cudaMemcpyDeviceToDevice));
}

void Kuiper_Example_MemcpySize_host_to_device_offset(uint64_t *dst,
    uint32_t dst_off, uint64_t *src, uint32_t src_off, uint32_t cnt)
{
    MUST(cudaMemcpy(dst + dst_off, src + src_off,
        (uint64_t) sizeof(uint64_t) * (uint64_t) cnt, cudaMemcpyHostToDevice));
}

void Kuiper_Example_MemcpySize_device_to_host_offset(uint64_t *dst,
    uint32_t dst_off, uint64_t *src, uint32_t src_off, uint32_t cnt)
{
    MUST(cudaMemcpy(dst + dst_off, src + src_off,
        (uint64_t) sizeof(uint64_t) * (uint64_t) cnt, cudaMemcpyDeviceToHost));
}

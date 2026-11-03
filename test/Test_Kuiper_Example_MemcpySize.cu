#include <kuiper.h>
#include <stdio.h>

static void *copied_dst;
static const void *copied_src;
static size_t copied_bytes;
static cudaMemcpyKind copied_kind;
static unsigned copy_calls;

static cudaError_t record_memcpy(
    void *dst, const void *src, size_t bytes, cudaMemcpyKind kind)
{
    copied_dst = dst;
    copied_src = src;
    copied_bytes = bytes;
    copied_kind = kind;
    ++copy_calls;
    return cudaSuccess;
}

// Exercise the generated calls without allocating or copying several GiB.
// Rename the entry points so the normal generated object can also be linked.
#define cudaMemcpy record_memcpy
#define Kuiper_Example_MemcpySize_host_to_device recorded_host_to_device
#define Kuiper_Example_MemcpySize_device_to_host recorded_device_to_host
#define Kuiper_Example_MemcpySize_device_to_device recorded_device_to_device
#define Kuiper_Example_MemcpySize_host_to_device_offset                        \
    recorded_host_to_device_offset
#define Kuiper_Example_MemcpySize_device_to_host_offset                        \
    recorded_device_to_host_offset
#include "Kuiper_Example_MemcpySize.cu"
#undef cudaMemcpy

static bool check(void *dst, const void *src, size_t bytes, cudaMemcpyKind kind)
{
    bool ok = copy_calls == 1 && copied_dst == dst && copied_src == src &&
              copied_bytes == bytes && copied_kind == kind;
    if (!ok)
        fprintf(stderr, "memcpy: expected %zu bytes, got %zu (%u calls)\n",
            bytes, copied_bytes, copy_calls);
    copy_calls = 0;
    return ok;
}

int main()
{
    uint64_t dst[3] = {}, src[3] = {};
    const uint32_t counts[] = {
        0, 1, (1U << 29) - 1, 1U << 29, (1U << 29) + 1, UINT32_MAX - 2};
    bool ok = true;
    for (uint32_t count : counts) {
        size_t bytes = size_t(count) * sizeof(uint64_t);
        recorded_host_to_device(dst, src, count);
        ok &= check(dst, src, bytes, cudaMemcpyHostToDevice);
        recorded_device_to_host(dst, src, count);
        ok &= check(dst, src, bytes, cudaMemcpyDeviceToHost);
        recorded_device_to_device(dst, src, count);
        ok &= check(dst, src, bytes, cudaMemcpyDeviceToDevice);
        recorded_host_to_device_offset(dst, 1, src, 2, count);
        ok &= check(dst + 1, src + 2, bytes, cudaMemcpyHostToDevice);
        recorded_device_to_host_offset(dst, 1, src, 2, count);
        ok &= check(dst + 1, src + 2, bytes, cudaMemcpyDeviceToHost);
    }
    if (!ok)
        return 1;
    printf("memcpy byte counts and offsets: OK\n");
    return 0;
}

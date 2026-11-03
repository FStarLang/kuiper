
#ifndef Kuiper_Example_MemcpySize_H
#define Kuiper_Example_MemcpySize_H

#include <kuiper.h>

void Kuiper_Example_MemcpySize_host_to_device(
    uint64_t *dst, uint64_t *src, uint32_t cnt);

void Kuiper_Example_MemcpySize_device_to_host(
    uint64_t *dst, uint64_t *src, uint32_t cnt);

void Kuiper_Example_MemcpySize_device_to_device(
    uint64_t *dst, uint64_t *src, uint32_t cnt);

void Kuiper_Example_MemcpySize_host_to_device_offset(uint64_t *dst,
    uint32_t dst_off, uint64_t *src, uint32_t src_off, uint32_t cnt);

void Kuiper_Example_MemcpySize_device_to_host_offset(uint64_t *dst,
    uint32_t dst_off, uint64_t *src, uint32_t src_off, uint32_t cnt);

#define Kuiper_Example_MemcpySize_H_DEFINED
#endif /* Kuiper_Example_MemcpySize_H */

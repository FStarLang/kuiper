
#ifndef Kuiper_IntApprox_H
#define Kuiper_IntApprox_H

#include <kuiper.h>

typedef struct Kuiper_Sized_sized__uint32_t_s {
    uint32_t size;
    uint32_t default;
} Kuiper_Sized_sized__uint32_t;

typedef struct Kuiper_Scalars_scalar__uint32_t_s {
    Kuiper_Sized_sized__uint32_t is_sized;
     uint32_t(*add) (uint32_t x0, uint32_t x1);
     uint32_t(*mul) (uint32_t x0, uint32_t x1);
    uint32_t zero;
    uint32_t one;
} Kuiper_Scalars_scalar__uint32_t;

#define Kuiper_IntApprox_H_DEFINED
#endif                          /* Kuiper_IntApprox_H */

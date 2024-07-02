#ifndef __PULSE__GPU_H
#define __PULSE__GPU_H 1

#define PULSE_KCALL(foo, nblk, nthr, ...) foo<<<nblk,nthr>>>(__VA_ARGS__)

#endif

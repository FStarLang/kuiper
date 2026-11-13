#ifndef KUIPER_FLOAT32_NATIVE_NEG_H
#define KUIPER_FLOAT32_NATIVE_NEG_H

/* C name retained when the public abstract Float32 type is extracted. */
typedef float Kuiper_Float32_t;

#ifdef __CUDACC__
__host__ __device__
#endif
static inline float Kuiper_Float32_NativeNeg_neg(float x) {
  return -x;
}

#endif

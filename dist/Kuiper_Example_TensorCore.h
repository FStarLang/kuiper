
#ifndef Kuiper_Example_TensorCore_H
#define Kuiper_Example_TensorCore_H

#include <kuiper.h>

inline
    __device__
    void Kuiper_Example_TensorCore_test(half * m1, half * m2, half * m3);

inline
    __device__
    void Kuiper_Example_TensorCore_test2(half * m1, half * m2, half * m3);

#define Kuiper_Example_TensorCore_H_DEFINED
#endif                          /* Kuiper_Example_TensorCore_H */

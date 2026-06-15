#include <stdio.h>
#include <stdint.h>
#include "Klas_Caxpy.h"

bool ok = true;

/* cuBLAS caxpy: y := alpha*x + y over single-precision complex (cuFloatComplex).
   The kernel is the generic Kuiper element-wise map instantiated at the complex
   scalar instance, extracted to cuComplex arithmetic. Exact small integers. */

#define CAXPY Klas_Caxpy_caxpy

static cuFloatComplex *dev(const cuFloatComplex *h, int n)
{
    cuFloatComplex *g = (cuFloatComplex *) KPR_GPU_ALLOC(sizeof(cuFloatComplex), n);
    MUST(cudaMemcpy(g, h, n * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    return g;
}

static bool ceq(cuFloatComplex a, cuFloatComplex b)
{
    return cuCrealf(a) == cuCrealf(b) && cuCimagf(a) == cuCimagf(b);
}

static void check(const char *name, int n, cuFloatComplex alpha,
                  const cuFloatComplex *x, const cuFloatComplex *y0, const cuFloatComplex *expy)
{
    cuFloatComplex *gx = dev(x, n), *gy = dev(y0, n);
    cuFloatComplex y[64];

    CAXPY(alpha, (uint32_t) n, gy, gx);

    MUST(cudaMemcpy(y, gy, n * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gx));
    MUST(cudaFree(gy));

    bool t = true;
    for (int i = 0; i < n; i++)
        if (!ceq(y[i], expy[i]))
            t = false;
    if (!t)
        ok = false;
    printf("%s =%s", name, t ? "" : " FAILED");
    for (int i = 0; i < n; i++)
        printf(" (%g,%g)", cuCrealf(y[i]), cuCimagf(y[i]));
    printf("\n");
}

int main()
{
    /* alpha = 2 (real), x[i]=1+i, y[i]=i  =>  y := 2*(1+i)+i = 2+3i. */
    cuFloatComplex a1 = make_cuFloatComplex(2, 0);
    cuFloatComplex x1[3] = { make_cuFloatComplex(1, 1), make_cuFloatComplex(1, 1),
        make_cuFloatComplex(1, 1)
    };
    cuFloatComplex y1[3] = { make_cuFloatComplex(0, 1), make_cuFloatComplex(0, 1),
        make_cuFloatComplex(0, 1)
    };
    cuFloatComplex e1[3] = { make_cuFloatComplex(2, 3), make_cuFloatComplex(2, 3),
        make_cuFloatComplex(2, 3)
    };
    check("caxpy_real_a", 3, a1, x1, y1, e1);

    /* alpha = i, x = 1, y = 0  =>  y := i*1 = i. */
    cuFloatComplex a2 = make_cuFloatComplex(0, 1);
    cuFloatComplex x2[2] = { make_cuFloatComplex(1, 0), make_cuFloatComplex(2, 0) };
    cuFloatComplex y2[2] = { make_cuFloatComplex(0, 0), make_cuFloatComplex(0, 0) };
    /* y0 := i*1 = (0,1); y1 := i*2 = (0,2). */
    cuFloatComplex e2[2] = { make_cuFloatComplex(0, 1), make_cuFloatComplex(0, 2) };
    check("caxpy_imag_a", 2, a2, x2, y2, e2);

    /* alpha = 1+i, x = 2+3i, y = 1+i  =>  (1+i)(2+3i)+ (1+i)
       = (-1+5i) + (1+i) = 0+6i. */
    cuFloatComplex a3 = make_cuFloatComplex(1, 1);
    cuFloatComplex x3[2] = { make_cuFloatComplex(2, 3), make_cuFloatComplex(2, 3) };
    cuFloatComplex y3[2] = { make_cuFloatComplex(1, 1), make_cuFloatComplex(1, 1) };
    cuFloatComplex e3[2] = { make_cuFloatComplex(0, 6), make_cuFloatComplex(0, 6) };
    check("caxpy_cplx_a", 2, a3, x3, y3, e3);

    return !ok;
}

#include <stdio.h>
#include <stdint.h>
#include "Klas_Zaxpy.h"

bool ok = true;

/* cuBLAS zaxpy: y := alpha*x + y over double-precision complex (cuDoubleComplex).
   The kernel is the generic Kuiper element-wise map instantiated at the complex
   scalar instance, extracted to cuComplex arithmetic. Exact small integers. */

#define ZAXPY Klas_Zaxpy_zaxpy

static cuDoubleComplex *dev(const cuDoubleComplex *h, int n)
{
    cuDoubleComplex *g = (cuDoubleComplex *) KPR_GPU_ALLOC(sizeof(cuDoubleComplex), n);
    MUST(cudaMemcpy(g, h, n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    return g;
}

static bool ceq(cuDoubleComplex a, cuDoubleComplex b)
{
    return cuCreal(a) == cuCreal(b) && cuCimag(a) == cuCimag(b);
}

static void check(const char *name, int n, cuDoubleComplex alpha,
                  const cuDoubleComplex *x, const cuDoubleComplex *y0, const cuDoubleComplex *expy)
{
    cuDoubleComplex *gx = dev(x, n), *gy = dev(y0, n);
    cuDoubleComplex y[64];

    ZAXPY(alpha, (uint32_t) n, gy, gx);

    MUST(cudaMemcpy(y, gy, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
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
        printf(" (%g,%g)", cuCreal(y[i]), cuCimag(y[i]));
    printf("\n");
}

int main()
{
    /* alpha = 2 (real), x[i]=1+i, y[i]=i  =>  y := 2*(1+i)+i = 2+3i. */
    cuDoubleComplex a1 = make_cuDoubleComplex(2, 0);
    cuDoubleComplex x1[3] = { make_cuDoubleComplex(1, 1), make_cuDoubleComplex(1, 1),
        make_cuDoubleComplex(1, 1)
    };
    cuDoubleComplex y1[3] = { make_cuDoubleComplex(0, 1), make_cuDoubleComplex(0, 1),
        make_cuDoubleComplex(0, 1)
    };
    cuDoubleComplex e1[3] = { make_cuDoubleComplex(2, 3), make_cuDoubleComplex(2, 3),
        make_cuDoubleComplex(2, 3)
    };
    check("zaxpy_real_a", 3, a1, x1, y1, e1);

    /* alpha = i, x = 1, y = 0  =>  y := i*1 = i. */
    cuDoubleComplex a2 = make_cuDoubleComplex(0, 1);
    cuDoubleComplex x2[2] = { make_cuDoubleComplex(1, 0), make_cuDoubleComplex(2, 0) };
    cuDoubleComplex y2[2] = { make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0) };
    /* y0 := i*1 = (0,1); y1 := i*2 = (0,2). */
    cuDoubleComplex e2[2] = { make_cuDoubleComplex(0, 1), make_cuDoubleComplex(0, 2) };
    check("zaxpy_imag_a", 2, a2, x2, y2, e2);

    /* alpha = 1+i, x = 2+3i, y = 1+i  =>  (1+i)(2+3i)+ (1+i)
       = (-1+5i) + (1+i) = 0+6i. */
    cuDoubleComplex a3 = make_cuDoubleComplex(1, 1);
    cuDoubleComplex x3[2] = { make_cuDoubleComplex(2, 3), make_cuDoubleComplex(2, 3) };
    cuDoubleComplex y3[2] = { make_cuDoubleComplex(1, 1), make_cuDoubleComplex(1, 1) };
    cuDoubleComplex e3[2] = { make_cuDoubleComplex(0, 6), make_cuDoubleComplex(0, 6) };
    check("zaxpy_cplx_a", 2, a3, x3, y3, e3);

    return !ok;
}

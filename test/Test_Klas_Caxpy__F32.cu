#include <stdio.h>
#include <stdint.h>
#include "Klas_Caxpy.h"

bool ok = true;

/* cuBLAS caxpy / zaxpy: y := alpha*x + y over complex. Both entry points come
   from the SAME generic Kuiper element-wise map kernel (Klas.Caxpy.caxpy_gen),
   instantiated at the cf32 / cf64 scalar instances; they extract to cuFloatComplex
   (cuCaddf/cuCmulf) and cuDoubleComplex (cuCadd/cuCmul). Exact small integers. */

static void *gpu_alloc(size_t elt, int n)
{
    return KPR_GPU_ALLOC(elt, n);
}

/* ---- single precision (caxpy / cuFloatComplex) ---- */
static void checkc(const char *name, int n, cuFloatComplex alpha,
                   const cuFloatComplex *x, const cuFloatComplex *y0, const cuFloatComplex *expy)
{
    cuFloatComplex *gx = (cuFloatComplex *) gpu_alloc(sizeof(cuFloatComplex), n);
    cuFloatComplex *gy = (cuFloatComplex *) gpu_alloc(sizeof(cuFloatComplex), n);
    MUST(cudaMemcpy(gx, x, n * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gy, y0, n * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    cuFloatComplex y[64];
    Klas_Caxpy_caxpy(alpha, (uint32_t) n, gy, gx);
    MUST(cudaMemcpy(y, gy, n * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gx));
    MUST(cudaFree(gy));
    bool t = true;
    for (int i = 0; i < n; i++)
        if (cuCrealf(y[i]) != cuCrealf(expy[i]) || cuCimagf(y[i]) != cuCimagf(expy[i]))
            t = false;
    if (!t)
        ok = false;
    printf("%s =%s", name, t ? "" : " FAILED");
    for (int i = 0; i < n; i++)
        printf(" (%g,%g)", cuCrealf(y[i]), cuCimagf(y[i]));
    printf("\n");
}

/* ---- double precision (zaxpy / cuDoubleComplex) ---- */
static void checkz(const char *name, int n, cuDoubleComplex alpha,
                   const cuDoubleComplex *x, const cuDoubleComplex *y0, const cuDoubleComplex *expy)
{
    cuDoubleComplex *gx = (cuDoubleComplex *) gpu_alloc(sizeof(cuDoubleComplex), n);
    cuDoubleComplex *gy = (cuDoubleComplex *) gpu_alloc(sizeof(cuDoubleComplex), n);
    MUST(cudaMemcpy(gx, x, n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    MUST(cudaMemcpy(gy, y0, n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    cuDoubleComplex y[64];
    Klas_Caxpy_zaxpy(alpha, (uint32_t) n, gy, gx);
    MUST(cudaMemcpy(y, gy, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    MUST(cudaFree(gx));
    MUST(cudaFree(gy));
    bool t = true;
    for (int i = 0; i < n; i++)
        if (cuCreal(y[i]) != cuCreal(expy[i]) || cuCimag(y[i]) != cuCimag(expy[i]))
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
    /* caxpy: alpha=2, x=1+i, y=i => 2+3i ; alpha=i,x=1,y=0 => i ;
       alpha=1+i, x=2+3i, y=1+i => (-1+5i)+(1+i)=0+6i. */
    cuFloatComplex a1 = make_cuFloatComplex(2, 0);
    cuFloatComplex x1[2] = { make_cuFloatComplex(1, 1), make_cuFloatComplex(1, 1) };
    cuFloatComplex y1[2] = { make_cuFloatComplex(0, 1), make_cuFloatComplex(0, 1) };
    cuFloatComplex e1[2] = { make_cuFloatComplex(2, 3), make_cuFloatComplex(2, 3) };
    checkc("caxpy_real", 2, a1, x1, y1, e1);

    cuFloatComplex a2 = make_cuFloatComplex(1, 1);
    cuFloatComplex x2[2] = { make_cuFloatComplex(2, 3), make_cuFloatComplex(2, 3) };
    cuFloatComplex y2[2] = { make_cuFloatComplex(1, 1), make_cuFloatComplex(1, 1) };
    cuFloatComplex e2[2] = { make_cuFloatComplex(0, 6), make_cuFloatComplex(0, 6) };
    checkc("caxpy_cplx", 2, a2, x2, y2, e2);

    /* zaxpy: same numbers in double precision. */
    cuDoubleComplex b1 = make_cuDoubleComplex(2, 0);
    cuDoubleComplex u1[2] = { make_cuDoubleComplex(1, 1), make_cuDoubleComplex(1, 1) };
    cuDoubleComplex v1[2] = { make_cuDoubleComplex(0, 1), make_cuDoubleComplex(0, 1) };
    cuDoubleComplex f1[2] = { make_cuDoubleComplex(2, 3), make_cuDoubleComplex(2, 3) };
    checkz("zaxpy_real", 2, b1, u1, v1, f1);

    cuDoubleComplex b2 = make_cuDoubleComplex(1, 1);
    cuDoubleComplex u2[2] = { make_cuDoubleComplex(2, 3), make_cuDoubleComplex(2, 3) };
    cuDoubleComplex v2[2] = { make_cuDoubleComplex(1, 1), make_cuDoubleComplex(1, 1) };
    cuDoubleComplex f2[2] = { make_cuDoubleComplex(0, 6), make_cuDoubleComplex(0, 6) };
    checkz("zaxpy_cplx", 2, b2, u2, v2, f2);

    return !ok;
}

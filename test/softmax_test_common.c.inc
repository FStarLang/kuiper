/*
 * Shared test infrastructure for softmax and logsoftmax tests.
 *
 * To use, define these macros before including:
 *   SCALAR        - the scalar type (float, double, half)
 *   TO_DOUBLE(x)  - convert SCALAR to double
 *   FROM_DOUBLE(x)- convert double to SCALAR
 *   FUN(lena, a)  - the fixed-thread softmax/logsoftmax call
 *   FUN_N(nth, lena, a) - the variable-thread call
 *   LABEL         - string label for output (e.g. "softmax_f32")
 *   ATOL          - absolute tolerance
 *   RTOL          - relative tolerance
 *   IS_LOGSOFTMAX - 0 or 1
 */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static int g_ok = 1;
static int g_tests_run = 0;

static void cpu_softmax(const double *in, double *out, int n)
{
    double mx = in[0];
    for (int i = 1; i < n; i++)
        if (in[i] > mx)
            mx = in[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        out[i] = exp(in[i] - mx);
        sum += out[i];
    }
    for (int i = 0; i < n; i++)
        out[i] /= sum;
}

static void cpu_logsoftmax(const double *in, double *out, int n)
{
    double mx = in[0];
    for (int i = 1; i < n; i++)
        if (in[i] > mx)
            mx = in[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += exp(in[i] - mx);
    double lse = mx + log(sum);
    for (int i = 0; i < n; i++)
        out[i] = in[i] - lse;
}

/* Fill array with bounded, deterministic values that won't overflow exp(). */
static void fill_ramp(double *d, SCALAR *s, int n)
{
    for (int i = 0; i < n; i++) {
        d[i] = (i % 16) * 0.5 - 4.0;
        s[i] = FROM_DOUBLE(d[i]);
    }
}

static void fill_constant(double *d, SCALAR *s, int n)
{
    for (int i = 0; i < n; i++) {
        d[i] = 1.5;
        s[i] = FROM_DOUBLE(1.5);
    }
}

static int check(const SCALAR *got, const double *ref, int n,
                 const char *label)
{
    double max_err = 0.0;
    int fail_idx = -1;
    for (int i = 0; i < n; i++) {
        double g = TO_DOUBLE(got[i]);
        double r = ref[i];
        double ae = fabs(g - r);
        double denom = fabs(r);
        if (denom < 1.0) denom = 1.0;
        if (ae > ATOL && ae / denom > RTOL) {
            if (ae > max_err) {
                max_err = ae;
                fail_idx = i;
            }
        }
    }
    g_tests_run++;
    if (fail_idx >= 0) {
        fprintf(stderr, "FAIL %s: at [%d] got %g, ref %g (err %g)\n",
                label, fail_idx, TO_DOUBLE(got[fail_idx]), ref[fail_idx], max_err);
        g_ok = 0;
        return 0;
    }
    return 1;
}

static void test_fixed(int n)
{
    SCALAR *arr = (SCALAR *)malloc(n * sizeof(SCALAR));
    double *din = (double *)malloc(n * sizeof(double));
    double *dref = (double *)malloc(n * sizeof(double));

    /* ramp pattern */
    fill_ramp(din, arr, n);
#if IS_LOGSOFTMAX
    cpu_logsoftmax(din, dref, n);
#else
    cpu_softmax(din, dref, n);
#endif
    FUN(n, arr);
    char label[128];
    snprintf(label, sizeof label, LABEL "(len=%d,ramp)", n);
    check(arr, dref, n, label);

    /* constant pattern */
    fill_constant(din, arr, n);
#if IS_LOGSOFTMAX
    cpu_logsoftmax(din, dref, n);
#else
    cpu_softmax(din, dref, n);
#endif
    FUN(n, arr);
    snprintf(label, sizeof label, LABEL "(len=%d,const)", n);
    check(arr, dref, n, label);

    free(arr);
    free(din);
    free(dref);
}

static void test_variable(int nth, int n)
{
    SCALAR *arr = (SCALAR *)malloc(n * sizeof(SCALAR));
    double *din = (double *)malloc(n * sizeof(double));
    double *dref = (double *)malloc(n * sizeof(double));

    fill_ramp(din, arr, n);
#if IS_LOGSOFTMAX
    cpu_logsoftmax(din, dref, n);
#else
    cpu_softmax(din, dref, n);
#endif
    FUN_N(nth, n, arr);
    char label[128];
    snprintf(label, sizeof label, LABEL "_n(nth=%d,len=%d)", nth, n);
    check(arr, dref, n, label);

    free(arr);
    free(din);
    free(dref);
}

static int run_all_tests(void)
{
    static const int lengths[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        32, 64, 100, 128, 255, 256, 257,
        512, 1000, 1023, 1024, 1025, 1500, 2000,
    };
    int nlens = sizeof lengths / sizeof lengths[0];

    /* Fixed thread count tests */
    for (int i = 0; i < nlens; i++)
        test_fixed(lengths[i]);

    /* Variable thread count tests */
    static const int thread_counts[] = { 1, 32, 33, 64, 128, 255, 256, 512, 1000, 1024 };
    static const int var_lengths[] = { 1, 10, 64, 255, 256, 512, 1024, 2000 };
    int nths = sizeof thread_counts / sizeof thread_counts[0];
    int nvlens = sizeof var_lengths / sizeof var_lengths[0];

    for (int t = 0; t < nths; t++)
        for (int l = 0; l < nvlens; l++)
            test_variable(thread_counts[t], var_lengths[l]);

    printf("%d tests, %s\n", g_tests_run, g_ok ? "OK" : "FAILED");
    return g_ok ? 0 : 1;
}

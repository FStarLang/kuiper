
#include "Kuiper_Kernel_FlashAttention.h"

void
Kuiper_Kernel_FlashAttention_flashattention_f32(float *gS,
                                                float *gKj,
                                                float *gVj,
                                                float *gQi,
                                                float *gOi,
                                                float *gl,
                                                float *gm, uint32_t tid)
{
    float row_m_prev = *gm;
    float row_l_prev = *gl;
    float row_m = 0.0f - INFINITY;
    uint32_t y1 = 0U;
    for (; y1 < 32U; y1++) {
        float sum = 0.0f;
        uint32_t x = 0U;
        for (; x < 128U; x++) {
            uint32_t vx = x;
            sum += gQi[tid * 128U + vx] * gKj[y1 * 128U + vx];
        }
        gS[tid * 32U + y1] = sum;
        row_m = fmaxf(row_m, sum);
    }
    float row_l = 0.0f;
    y1 = 0U;
    for (; y1 < 32U; y1++) {
        uint32_t vy = y1;
        float vs = expf(gS[tid * 32U + vy] - row_m);
        gS[tid * 32U + vy] = vs;
        row_l += vs;
    }
    float row_m_new = fmaxf(row_m_prev, row_m);
    float __anf1 = row_l;
    float __anf01 = row_m;
    float
     row_l_new =
        row_l_prev * expf(row_m_prev - row_m_new) + __anf1 * expf(__anf01 -
                                                                  row_m_new);
    uint32_t x = 0U;
    for (; x < 128U; x++) {
        float pv = 0.0f;
        y1 = 0U;
        for (; y1 < 32U; y1++) {
            uint32_t vy = y1;
            pv += gS[tid * 32U + vy] * gVj[vy * 128U + x];
        }
        uint32_t vx = x;
        float vo = gOi[tid * 128U + vx];
        float vo1 = vo * row_l_prev * expf(row_m_prev - row_m_new) / row_l_new;
        float __anf02 = pv;
        gOi[tid * 128U + vx] = vo1 + expf(row_m - row_m_new) * __anf02;
    }
}

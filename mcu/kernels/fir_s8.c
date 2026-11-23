/* The int8 FIR filter bank of fir_s8.h. */
#include "fir_s8.h"

#include "arm_nnsupportfunctions.h"

static int8_t finish(int32_t acc, int32_t multiplier, int32_t shift, int32_t out_offset)
{
    acc = arm_nn_requantize(acc, multiplier, shift) + out_offset;
    acc = acc < -128 ? -128 : acc;
    return (int8_t)(acc > 127 ? 127 : acc);
}

void fir_bank_s8(const int8_t *x, int32_t n, int32_t x_stride, const int8_t *w8, const int16_t *w16,
                 const int32_t *wsum, const int32_t *bias, const int32_t *multiplier, const int32_t *shift,
                 int32_t n_filters, int32_t k, int32_t pad, int32_t in_offset, int32_t out_offset, int8_t *y,
                 int32_t y_stride_t, int32_t y_stride_f, int8_t *row8, int16_t *row16)
{
    const int32_t padded = n + 2 * pad;
    const int32_t outputs = padded - k + 1;
#if defined(ARM_MATH_MVEI)
    (void)w16;
    (void)row16;
    /* the row in int8 with zero-point padding: sum_j w (x + in_offset) = sum_j w x + in_offset * sum_j w */
    const int8_t zero = (int8_t)(-in_offset);
    for (int32_t i = 0; i < padded + 16; i++) {
        const int32_t s = i - pad;
        row8[i] = (s >= 0 && s < n) ? x[s * x_stride] : zero;
    }
    for (int32_t f = 0; f < n_filters; f++) {
        const int8_t *taps = w8 + f * k;
        const int32_t base = bias[f] + in_offset * wsum[f];
        if (k == 64) { /* EEGNet's kernel: the four tap vectors stay in registers for the whole row */
            const int8x16_t w0 = vldrbq_s8(taps), w1 = vldrbq_s8(taps + 16), w2 = vldrbq_s8(taps + 32),
                            w3 = vldrbq_s8(taps + 48);
            for (int32_t t = 0; t < outputs; t++) {
                const int8_t *window = row8 + t;
                int32_t acc = vmladavaq_s8(base, vldrbq_s8(window), w0);
                acc = vmladavaq_s8(acc, vldrbq_s8(window + 16), w1);
                acc = vmladavaq_s8(acc, vldrbq_s8(window + 32), w2);
                acc = vmladavaq_s8(acc, vldrbq_s8(window + 48), w3);
                y[t * y_stride_t + f * y_stride_f] = finish(acc, multiplier[f], shift[f], out_offset);
            }
            continue;
        }
        for (int32_t t = 0; t < outputs; t++) {
            const int8_t *window = row8 + t;
            int32_t acc = base;
            for (int32_t j = 0; j < k; j += 16) {
                acc = vmladavaq_s8(acc, vldrbq_s8(window + j), vldrbq_s8(taps + j));
            }
            y[t * y_stride_t + f * y_stride_f] = finish(acc, multiplier[f], shift[f], out_offset);
        }
    }
#elif defined(ARM_MATH_DSP)
    (void)wsum;
    (void)row8;
    /* the row in int16 with the offset added and zero padding, so that SMLAD takes two products at a time */
    for (int32_t i = 0; i < padded + 16; i++) {
        const int32_t s = i - pad;
        row16[i] = (int16_t)((s >= 0 && s < n) ? x[s * x_stride] + in_offset : 0);
    }
    for (int32_t f = 0; f < n_filters; f++) {
        const int16_t *taps = w16 + f * k;
        int32_t t = 0;
        for (; t + 1 < outputs; t += 2) { /* two outputs share every pair of taps */
            const int16_t *window = row16 + t;
            int32_t acc0 = bias[f], acc1 = bias[f];
#pragma GCC unroll 8
            for (int32_t j = 0; j < k; j += 2) {
                const int32_t pair = arm_nn_read_s16x2(taps + j);
                acc0 = SMLAD(arm_nn_read_s16x2(window + j), pair, acc0);
                acc1 = SMLAD(arm_nn_read_s16x2(window + j + 1), pair, acc1);
            }
            y[t * y_stride_t + f * y_stride_f] = finish(acc0, multiplier[f], shift[f], out_offset);
            y[(t + 1) * y_stride_t + f * y_stride_f] = finish(acc1, multiplier[f], shift[f], out_offset);
        }
        for (; t < outputs; t++) {
            const int16_t *window = row16 + t;
            int32_t acc = bias[f];
            for (int32_t j = 0; j < k; j += 2) {
                acc = SMLAD(arm_nn_read_s16x2(window + j), arm_nn_read_s16x2(taps + j), acc);
            }
            y[t * y_stride_t + f * y_stride_f] = finish(acc, multiplier[f], shift[f], out_offset);
        }
    }
#else
    (void)wsum;
    (void)w16;
    (void)row8;
    for (int32_t i = 0; i < padded + 16; i++) {
        const int32_t s = i - pad;
        row16[i] = (int16_t)((s >= 0 && s < n) ? x[s * x_stride] + in_offset : 0);
    }
    for (int32_t f = 0; f < n_filters; f++) {
        const int8_t *taps = w8 + f * k;
        int32_t t = 0;
        for (; t + 1 < outputs; t += 2) { /* two outputs share every tap load */
            const int16_t *window = row16 + t;
            int32_t acc0 = bias[f], acc1 = bias[f];
            int32_t next = window[0];
#pragma GCC unroll 4
            for (int32_t j = 0; j < k; j++) {
                const int32_t w = taps[j], here = next;
                next = window[j + 1];
                acc0 += here * w;
                acc1 += next * w;
            }
            y[t * y_stride_t + f * y_stride_f] = finish(acc0, multiplier[f], shift[f], out_offset);
            y[(t + 1) * y_stride_t + f * y_stride_f] = finish(acc1, multiplier[f], shift[f], out_offset);
        }
        for (; t < outputs; t++) {
            const int16_t *window = row16 + t;
            int32_t acc = bias[f];
            for (int32_t j = 0; j < k; j++) {
                acc += window[j] * taps[j];
            }
            y[t * y_stride_t + f * y_stride_f] = finish(acc, multiplier[f], shift[f], out_offset);
        }
    }
#endif
}

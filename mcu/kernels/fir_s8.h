/* An int8 FIR filter bank: EEGNet's temporal convolution (one 1 x K kernel per filter) as a kernel of its own.
 *
 * CMSIS-NN runs this layer through its general paths: a CONV_2D with one input channel in the original
 * layer order, a DEPTHWISE convolution with a 1 x 64 kernel in the swapped order. Both are, per input row
 * (an electrode, or one channel of the swapped network), a bank of FIR filters over time. This kernel does
 * exactly that, with the arithmetic of the TFLite int8 specification as CMSIS-NN implements it: each input
 * plus in_offset (padding is the zero point, so it contributes nothing), an exact int32 sum, plus the bias,
 * arm_nn_requantize (the same inline function the library uses), plus out_offset, clamped to int8. The
 * result is therefore bit-identical to CMSIS-NN and to mibci.mcu's NumPy reference.
 *
 * Code paths, chosen like CMSIS-NN's from the target: Helium (ARM_MATH_MVEI) takes 16-tap dot products of
 * int8 vectors (VMLADAVA); the DSP extension (ARM_MATH_DSP) takes two 16-bit products per SMLAD and computes
 * two outputs per pass; otherwise plain C. */
#pragma once
#include <stdint.h>

/* One input row filtered by n_filters filters of k taps.
 *   x, x_stride    the row: n samples, x[i * x_stride]
 *   w8             int8 taps, w8[f * k + j]          (all paths)
 *   w16            the same taps as int16             (DSP path; may be 0 otherwise)
 *   wsum           sum over j of w8[f * k + j]        (Helium path; may be 0 otherwise)
 *   bias, multiplier, shift   per filter
 *   pad            zero-point padding on each side; the row gives n + 2 * pad - k + 1 outputs
 *   y              output f at time t goes to y[t * y_stride_t + f * y_stride_f]
 *   row8, row16    work buffers of at least n + 2 * pad + 16 elements */
void fir_bank_s8(const int8_t *x, int32_t n, int32_t x_stride, const int8_t *w8, const int16_t *w16,
                 const int32_t *wsum, const int32_t *bias, const int32_t *multiplier, const int32_t *shift,
                 int32_t n_filters, int32_t k, int32_t pad, int32_t in_offset, int32_t out_offset, int8_t *y,
                 int32_t y_stride_t, int32_t y_stride_f, int8_t *row8, int16_t *row16);

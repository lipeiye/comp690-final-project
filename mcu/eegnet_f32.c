/* EEGNet in float32 with CMSIS-NN's (experimental) float kernels: the same layers, shapes and
 * order switch as eegnet_int8.c, BatchNorm folded, so only the number format changes. CMSIS-NN
 * runs these kernels in plain C on cores without Helium and with Helium vectors where it has them.
 * CMSIS-NN has no ELU, so ELU is written here with expm1f from the C library. */
#include <math.h>

#include "arm_nnfunctions_flt.h"
#include "dims.h"
#include "net.h"
#include "weights.h"

#ifndef SCRATCH_BYTES
#define SCRATCH_BYTES 0
#endif

static float buf_a[BUF_A_ELEMS] __attribute__((aligned(16)));
static float buf_b[BUF_B_ELEMS] __attribute__((aligned(16)));
static float logits[N_CLASSES];
#if SCRATCH_BYTES > 0
static float scratch[(SCRATCH_BYTES + 3) / 4] __attribute__((aligned(16)));
#define SCRATCH scratch
#else
#define SCRATCH 0
#endif
static int32_t scratch_needed;

#define ACT {.min = -INFINITY, .max = INFINITY}
#define ONE {.w = 1, .h = 1}
#define FC_IN (W2 * F2)

#if MODEL_SWAPPED
#define L1_COUT F1D
static const cmsis_nn_dims l1_in = {1, 1, N_TIMES, N_CHANS}, l1_f = {F1D, 1, 1, N_CHANS}, l1_out = {1, 1, N_TIMES, F1D};
static const cmsis_nn_dims l2_in = {1, 1, N_TIMES, F1D}, l2_f = {1, 1, K_TEMPORAL, F1D}, l2_out = {1, 1, W0, F1D};
static const cmsis_nn_conv_params_f32 l1_p = {ONE, {.w = 0, .h = 0}, ONE, ACT, ARM_NN_WEIGHT_FORMAT_STANDARD};
static const cmsis_nn_dw_conv_params_f32 l2_p = {1, ONE, {.w = K_TEMPORAL / 2, .h = 0}, ONE, ACT};
#else
#define L1_COUT F1
static const cmsis_nn_dims l1_in = {1, N_CHANS, N_TIMES, 1}, l1_f = {F1, 1, K_TEMPORAL, 1}, l1_out = {1, N_CHANS, W0, F1};
static const cmsis_nn_dims l2_in = {1, N_CHANS, W0, F1}, l2_f = {1, N_CHANS, 1, F1D}, l2_out = {1, 1, W0, F1D};
static const cmsis_nn_conv_params_f32 l1_p = {ONE, {.w = K_TEMPORAL / 2, .h = 0}, ONE, ACT, ARM_NN_WEIGHT_FORMAT_STANDARD};
static const cmsis_nn_dw_conv_params_f32 l2_p = {DEPTH, ONE, {.w = 0, .h = 0}, ONE, ACT};
#endif
static const cmsis_nn_dims l1_bias = {1, 1, 1, L1_COUT}, l2_bias = {1, 1, 1, F1D};
static const cmsis_nn_dims pool1_in = {1, 1, W0, F1D}, pool1_f = {1, 1, 4, 1}, pool1_out = {1, 1, W1, F1D};
static const cmsis_nn_pool_params_f32 pool1_p = {{.w = 4, .h = 1}, {.w = 0, .h = 0}, ACT};
static const cmsis_nn_dims d_in = {1, 1, W1, F1D}, d_f = {1, 1, 16, F1D}, d_out = {1, 1, WD, F1D}, d_bias = {1, 1, 1, F1D};
static const cmsis_nn_dw_conv_params_f32 d_p = {1, ONE, {.w = 8, .h = 0}, ONE, ACT};
static const cmsis_nn_dims p_in = {1, 1, WD, F1D}, p_f = {F2, 1, 1, F1D}, p_out = {1, 1, WD, F2}, p_bias = {1, 1, 1, F2};
static const cmsis_nn_conv_params_f32 p_p = {ONE, {.w = 0, .h = 0}, ONE, ACT, ARM_NN_WEIGHT_FORMAT_STANDARD};
static const cmsis_nn_dims pool2_in = {1, 1, WD, F2}, pool2_f = {1, 1, 8, 1}, pool2_out = {1, 1, W2, F2};
static const cmsis_nn_pool_params_f32 pool2_p = {{.w = 8, .h = 1}, {.w = 0, .h = 0}, ACT};
static const cmsis_nn_dims fc_in = {1, 1, 1, FC_IN}, fc_f = {FC_IN, 1, 1, N_CLASSES}, fc_out = {1, 1, 1, N_CLASSES};
static const cmsis_nn_dims fc_bias = {1, 1, 1, N_CLASSES};
static const cmsis_nn_fc_params_f32 fc_p = {ACT, ARM_NN_WEIGHT_FORMAT_STANDARD};

static int32_t max32(int32_t a, int32_t b)
{
    return a > b ? a : b;
}

int net_init(void)
{
    int32_t need = arm_convolve_wrapper_f32_get_buffer_size(&l1_p, &l1_in, &l1_f, &l1_out);
    need = max32(need, arm_depthwise_conv_wrapper_f32_get_buffer_size(&l2_p, &l2_in, &l2_f, &l2_out));
    need = max32(need, arm_depthwise_conv_wrapper_f32_get_buffer_size(&d_p, &d_in, &d_f, &d_out));
    need = max32(need, arm_convolve_wrapper_f32_get_buffer_size(&p_p, &p_in, &p_f, &p_out));
    need = max32(need, arm_fully_connected_f32_get_buffer_size(&fc_p, &fc_in, &fc_f, &fc_out, ARM_NN_LAYOUT_NHWC));
    scratch_needed = need;
    return need > SCRATCH_BYTES;
}

int32_t net_scratch_needed(void)
{
    return scratch_needed;
}

void *net_input(void)
{
    return buf_a;
}

const void *net_output(void)
{
    return logits;
}

static void elu(float *x, int32_t n)
{
    for (int32_t i = 0; i < n; i++) {
        x[i] = x[i] > 0.0f ? x[i] : expm1f(x[i]);
    }
}

void net_run(void)
{
    cmsis_nn_context ctx = {SCRATCH, SCRATCH_BYTES};

    mark_start();
    arm_convolve_wrapper_f32(&ctx, &l1_p, &l1_in, buf_a, &l1_f, l1_w, &l1_bias, l1_b, &l1_out, buf_b);
    mark_stop();

    mark_start();
    arm_depthwise_conv_wrapper_f32(&ctx, &l2_p, &l2_in, buf_b, &l2_f, l2_w, &l2_bias, l2_b, &l2_out, buf_a);
    mark_stop();

    mark_start();
    elu(buf_a, W0 * F1D);
    mark_stop();

    mark_start();
    arm_avg_pool_f32(&ctx, &pool1_p, &pool1_in, buf_a, &pool1_f, &pool1_out, buf_b);
    mark_stop();

    mark_start();
    arm_depthwise_conv_wrapper_f32(&ctx, &d_p, &d_in, buf_b, &d_f, conv_d_w, &d_bias, conv_d_b, &d_out, buf_a);
    mark_stop();

    mark_start();
    arm_convolve_wrapper_f32(&ctx, &p_p, &p_in, buf_a, &p_f, conv_p_w, &p_bias, conv_p_b, &p_out, buf_b);
    mark_stop();

    mark_start();
    elu(buf_b, WD * F2);
    mark_stop();

    mark_start();
    arm_avg_pool_f32(&ctx, &pool2_p, &pool2_in, buf_b, &pool2_f, &pool2_out, buf_a);
    mark_stop();

    mark_start();
    arm_fully_connected_f32(&ctx, &fc_p, &fc_in, buf_a, &fc_f, fc_w, &fc_bias, fc_b, &fc_out, logits, ARM_NN_LAYOUT_NHWC);
    mark_stop();
}

/* EEGNet in int8 with CMSIS-NN, as TensorFlow Lite for Microcontrollers would run it.
 *
 * The tensors and constants come from weights.h and the shapes from dims.h (both written by
 * mibci.mcu). MODEL_SWAPPED selects the layer order: originally a temporal CONV_2D over every
 * electrode followed by a DEPTHWISE spatial layer with channel multiplier DEPTH; swapped, a 1x1
 * CONV_2D across the electrodes followed by a DEPTHWISE temporal layer. CMSIS-NN's wrappers pick
 * the kernel (pure C, DSP or Helium) from the compiler's target features. ELU is a table lookup. */
#include "arm_nnfunctions.h"
#include "dims.h"
#include "net.h"
#include "weights.h"

#ifndef SCRATCH_BYTES
#define SCRATCH_BYTES 0
#endif

static int8_t buf_a[BUF_A_ELEMS] __attribute__((aligned(16)));
static int8_t buf_b[BUF_B_ELEMS] __attribute__((aligned(16)));
static int8_t logits[N_CLASSES];
#if SCRATCH_BYTES > 0
static int8_t scratch[SCRATCH_BYTES] __attribute__((aligned(16)));
#define SCRATCH scratch
#else
#define SCRATCH 0
#endif
static int32_t fc_kernel_sum[N_CLASSES]; /* Helium's fully connected kernel adds bias and offsets from here */
static int32_t scratch_needed;

#define ACT {.min = -128, .max = 127}
#define ONE {.w = 1, .h = 1}
#define FC_IN (W2 * F2)

#if MODEL_SWAPPED
#define L1_COUT F1D
static const cmsis_nn_dims l1_in = {1, 1, N_TIMES, N_CHANS}, l1_f = {F1D, 1, 1, N_CHANS}, l1_out = {1, 1, N_TIMES, F1D};
static const cmsis_nn_dims l2_in = {1, 1, N_TIMES, F1D}, l2_f = {1, 1, K_TEMPORAL, F1D}, l2_out = {1, 1, W0, F1D};
static const cmsis_nn_conv_params l1_p = {L1_IN_OFFSET, L1_OUT_OFFSET, ONE, {.w = 0, .h = 0}, ONE, ACT};
static const cmsis_nn_dw_conv_params l2_p = {L2_IN_OFFSET, L2_OUT_OFFSET, 1, ONE, {.w = K_TEMPORAL / 2, .h = 0}, ONE, ACT};
#else
#define L1_COUT F1
static const cmsis_nn_dims l1_in = {1, N_CHANS, N_TIMES, 1}, l1_f = {F1, 1, K_TEMPORAL, 1}, l1_out = {1, N_CHANS, W0, F1};
static const cmsis_nn_dims l2_in = {1, N_CHANS, W0, F1}, l2_f = {1, N_CHANS, 1, F1D}, l2_out = {1, 1, W0, F1D};
static const cmsis_nn_conv_params l1_p = {L1_IN_OFFSET, L1_OUT_OFFSET, ONE, {.w = K_TEMPORAL / 2, .h = 0}, ONE, ACT};
static const cmsis_nn_dw_conv_params l2_p = {L2_IN_OFFSET, L2_OUT_OFFSET, DEPTH, ONE, {.w = 0, .h = 0}, ONE, ACT};
#endif
static const cmsis_nn_dims l1_bias = {1, 1, 1, L1_COUT}, l2_bias = {1, 1, 1, F1D};
static const cmsis_nn_dims pool1_in = {1, 1, W0, F1D}, pool1_f = {1, 1, 4, 1}, pool1_out = {1, 1, W1, F1D};
static const cmsis_nn_pool_params pool1_p = {{.w = 4, .h = 1}, {.w = 0, .h = 0}, ACT};
static const cmsis_nn_dims d_in = {1, 1, W1, F1D}, d_f = {1, 1, 16, F1D}, d_out = {1, 1, WD, F1D}, d_bias = {1, 1, 1, F1D};
static const cmsis_nn_dw_conv_params d_p = {CONV_D_IN_OFFSET, CONV_D_OUT_OFFSET, 1, ONE, {.w = 8, .h = 0}, ONE, ACT};
static const cmsis_nn_dims p_in = {1, 1, WD, F1D}, p_f = {F2, 1, 1, F1D}, p_out = {1, 1, WD, F2}, p_bias = {1, 1, 1, F2};
static const cmsis_nn_conv_params p_p = {CONV_P_IN_OFFSET, CONV_P_OUT_OFFSET, ONE, {.w = 0, .h = 0}, ONE, ACT};
static const cmsis_nn_dims pool2_in = {1, 1, WD, F2}, pool2_f = {1, 1, 8, 1}, pool2_out = {1, 1, W2, F2};
static const cmsis_nn_pool_params pool2_p = {{.w = 8, .h = 1}, {.w = 0, .h = 0}, ACT};
static const cmsis_nn_dims fc_in = {1, 1, 1, FC_IN}, fc_f = {FC_IN, 1, 1, N_CLASSES}, fc_out = {1, 1, 1, N_CLASSES};
static const cmsis_nn_dims fc_bias = {1, 1, 1, N_CLASSES};
static const cmsis_nn_fc_params fc_p = {FC_IN_OFFSET, 0, FC_OUT_OFFSET, ACT};
static const cmsis_nn_per_tensor_quant_params fc_q = {FC_MULTIPLIER, FC_SHIFT};

static cmsis_nn_per_channel_quant_params quant(const int32_t *multiplier, const int32_t *shift)
{
    cmsis_nn_per_channel_quant_params q = {(int32_t *)multiplier, (int32_t *)shift};
    return q;
}

static int32_t max32(int32_t a, int32_t b)
{
    return a > b ? a : b;
}

int net_init(void)
{
    int32_t need = arm_convolve_wrapper_s8_get_buffer_size(&l1_p, &l1_in, &l1_f, &l1_out);
    need = max32(need, arm_depthwise_conv_wrapper_s8_get_buffer_size(&l2_p, &l2_in, &l2_f, &l2_out));
    need = max32(need, arm_avgpool_s8_get_buffer_size(W1, F1D));
    need = max32(need, arm_depthwise_conv_wrapper_s8_get_buffer_size(&d_p, &d_in, &d_f, &d_out));
    need = max32(need, arm_convolve_wrapper_s8_get_buffer_size(&p_p, &p_in, &p_f, &p_out));
    need = max32(need, arm_avgpool_s8_get_buffer_size(W2, F2));
    scratch_needed = need;
#if defined(ARM_MATH_MVEI)
    arm_vector_sum_s8(fc_kernel_sum, FC_IN, N_CLASSES, fc_w, FC_IN_OFFSET, 0, fc_b);
#endif
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

static void elu(int8_t *x, int32_t n, const int8_t *table)
{
    for (int32_t i = 0; i < n; i++) {
        x[i] = table[(uint8_t)(x[i] + 128)];
    }
}

void net_run(void)
{
    cmsis_nn_context ctx = {SCRATCH, SCRATCH_BYTES};
    cmsis_nn_per_channel_quant_params q;

    mark_start();
    q = quant(l1_multiplier, l1_shift);
    arm_convolve_wrapper_s8(&ctx, &l1_p, &q, &l1_in, buf_a, &l1_f, l1_w, &l1_bias, l1_b, &l1_out, buf_b);
    mark_stop();

    mark_start();
    q = quant(l2_multiplier, l2_shift);
    arm_depthwise_conv_wrapper_s8(&ctx, &l2_p, &q, &l2_in, buf_b, &l2_f, l2_w, &l2_bias, l2_b, &l2_out, buf_a);
    mark_stop();

    mark_start();
    elu(buf_a, W0 * F1D, elu1_table);
    mark_stop();

    mark_start();
    arm_avgpool_s8(&ctx, &pool1_p, &pool1_in, buf_a, &pool1_f, &pool1_out, buf_b);
    mark_stop();

    mark_start();
    q = quant(conv_d_multiplier, conv_d_shift);
    arm_depthwise_conv_wrapper_s8(&ctx, &d_p, &q, &d_in, buf_b, &d_f, conv_d_w, &d_bias, conv_d_b, &d_out, buf_a);
    mark_stop();

    mark_start();
    q = quant(conv_p_multiplier, conv_p_shift);
    arm_convolve_wrapper_s8(&ctx, &p_p, &q, &p_in, buf_a, &p_f, conv_p_w, &p_bias, conv_p_b, &p_out, buf_b);
    mark_stop();

    mark_start();
    elu(buf_b, WD * F2, elu2_table);
    mark_stop();

    mark_start();
    arm_avgpool_s8(&ctx, &pool2_p, &pool2_in, buf_b, &pool2_f, &pool2_out, buf_a);
    mark_stop();

    mark_start();
    cmsis_nn_context fc_ctx = {fc_kernel_sum, sizeof fc_kernel_sum};
    arm_fully_connected_s8(&fc_ctx, &fc_p, &fc_q, &fc_in, buf_a, &fc_f, fc_w, &fc_bias, fc_b, &fc_out, logits);
    mark_stop();
}

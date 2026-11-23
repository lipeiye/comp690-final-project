/* ONNX Runtime custom operator: the INT8 spatial convolution of EEGNet with its first two layers swapped.
 *
 * After the swap (mibci.variants.spatial_first) the network starts with 16 spatial filters over the
 * raw input: a convolution with one input channel and a kernel spanning all electrodes (C x 1). In
 * ONNX Runtime 1.31.1 this INT8 QLinearConv runs on a generic path, slower than in FP32 (25 against
 * 7 microseconds on the Apple M4), the second gap of the same kind as kernels/ort_grouped_conv.c.
 *
 *   domain "mibci.swapped", op "QLinearSpatialConvNhwc", with the nine inputs of com.microsoft
 *   QLinearConv (channels_last = 1): x int8 NHWC (N, C, W, 1), w int8 (M, 1, C, 1) with zero point 0,
 *   optional B int32 (M; the swap leaves this layer without one); the output is int8 NHWC (N, 1, W, M), the layout the next NHWC convolution reads.
 *
 * Arithmetic as in ort_grouped_conv.c (MLAS's): exact int32 sums with the input zero point folded
 * into the bias, float32 scaling by x_scale * w_scale / y_scale, clamping, rounding to nearest even.
 * NEON: four samples at a time, each electrode's int8 values widened to int16 and multiplied by the 16
 * filters' weights of that electrode (multiply-accumulate by lane), 16 int32x4 accumulators.
 */
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "onnxruntime_c_api.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#define N_INPUTS 9

static const OrtApi *ort;

typedef struct {
    const void *data;
    int64_t dims[4];
    size_t rank, count;
} Tensor;

static OrtStatusPtr tensor(OrtKernelContext *ctx, size_t index, Tensor *t)
{
    const OrtValue *value = NULL;
    OrtTensorTypeAndShapeInfo *info = NULL;
    OrtStatusPtr status = ort->KernelContext_GetInput(ctx, index, &value);
    if (!status) status = ort->GetTensorTypeAndShape(value, &info);
    if (!status) status = ort->GetDimensionsCount(info, &t->rank);
    if (!status && t->rank > 4) status = ort->CreateStatus(ORT_INVALID_ARGUMENT, "QLinearSpatialConvNhwc: rank > 4");
    if (!status) status = ort->GetDimensions(info, t->dims, t->rank);
    if (!status) status = ort->GetTensorShapeElementCount(info, &t->count);
    if (!status) status = ort->GetTensorData(value, &t->data);
    if (info) ort->ReleaseTensorTypeAndShapeInfo(info);
    return status;
}

static inline int8_t requantize(int32_t acc, float scale, float lo, float hi, int32_t zero)
{
    float f = (float)acc * scale;
    f = f < lo ? lo : (f > hi ? hi : f);
    return (int8_t)((int32_t)nearbyintf(f) + zero);
}

/* y[n][0][t][o] = requant(sum_c w[o][c] * x[n][c][t] + bias[o]) */
static void conv_c(const int8_t *x, const int8_t *w, const int32_t *bias, const float *scale, int64_t n_batch,
                   int64_t h, int64_t wid, int64_t m, int64_t t0, float lo, float hi, int32_t zero, int8_t *y)
{
    for (int64_t n = 0; n < n_batch; n++) {
        const int8_t *xn = x + n * h * wid;
        int8_t *yn = y + n * wid * m;
        for (int64_t t = t0; t < wid; t++) {
            for (int64_t o = 0; o < m; o++) {
                int32_t acc = bias[o];
                for (int64_t c = 0; c < h; c++) {
                    acc += (int32_t)w[o * h + c] * (int32_t)xn[c * wid + t];
                }
                yn[t * m + o] = requantize(acc, scale[o], lo, hi, zero);
            }
        }
    }
}

#if defined(__aarch64__)
/* 16 filters: four samples per step; returns the first sample left for the plain C tail */
static int64_t conv_neon_16(const int8_t *x, const int16_t *w16, const int32_t *bias, const float *scale,
                            int64_t n_batch, int64_t h, int64_t wid, float lo, float hi, int32_t zero, int8_t *y)
{
    const float32x4_t vlo = vdupq_n_f32(lo), vhi = vdupq_n_f32(hi);
    const int32x4_t vzero = vdupq_n_s32(zero);
    const int64_t end = wid - wid % 4;
    for (int64_t n = 0; n < n_batch; n++) {
        const int8_t *xn = x + n * h * wid;
        int8_t *yn = y + n * wid * 16;
        for (int64_t t = 0; t < end; t += 4) {
            int32x4_t acc[16];
            for (int o = 0; o < 16; o++) {
                acc[o] = vdupq_n_s32(bias[o]); /* lanes: samples t..t+3 of filter o */
            }
            for (int64_t c = 0; c < h; c++) {
                int8_t four[8] = {0};
                memcpy(four, xn + c * wid + t, 4);
                const int16x4_t v = vget_low_s16(vmovl_s8(vld1_s8(four)));
                const int16x8_t wa = vld1q_s16(w16 + c * 16), wb = vld1q_s16(w16 + c * 16 + 8);
                acc[0] = vmlal_laneq_s16(acc[0], v, wa, 0);
                acc[1] = vmlal_laneq_s16(acc[1], v, wa, 1);
                acc[2] = vmlal_laneq_s16(acc[2], v, wa, 2);
                acc[3] = vmlal_laneq_s16(acc[3], v, wa, 3);
                acc[4] = vmlal_laneq_s16(acc[4], v, wa, 4);
                acc[5] = vmlal_laneq_s16(acc[5], v, wa, 5);
                acc[6] = vmlal_laneq_s16(acc[6], v, wa, 6);
                acc[7] = vmlal_laneq_s16(acc[7], v, wa, 7);
                acc[8] = vmlal_laneq_s16(acc[8], v, wb, 0);
                acc[9] = vmlal_laneq_s16(acc[9], v, wb, 1);
                acc[10] = vmlal_laneq_s16(acc[10], v, wb, 2);
                acc[11] = vmlal_laneq_s16(acc[11], v, wb, 3);
                acc[12] = vmlal_laneq_s16(acc[12], v, wb, 4);
                acc[13] = vmlal_laneq_s16(acc[13], v, wb, 5);
                acc[14] = vmlal_laneq_s16(acc[14], v, wb, 6);
                acc[15] = vmlal_laneq_s16(acc[15], v, wb, 7);
            }
            int32_t q[16][4];
            for (int o = 0; o < 16; o++) {
                float32x4_t f = vmulq_f32(vcvtq_f32_s32(acc[o]), vdupq_n_f32(scale[o]));
                f = vminq_f32(vmaxq_f32(f, vlo), vhi);
                vst1q_s32(q[o], vaddq_s32(vcvtnq_s32_f32(f), vzero)); /* round to nearest, ties to even */
            }
            for (int s = 0; s < 4; s++) {
                for (int o = 0; o < 16; o++) {
                    yn[(t + s) * 16 + o] = (int8_t)q[o][s];
                }
            }
        }
    }
    return end;
}
#endif

static OrtStatusPtr ORT_API_CALL compute(void *op_kernel, OrtKernelContext *ctx)
{
    (void)op_kernel;
    Tensor in[N_INPUTS];
    size_t given = 0;
    OrtStatusPtr counted = ort->KernelContext_GetInputCount(ctx, &given);
    if (counted) return counted;
    for (size_t i = 0; i < N_INPUTS; i++) {
        if (i >= given) { /* only the bias is optional: after the swap this convolution has none */
            memset(&in[i], 0, sizeof in[i]);
            continue;
        }
        OrtStatusPtr status = tensor(ctx, i, &in[i]);
        if (status) return status;
    }
    if (given < N_INPUTS - 1) return ort->CreateStatus(ORT_INVALID_ARGUMENT, "QLinearSpatialConvNhwc: missing inputs");
    const Tensor *x = &in[0], *w = &in[3];
    if (x->rank != 4 || w->rank != 4) return ort->CreateStatus(ORT_INVALID_ARGUMENT, "expects 4-D x and w");
    const int64_t n_batch = x->dims[0], h = x->dims[1], wid = x->dims[2], m = w->dims[0];
    if (x->dims[3] != 1 || w->dims[1] != 1 || w->dims[2] != h || w->dims[3] != 1 ||
        (given == N_INPUTS && in[8].count != (size_t)m) ||
        (in[4].count != 1 && in[4].count != (size_t)m) || in[1].count != 1 || in[6].count != 1) {
        return ort->CreateStatus(ORT_INVALID_ARGUMENT, "QLinearSpatialConvNhwc: unsupported shapes");
    }
    const int8_t *w_zero = in[5].data;
    for (size_t i = 0; i < in[5].count; i++) {
        if (w_zero[i] != 0) return ort->CreateStatus(ORT_INVALID_ARGUMENT, "weights must be symmetric");
    }
    const float x_scale = *(const float *)in[1].data, y_scale = *(const float *)in[6].data;
    const int32_t x_zero = *(const int8_t *)in[2].data, y_zero = *(const int8_t *)in[7].data;
    const float *w_scale = in[4].data;
    const int8_t *wq = w->data;
    const int32_t *b = in[8].data;

    int32_t *bias = malloc(sizeof(int32_t) * m);
    float *scale = malloc(sizeof(float) * m);
    int16_t *w16 = malloc(sizeof(int16_t) * m * h);
    if (!bias || !scale || !w16) {
        free(bias), free(scale), free(w16);
        return ort->CreateStatus(ORT_FAIL, "out of memory");
    }
    for (int64_t o = 0; o < m; o++) {
        int32_t sum = 0;
        for (int64_t c = 0; c < h; c++) {
            sum += wq[o * h + c];
            w16[c * m + o] = wq[o * h + c];
        }
        bias[o] = (b ? b[o] : 0) - x_zero * sum;
        scale[o] = x_scale * w_scale[in[4].count == 1 ? 0 : o] / y_scale; /* as QLinearConv computes it */
    }
    const float lo = (float)(INT8_MIN - y_zero), hi = (float)(INT8_MAX - y_zero);

    const int64_t out_dims[4] = {n_batch, 1, wid, m};
    OrtValue *out = NULL;
    void *y = NULL;
    OrtStatusPtr status = ort->KernelContext_GetOutput(ctx, 0, out_dims, 4, &out);
    if (!status) status = ort->GetTensorMutableData(out, &y);
    if (!status) {
        int64_t done = 0;
#if defined(__aarch64__)
        if (m == 16) {
            done = conv_neon_16(x->data, w16, bias, scale, n_batch, h, wid, lo, hi, y_zero, y);
        }
#endif
        conv_c(x->data, wq, bias, scale, n_batch, h, wid, m, done, lo, hi, y_zero, y);
    }
    free(bias), free(scale), free(w16);
    return status;
}

static OrtStatusPtr ORT_API_CALL create(const OrtCustomOp *op, const OrtApi *api, const OrtKernelInfo *info,
                                        void **kernel)
{
    (void)op;
    (void)api;
    (void)info;
    *kernel = (void *)1; /* stateless; ONNX Runtime only needs a non-null handle */
    return NULL;
}

static void ORT_API_CALL destroy(void *kernel)
{
    (void)kernel;
}

static const char *ORT_API_CALL name(const OrtCustomOp *op)
{
    (void)op;
    return "QLinearSpatialConvNhwc";
}

static const char *ORT_API_CALL provider(const OrtCustomOp *op)
{
    (void)op;
    return "CPUExecutionProvider";
}

static ONNXTensorElementDataType ORT_API_CALL input_type(const OrtCustomOp *op, size_t i)
{
    (void)op;
    static const ONNXTensorElementDataType types[N_INPUTS] = {
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
    };
    return i < N_INPUTS ? types[i] : ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}

static size_t ORT_API_CALL input_count(const OrtCustomOp *op)
{
    (void)op;
    return N_INPUTS;
}

static ONNXTensorElementDataType ORT_API_CALL output_type(const OrtCustomOp *op, size_t i)
{
    (void)op;
    (void)i;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
}

static size_t ORT_API_CALL output_count(const OrtCustomOp *op)
{
    (void)op;
    return 1;
}

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL input_kind(const OrtCustomOp *op, size_t i)
{
    (void)op;
    return i == N_INPUTS - 1 ? INPUT_OUTPUT_OPTIONAL : INPUT_OUTPUT_REQUIRED; /* B */
}

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL required(const OrtCustomOp *op, size_t i)
{
    (void)op;
    (void)i;
    return INPUT_OUTPUT_REQUIRED;
}

static OrtMemType ORT_API_CALL memory(const OrtCustomOp *op, size_t i)
{
    (void)op;
    (void)i;
    return OrtMemTypeDefault;
}

static int ORT_API_CALL one(const OrtCustomOp *op)
{
    (void)op;
    return 1;
}

static int ORT_API_CALL last(const OrtCustomOp *op)
{
    (void)op;
    return INT_MAX;
}

static OrtCustomOp op;

__attribute__((visibility("default"))) OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options,
                                                                                const OrtApiBase *base)
{
    ort = base->GetApi(ORT_API_VERSION);
    if (!ort) return NULL;
    memset(&op, 0, sizeof op);
    op.version = ORT_API_VERSION;
    op.GetName = name;
    op.GetExecutionProviderType = provider;
    op.GetInputType = input_type;
    op.GetInputTypeCount = input_count;
    op.GetOutputType = output_type;
    op.GetOutputTypeCount = output_count;
    op.GetInputCharacteristic = input_kind;
    op.GetOutputCharacteristic = required;
    op.GetInputMemoryType = memory;
    op.GetVariadicInputMinArity = one;
    op.GetVariadicInputHomogeneity = one;
    op.GetVariadicOutputMinArity = one;
    op.GetVariadicOutputHomogeneity = one;
    op.CreateKernelV2 = create;
    op.KernelComputeV2 = compute;
    op.KernelDestroy = destroy;
    op.GetStartVersion = one;
    op.GetEndVersion = last;
    OrtCustomOpDomain *domain = NULL;
    OrtStatus *status = ort->CreateCustomOpDomain("mibci.swapped", &domain);
    if (!status) status = ort->CustomOpDomain_Add(domain, &op);
    if (!status) status = ort->AddCustomOpDomain(options, domain);
    return status;
}

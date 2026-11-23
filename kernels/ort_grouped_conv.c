/* ONNX Runtime custom operator: the INT8 grouped spatial convolution of EEGNet, for ARM64 NEON.
 *
 * ONNX Runtime 1.31's CPU provider has fast INT8 convolution kernels for ordinary convolutions and
 * for depthwise ones (one filter per input channel). EEGNet's spatial layer is neither: 8 groups of
 * 2 filters, each filter spanning all electrodes (kernel C x 1). It falls to the generic path, which
 * copies every group's input into a buffer and multiplies it by a matrix only 2 columns wide (about
 * 1 ns per multiply-accumulate, against 0.03 ns on the fast paths). This operator is the missing kernel:
 *
 *   domain "mibci", op "QLinearGroupedConvNhwc", with the nine inputs of com.microsoft QLinearConv
 *   (x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, B) and attribute
 *   "group". x is int8 NHWC (N, H, W, G), as ONNX Runtime keeps it between its NHWC convolutions;
 *   w is int8 (G*D, 1, H, 1) with zero point 0; B is int32. The output is int8 NCHW (N, G*D, 1, W):
 *   the transpose that ONNX Runtime puts after its NHWC convolutions is folded into the store.
 *
 * The arithmetic is MLAS's (ONNX Runtime's kernel library): an exact int32 sum plus bias, scaled in
 * float32 by x_scale * w_scale / y_scale, clamped, rounded to nearest even, plus the zero point. The
 * zero point of x is folded into the bias (sum w*(x - zp) = sum w*x - zp * sum w), so the inner loop
 * multiplies raw int8 values. With 8 groups of 2 filters, NEON widens each (electrode, sample) row of
 * 8 group values to int16, duplicates each value for its two filters and accumulates all 16 outputs
 * with four 4-lane int32 multiply-accumulates. Other shapes take a plain C loop with the same result.
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
    int64_t group;
} Kernel;

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
    if (!status && t->rank > 4) status = ort->CreateStatus(ORT_INVALID_ARGUMENT, "QLinearGroupedConvNhwc: rank > 4");
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

/* y[n][o][0][t] for o in 0..M-1: plain C, any group size */
static void conv_c(const int8_t *x, const int8_t *w, const int32_t *bias, const float *scale, int64_t n_batch,
                   int64_t h, int64_t wid, int64_t g, int64_t m, float lo, float hi, int32_t zero, int8_t *y)
{
    const int64_t depth = m / g;
    for (int64_t n = 0; n < n_batch; n++) {
        const int8_t *xn = x + n * h * wid * g;
        int8_t *yn = y + n * m * wid;
        for (int64_t o = 0; o < m; o++) {
            const int64_t grp = o / depth;
            for (int64_t t = 0; t < wid; t++) {
                int32_t acc = bias[o];
                for (int64_t r = 0; r < h; r++) {
                    acc += (int32_t)w[o * h + r] * (int32_t)xn[(r * wid + t) * g + grp];
                }
                yn[o * wid + t] = requantize(acc, scale[o], lo, hi, zero);
            }
        }
    }
}

#if defined(__aarch64__)
/* 8 groups x 2 filters (EEGNet-8,2): all 16 outputs of one sample at a time */
static void conv_neon_8x2(const int8_t *x, const int16_t *w16, const int32_t *bias, const float *scale,
                          int64_t n_batch, int64_t h, int64_t wid, float lo, float hi, int32_t zero, int8_t *y)
{
    const int32x4_t b0 = vld1q_s32(bias), b1 = vld1q_s32(bias + 4), b2 = vld1q_s32(bias + 8), b3 = vld1q_s32(bias + 12);
    const float32x4_t s0 = vld1q_f32(scale), s1 = vld1q_f32(scale + 4), s2 = vld1q_f32(scale + 8),
                      s3 = vld1q_f32(scale + 12);
    const float32x4_t vlo = vdupq_n_f32(lo), vhi = vdupq_n_f32(hi);
    const int32x4_t vzero = vdupq_n_s32(zero);
    for (int64_t n = 0; n < n_batch; n++) {
        const int8_t *xn = x + n * h * wid * 8;
        int8_t *yn = y + n * 16 * wid;
        for (int64_t t = 0; t < wid; t++) {
            int32x4_t a0 = b0, a1 = b1, a2 = b2, a3 = b3;
            const int8_t *xp = xn + t * 8;
            for (int64_t r = 0; r < h; r++, xp += wid * 8) {
                const int16x8_t v = vmovl_s8(vld1_s8(xp));       /* the 8 group values of electrode r */
                const int16x8_t lo4 = vzip1q_s16(v, v);           /* g0 g0 g1 g1 g2 g2 g3 g3 */
                const int16x8_t hi4 = vzip2q_s16(v, v);           /* g4 g4 ... g7 g7 */
                const int16x8_t wl = vld1q_s16(w16 + r * 16), wh = vld1q_s16(w16 + r * 16 + 8);
                a0 = vmlal_s16(a0, vget_low_s16(lo4), vget_low_s16(wl));
                a1 = vmlal_high_s16(a1, lo4, wl);
                a2 = vmlal_s16(a2, vget_low_s16(hi4), vget_low_s16(wh));
                a3 = vmlal_high_s16(a3, hi4, wh);
            }
            int32x4_t q[4];
            const int32x4_t acc[4] = {a0, a1, a2, a3};
            const float32x4_t sc[4] = {s0, s1, s2, s3};
            for (int i = 0; i < 4; i++) {
                float32x4_t f = vmulq_f32(vcvtq_f32_s32(acc[i]), sc[i]);
                f = vminq_f32(vmaxq_f32(f, vlo), vhi);
                q[i] = vaddq_s32(vcvtnq_s32_f32(f), vzero); /* round to nearest, ties to even */
            }
            for (int i = 0; i < 4; i++) {
                yn[(4 * i + 0) * wid + t] = (int8_t)vgetq_lane_s32(q[i], 0);
                yn[(4 * i + 1) * wid + t] = (int8_t)vgetq_lane_s32(q[i], 1);
                yn[(4 * i + 2) * wid + t] = (int8_t)vgetq_lane_s32(q[i], 2);
                yn[(4 * i + 3) * wid + t] = (int8_t)vgetq_lane_s32(q[i], 3);
            }
        }
    }
}
#endif

static OrtStatusPtr ORT_API_CALL compute(void *op_kernel, OrtKernelContext *ctx)
{
    const Kernel *k = op_kernel;
    Tensor in[N_INPUTS];
    for (size_t i = 0; i < N_INPUTS; i++) {
        OrtStatusPtr status = tensor(ctx, i, &in[i]);
        if (status) return status;
    }
    const Tensor *x = &in[0], *w = &in[3];
    if (x->rank != 4 || w->rank != 4) return ort->CreateStatus(ORT_INVALID_ARGUMENT, "expects 4-D x and w");
    const int64_t n_batch = x->dims[0], h = x->dims[1], wid = x->dims[2], g = x->dims[3], m = w->dims[0];
    if (g != k->group || m % g || w->dims[1] != 1 || w->dims[2] != h || w->dims[3] != 1 || in[8].count != (size_t)m ||
        (in[4].count != 1 && in[4].count != (size_t)m) || in[1].count != 1 || in[6].count != 1) {
        return ort->CreateStatus(ORT_INVALID_ARGUMENT, "QLinearGroupedConvNhwc: unsupported shapes");
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
        for (int64_t r = 0; r < h; r++) {
            sum += wq[o * h + r];
            w16[r * m + o] = wq[o * h + r];
        }
        bias[o] = b[o] - x_zero * sum;
        scale[o] = x_scale * w_scale[in[4].count == 1 ? 0 : o] / y_scale; /* as QLinearConv computes it */
    }
    const float lo = (float)(INT8_MIN - y_zero), hi = (float)(INT8_MAX - y_zero);

    const int64_t out_dims[4] = {n_batch, m, 1, wid};
    OrtValue *out = NULL;
    void *y = NULL;
    OrtStatusPtr status = ort->KernelContext_GetOutput(ctx, 0, out_dims, 4, &out);
    if (!status) status = ort->GetTensorMutableData(out, &y);
    if (!status) {
#if defined(__aarch64__)
        if (g == 8 && m == 16) {
            conv_neon_8x2(x->data, w16, bias, scale, n_batch, h, wid, lo, hi, y_zero, y);
        } else
#endif
        {
            conv_c(x->data, wq, bias, scale, n_batch, h, wid, g, m, lo, hi, y_zero, y);
        }
    }
    free(bias), free(scale), free(w16);
    return status;
}

static OrtStatusPtr ORT_API_CALL create(const OrtCustomOp *op, const OrtApi *api, const OrtKernelInfo *info,
                                        void **kernel)
{
    (void)op;
    Kernel *k = calloc(1, sizeof *k);
    if (!k) return api->CreateStatus(ORT_FAIL, "out of memory");
    OrtStatusPtr status = api->KernelInfoGetAttribute_int64(info, "group", &k->group);
    if (status) {
        free(k);
        return status;
    }
    *kernel = k;
    return NULL;
}

static void ORT_API_CALL destroy(void *kernel)
{
    free(kernel);
}

static const char *ORT_API_CALL name(const OrtCustomOp *op)
{
    (void)op;
    return "QLinearGroupedConvNhwc";
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
    if (!ort) return NULL; /* the runtime is older than the header; ONNX Runtime then reports the missing op */
    memset(&op, 0, sizeof op);
    op.version = ORT_API_VERSION;
    op.GetName = name;
    op.GetExecutionProviderType = provider;
    op.GetInputType = input_type;
    op.GetInputTypeCount = input_count;
    op.GetOutputType = output_type;
    op.GetOutputTypeCount = output_count;
    op.GetInputCharacteristic = required;
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
    OrtStatus *status = ort->CreateCustomOpDomain("mibci", &domain);
    if (!status) status = ort->CustomOpDomain_Add(domain, &op);
    if (!status) status = ort->AddCustomOpDomain(options, domain);
    return status;
}

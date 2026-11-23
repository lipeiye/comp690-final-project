/* ONNX Runtime custom operators: the two EEGNet convolutions of kernels/ort_grouped_conv.c and
 * kernels/ort_spatial_first_conv.c, for unsigned 8-bit activations (x86).
 *
 * On x86, ONNX Runtime 1.31 rewrites an S8S8 QDQ graph as U8S8 before fusing it: activations and their
 * zero points become uint8 (x + 128), weights stay int8. The two int8 operators therefore cannot be
 * swapped into an x86 graph. These are the same operators for uint8 x and y, in domain "mibci.u8":
 *
 *   QLinearGroupedConvNhwc  attribute "group"; x uint8 NHWC (N, H, W, G); w int8 (G*D, 1, H, 1), zero point 0;
 *                           B int32 (G*D); output uint8 NCHW (N, G*D, 1, W) (the following transpose folded in)
 *   QLinearSpatialConvNhwc  x uint8 NHWC (N, C, W, 1); w int8 (M, 1, C, 1), zero point 0; optional B int32 (M);
 *                           output uint8 NHWC (N, 1, W, M)
 *
 * Arithmetic as in the int8 files (MLAS's): an exact int32 sum of w * x with the input zero point folded into
 * the bias (sum w*(x - zp) = sum w*x - zp * sum w), scaled in float32 by x_scale * w_scale / y_scale,
 * clamped to the uint8 range, rounded to nearest even, plus the output zero point. Plain C loops only (no
 * hand-written SIMD); mibci.ortkernel_u8 compiles them for the machine they run on.
 */
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "onnxruntime_c_api.h"

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
    if (!status && t->rank > 4) status = ort->CreateStatus(ORT_INVALID_ARGUMENT, "mibci.u8: rank > 4");
    if (!status) status = ort->GetDimensions(info, t->dims, t->rank);
    if (!status) status = ort->GetTensorShapeElementCount(info, &t->count);
    if (!status) status = ort->GetTensorData(value, &t->data);
    if (info) ort->ReleaseTensorTypeAndShapeInfo(info);
    return status;
}

static inline uint8_t requantize(int32_t acc, float scale, float lo, float hi, int32_t zero)
{
    float f = (float)acc * scale;
    f = f < lo ? lo : (f > hi ? hi : f);
    return (uint8_t)((int32_t)nearbyintf(f) + zero);
}

/* Inputs, bias with the input zero point folded in, and per-filter scales; m filters of h weights each. */
typedef struct {
    Tensor in[N_INPUTS];
    int32_t *bias;
    float *scale, lo, hi;
    int32_t y_zero;
} Prepared;

static OrtStatusPtr prepare(OrtKernelContext *ctx, int bias_optional, int64_t m, int64_t h, Prepared *p)
{
    size_t given = 0;
    OrtStatusPtr status = ort->KernelContext_GetInputCount(ctx, &given);
    if (status) return status;
    if (given < N_INPUTS - (bias_optional ? 1 : 0)) return ort->CreateStatus(ORT_INVALID_ARGUMENT, "mibci.u8: missing inputs");
    for (size_t i = 0; i < N_INPUTS; i++) {
        if (i >= given) {
            memset(&p->in[i], 0, sizeof p->in[i]);
            continue;
        }
        status = tensor(ctx, i, &p->in[i]);
        if (status) return status;
    }
    const Tensor *in = p->in;
    if ((in[4].count != 1 && in[4].count != (size_t)m) || in[1].count != 1 || in[6].count != 1 ||
        (given == N_INPUTS && in[8].count != (size_t)m)) {
        return ort->CreateStatus(ORT_INVALID_ARGUMENT, "mibci.u8: unsupported scales or bias");
    }
    const int8_t *w_zero = in[5].data;
    for (size_t i = 0; i < in[5].count; i++) {
        if (w_zero[i] != 0) return ort->CreateStatus(ORT_INVALID_ARGUMENT, "weights must be symmetric");
    }
    const float x_scale = *(const float *)in[1].data, y_scale = *(const float *)in[6].data;
    const int32_t x_zero = *(const uint8_t *)in[2].data;
    p->y_zero = *(const uint8_t *)in[7].data;
    const float *w_scale = in[4].data;
    const int8_t *wq = in[3].data;
    const int32_t *b = given == N_INPUTS ? in[8].data : NULL;
    p->bias = malloc(sizeof(int32_t) * m);
    p->scale = malloc(sizeof(float) * m);
    if (!p->bias || !p->scale) {
        free(p->bias), free(p->scale);
        return ort->CreateStatus(ORT_FAIL, "out of memory");
    }
    for (int64_t o = 0; o < m; o++) {
        int32_t sum = 0;
        for (int64_t r = 0; r < h; r++) sum += wq[o * h + r];
        p->bias[o] = (b ? b[o] : 0) - x_zero * sum;
        p->scale[o] = x_scale * w_scale[in[4].count == 1 ? 0 : o] / y_scale; /* as QLinearConv computes it */
    }
    p->lo = (float)(0 - p->y_zero);
    p->hi = (float)(UINT8_MAX - p->y_zero);
    return NULL;
}

/* grouped spatial convolution: y[n][o][0][t] = requant(sum_r w[o][r] * x[n][r][t][o / depth] + bias[o]) */
static OrtStatusPtr ORT_API_CALL compute_grouped(void *op_kernel, OrtKernelContext *ctx)
{
    const Kernel *k = op_kernel;
    Tensor x, w;
    OrtStatusPtr status = tensor(ctx, 0, &x);
    if (!status) status = tensor(ctx, 3, &w);
    if (status) return status;
    if (x.rank != 4 || w.rank != 4) return ort->CreateStatus(ORT_INVALID_ARGUMENT, "expects 4-D x and w");
    const int64_t n_batch = x.dims[0], h = x.dims[1], wid = x.dims[2], g = x.dims[3], m = w.dims[0];
    if (g != k->group || m % g || w.dims[1] != 1 || w.dims[2] != h || w.dims[3] != 1) {
        return ort->CreateStatus(ORT_INVALID_ARGUMENT, "QLinearGroupedConvNhwc (u8): unsupported shapes");
    }
    Prepared p;
    status = prepare(ctx, 0, m, h, &p);
    if (status) return status;
    const int64_t out_dims[4] = {n_batch, m, 1, wid};
    OrtValue *out = NULL;
    void *yv = NULL;
    status = ort->KernelContext_GetOutput(ctx, 0, out_dims, 4, &out);
    if (!status) status = ort->GetTensorMutableData(out, &yv);
    if (!status) {
        const uint8_t *xd = x.data;
        const int8_t *wq = w.data;
        uint8_t *y = yv;
        const int64_t depth = m / g;
        for (int64_t n = 0; n < n_batch; n++) {
            const uint8_t *xn = xd + n * h * wid * g;
            uint8_t *yn = y + n * m * wid;
            for (int64_t o = 0; o < m; o++) {
                const int64_t grp = o / depth;
                for (int64_t t = 0; t < wid; t++) {
                    int32_t acc = p.bias[o];
                    for (int64_t r = 0; r < h; r++) acc += (int32_t)wq[o * h + r] * (int32_t)xn[(r * wid + t) * g + grp];
                    yn[o * wid + t] = requantize(acc, p.scale[o], p.lo, p.hi, p.y_zero);
                }
            }
        }
    }
    free(p.bias), free(p.scale);
    return status;
}

/* spatial convolution of the swapped network: y[n][0][t][o] = requant(sum_c w[o][c] * x[n][c][t][0] + bias[o]) */
static OrtStatusPtr ORT_API_CALL compute_spatial(void *op_kernel, OrtKernelContext *ctx)
{
    (void)op_kernel;
    Tensor x, w;
    OrtStatusPtr status = tensor(ctx, 0, &x);
    if (!status) status = tensor(ctx, 3, &w);
    if (status) return status;
    if (x.rank != 4 || w.rank != 4) return ort->CreateStatus(ORT_INVALID_ARGUMENT, "expects 4-D x and w");
    const int64_t n_batch = x.dims[0], h = x.dims[1], wid = x.dims[2], m = w.dims[0];
    if (x.dims[3] != 1 || w.dims[1] != 1 || w.dims[2] != h || w.dims[3] != 1) {
        return ort->CreateStatus(ORT_INVALID_ARGUMENT, "QLinearSpatialConvNhwc (u8): unsupported shapes");
    }
    Prepared p;
    status = prepare(ctx, 1, m, h, &p);
    if (status) return status;
    const int64_t out_dims[4] = {n_batch, 1, wid, m};
    OrtValue *out = NULL;
    void *yv = NULL;
    status = ort->KernelContext_GetOutput(ctx, 0, out_dims, 4, &out);
    if (!status) status = ort->GetTensorMutableData(out, &yv);
    if (!status) {
        const uint8_t *xd = x.data;
        const int8_t *wq = w.data;
        uint8_t *y = yv;
        for (int64_t n = 0; n < n_batch; n++) {
            const uint8_t *xn = xd + n * h * wid;
            uint8_t *yn = y + n * wid * m;
            for (int64_t t = 0; t < wid; t++) {
                for (int64_t o = 0; o < m; o++) {
                    int32_t acc = p.bias[o];
                    for (int64_t c = 0; c < h; c++) acc += (int32_t)wq[o * h + c] * (int32_t)xn[c * wid + t];
                    yn[t * m + o] = requantize(acc, p.scale[o], p.lo, p.hi, p.y_zero);
                }
            }
        }
    }
    free(p.bias), free(p.scale);
    return status;
}

static OrtStatusPtr ORT_API_CALL create_grouped(const OrtCustomOp *op, const OrtApi *api, const OrtKernelInfo *info,
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

static OrtStatusPtr ORT_API_CALL create_spatial(const OrtCustomOp *op, const OrtApi *api, const OrtKernelInfo *info,
                                                void **kernel)
{
    (void)op, (void)info;
    *kernel = calloc(1, sizeof(Kernel));
    return *kernel ? NULL : api->CreateStatus(ORT_FAIL, "out of memory");
}

static void ORT_API_CALL destroy(void *kernel)
{
    free(kernel);
}

static const char *ORT_API_CALL name_grouped(const OrtCustomOp *op)
{
    (void)op;
    return "QLinearGroupedConvNhwc";
}

static const char *ORT_API_CALL name_spatial(const OrtCustomOp *op)
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
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
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
    (void)op, (void)i;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
}

static size_t ORT_API_CALL output_count(const OrtCustomOp *op)
{
    (void)op;
    return 1;
}

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL required(const OrtCustomOp *op, size_t i)
{
    (void)op, (void)i;
    return INPUT_OUTPUT_REQUIRED;
}

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL bias_optional(const OrtCustomOp *op, size_t i)
{
    (void)op;
    return i == N_INPUTS - 1 ? INPUT_OUTPUT_OPTIONAL : INPUT_OUTPUT_REQUIRED;
}

static OrtMemType ORT_API_CALL memory(const OrtCustomOp *op, size_t i)
{
    (void)op, (void)i;
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

static OrtCustomOp grouped, spatial;

static void describe(OrtCustomOp *op)
{
    memset(op, 0, sizeof *op);
    op->version = ORT_API_VERSION;
    op->GetExecutionProviderType = provider;
    op->GetInputType = input_type;
    op->GetInputTypeCount = input_count;
    op->GetOutputType = output_type;
    op->GetOutputTypeCount = output_count;
    op->GetInputCharacteristic = required;
    op->GetOutputCharacteristic = required;
    op->GetInputMemoryType = memory;
    op->GetVariadicInputMinArity = one;
    op->GetVariadicInputHomogeneity = one;
    op->GetVariadicOutputMinArity = one;
    op->GetVariadicOutputHomogeneity = one;
    op->KernelDestroy = destroy;
    op->GetStartVersion = one;
    op->GetEndVersion = last;
}

__attribute__((visibility("default"))) OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options,
                                                                                const OrtApiBase *base)
{
    ort = base->GetApi(ORT_API_VERSION);
    if (!ort) return NULL; /* the runtime is older than the header; ONNX Runtime then reports the missing op */
    describe(&grouped);
    grouped.GetName = name_grouped;
    grouped.CreateKernelV2 = create_grouped;
    grouped.KernelComputeV2 = compute_grouped;
    describe(&spatial);
    spatial.GetName = name_spatial;
    spatial.GetInputCharacteristic = bias_optional;
    spatial.CreateKernelV2 = create_spatial;
    spatial.KernelComputeV2 = compute_spatial;
    OrtCustomOpDomain *domain = NULL;
    OrtStatus *status = ort->CreateCustomOpDomain("mibci.u8", &domain);
    if (!status) status = ort->CustomOpDomain_Add(domain, &grouped);
    if (!status) status = ort->CustomOpDomain_Add(domain, &spatial);
    if (!status) status = ort->AddCustomOpDomain(options, domain);
    return status;
}

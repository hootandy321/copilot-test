#ifndef MODEL_QWEN3_H
#define MODEL_QWEN3_H

#include <infiniccl.h>
#include <infiniop.h>
#include <infinirt.h>

#include <stdint.h>

struct Qwen3Model;

typedef struct
{
    infiniDtype_t dt_logits;
    size_t nlayer, d, nh, nkvh, dh, di, dctx, dvoc;
    float epsilon, theta;
    uint32_t end_token;
} Qwen3Meta;

typedef struct
{
    size_t nlayer;
    infiniDtype_t dt_norm, dt_mat;
    // 0 if linear weights are passed as W, any other value if passed as W^T (default format in pytorch)
    int transpose_linear_weights;
    // [dvoc, d]
    const void *input_embd;
    // [d]
    const void *output_norm;
    // [dvoc, d]
    const void *output_embd;
    // nlayer * [d]
    const void *const *attn_norm;
    // nlayer * [ndev, (nh + 2 * nkvh) / ndev * dh, d]
    const void *const *attn_qkv;
    // nlayer * [ndev, (nh + 2 * nkvh) / ndev * dh]
    const void *const *attn_qkv_b;
    // nlayer * [ndev, d, nh / ndev * dh]
    const void *const *attn_o;
    // nlayer * [d]
    const void *const *ffn_norm;
    // nlayer * [ndev, 2 * di / ndev, d]
    const void *const *ffn_gate_up;
    // nlayer * [ndev, d, di / ndev]
    const void *const *ffn_down;
    // Qwen3-specific: Q/K normalization weights
    // nlayer * [dh] - per head normalization
    const void *const *q_norm;
    const void *const *k_norm;
} Qwen3Weights;

//////////////////// APIs ///////////////////////
/// @brief 创建 Qwen3 模型
/// @param device 协处理器种类
/// @param ndev 协处理器数量
/// @param dev_ids 协处理器编号，长度为 ndev
__C __export struct Qwen3Model *
createQwen3Model(const Qwen3Meta *,
                 const Qwen3Weights *,
                 infiniDevice_t device,
                 int ndev,
                 const int *dev_ids);

/// @brief 销毁 Qwen3 模型
__C __export void
destroyQwen3Model(struct Qwen3Model *);

/// @brief 创建 Qwen3 KV Cache
__C __export struct KVCache *
createQwen3KVCache(const struct Qwen3Model *);

/// @brief 复制 Qwen3 KV Cache
__C __export struct KVCache *
duplicateQwen3KVCache(const struct Qwen3Model *,
                      const struct KVCache *, uint32_t seq_len);

/// @brief 销毁 Qwen3 KV Cache
__C __export void
dropQwen3KVCache(const struct Qwen3Model *,
                 struct KVCache *);

/// @brief Qwen3 批次推理一轮
/// @param tokens 输入 token 地址
/// @param ntok 输入 token 数量
/// @param nreq 请求数量
/// @param req_lens 每个请求的 token 数量
/// @param req_pos 每个请求的起始位置
/// @param kv_caches 每个请求的 KV Cache
/// @param temperature 采样温度（0. 表示贪心采样）
/// @param topk 采样 topk（1 表示贪心采样）
/// @param topp 采样 topp
/// @param output 输出 token 数组，每个请求一个输出，长度至少为nreq
__C __export void
inferQwen3Batch(struct Qwen3Model *,
                const uint32_t *tokens, uint32_t ntok,
                const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                struct KVCache **kv_caches,
                const float *temperature, const uint32_t *topk, const float *topp,
                uint32_t *output);

//////////////////// Layer-by-Layer Extraction APIs ///////////////////////
/// @brief 获取 Qwen3 嵌入层输出
/// @param model Qwen3 模型
/// @param tokens 输入 token 地址
/// @param ntok 输入 token 数量
/// @param output 输出嵌入向量，形状为 [ntok, d]
__C __export void
getQwen3EmbeddingOutput(struct Qwen3Model *model,
                        const uint32_t *tokens, uint32_t ntok,
                        float *output);

/// @brief 获取 Qwen3 指定 Transformer 层输出
/// @param model Qwen3 模型
/// @param tokens 输入 token 地址
/// @param ntok 输入 token 数量
/// @param layer_idx 层索引 (0 到 nlayer-1)
/// @param output 输出层向量，形状为 [ntok, d]
__C __export void
getQwen3TransformerLayerOutput(struct Qwen3Model *model,
                               const uint32_t *tokens, uint32_t ntok,
                               uint32_t layer_idx,
                               float *output);

/// @brief 获取 Qwen3 最终归一化层输出
/// @param model Qwen3 模型
/// @param tokens 输入 token 地址
/// @param ntok 输入 token 数量
/// @param output 输出向量，形状为 [ntok, d]
__C __export void
getQwen3FinalNormOutput(struct Qwen3Model *model,
                        const uint32_t *tokens, uint32_t ntok,
                        float *output);

/// @brief 运行 Qwen3 前向传播并输出所有层结果
/// @param model Qwen3 模型
/// @param tokens 输入 token 地址
/// @param ntok 输入 token 数量
/// @param layer_outputs 输出层向量数组，长度为 nlayer+2 (嵌入层 + nlayer个transformer层 + 最终归一化层)
__C __export void
runQwen3ForwardWithLayerOutputs(struct Qwen3Model *model,
                                const uint32_t *tokens, uint32_t ntok,
                                float **layer_outputs);

#endif
#ifndef QWEN3_WEIGHT_HPP
#define QWEN3_WEIGHT_HPP

#include "qwen3_impl.hpp"

#include <cmath>

inline std::shared_ptr<Tensor> getInEmbd(
    Qwen3Meta const *meta,
    Qwen3Weights const *w) {
    auto shape = std::vector<size_t>({meta->dvoc, meta->d});
    return Tensor::weight((char *)w->input_embd, meta->dt_logits, shape);
}

inline std::shared_ptr<Tensor> getOutNorm(
    Qwen3Meta const *meta,
    Qwen3Weights const *w) {
    auto shape = std::vector<size_t>({meta->d});
    return Tensor::weight((char *)w->output_norm, w->dt_norm, shape);
}

inline std::shared_ptr<Tensor> getOutEmbd(
    Qwen3Meta const *meta,
    Qwen3Weights const *w) {
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({meta->dvoc, meta->d});
        return Tensor::weight((char *)w->output_embd, meta->dt_logits, shape)
            ->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({meta->d, meta->dvoc});
        return Tensor::weight((char *)w->output_embd, meta->dt_logits, shape);
    }
}

inline std::shared_ptr<Tensor> getAttnNorm(
    Qwen3Meta const *meta,
    Qwen3Weights const *w,
    size_t layer) {
    auto shape = std::vector<size_t>({meta->d});
    return Tensor::weight((char *)(w->attn_norm[layer]), w->dt_norm, shape);
}

inline std::shared_ptr<Tensor> getAttnQKV(
    Qwen3Meta const *meta,
    Qwen3Weights const *w,
    size_t layer, size_t idev, size_t ndev) {
    auto nkvh = meta->nkvh;
    auto nh = meta->nh;
    auto dh = meta->dh;
    auto d = meta->d;
    size_t offset = idev * ((nkvh * 2 + nh) / ndev * dh) * d * dsize(w->dt_mat);
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({(nh + 2 * nkvh) / ndev * dh, d});
        return Tensor::weight((char *)(w->attn_qkv[layer]) + offset, w->dt_mat, shape)
            ->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({d, (nh + 2 * nkvh) / ndev * dh});
        return Tensor::weight((char *)(w->attn_qkv[layer]) + offset, w->dt_mat, shape);
    }
}

inline std::shared_ptr<Tensor> getAttnQKVBias(
    Qwen3Meta const *meta,
    Qwen3Weights const *w,
    size_t layer, size_t idev, size_t ndev) {
    auto nkvh = meta->nkvh;
    auto nh = meta->nh;
    auto dh = meta->dh;
    size_t offset = idev * ((nkvh * 2 + nh) / ndev * dh) * dsize(w->dt_mat);
    auto shape = std::vector<size_t>({(nh + 2 * nkvh) / ndev * dh});
    return Tensor::weight((char *)(w->attn_qkv_b[layer]) + offset, w->dt_mat, shape);
}

inline std::shared_ptr<Tensor> getAttnO(Qwen3Meta const *meta,
                                        Qwen3Weights const *w, size_t layer,
                                        size_t idev, size_t ndev) {
    auto nh = meta->nh;
    auto dh = meta->dh;
    auto d = meta->d;
    size_t offset = idev * d * (nh / ndev * dh) * dsize(w->dt_mat);
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({d, nh / ndev * dh});
        return Tensor::weight((char *)(w->attn_o[layer]) + offset, w->dt_mat, shape)
            ->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({nh / ndev * dh, d});
        return Tensor::weight((char *)(w->attn_o[layer]) + offset, w->dt_mat, shape);
    }
}

inline std::shared_ptr<Tensor> getFFNNorm(
    Qwen3Meta const *meta,
    Qwen3Weights const *w,
    size_t layer) {
    auto shape = std::vector<size_t>({meta->d});
    return Tensor::weight((char *)(w->ffn_norm[layer]), w->dt_norm, shape);
}

inline std::shared_ptr<Tensor> getFFNGateUp(
    Qwen3Meta const *meta,
    Qwen3Weights const *w,
    size_t layer, size_t idev, size_t ndev) {
    auto di = meta->di;
    auto d = meta->d;
    size_t offset = idev * (2 * di / ndev) * d * dsize(w->dt_mat);
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({2 * di / ndev, d});
        return Tensor::weight((char *)(w->ffn_gate_up[layer]) + offset, w->dt_mat, shape)
            ->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({d, 2 * di / ndev});
        return Tensor::weight((char *)(w->ffn_gate_up[layer]) + offset, w->dt_mat, shape);
    }
}

inline std::shared_ptr<Tensor> getFFNDown(
    Qwen3Meta const *meta,
    Qwen3Weights const *w,
    size_t layer, size_t idev, size_t ndev) {
    auto di = meta->di;
    auto d = meta->d;
    size_t offset = idev * d * (di / ndev) * dsize(w->dt_mat);
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({d, di / ndev});
        return Tensor::weight((char *)(w->ffn_down[layer]) + offset, w->dt_mat, shape)
            ->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({di / ndev, d});
        return Tensor::weight((char *)(w->ffn_down[layer]) + offset, w->dt_mat, shape);
    }
}

// Qwen3 specific helper functions for RoPE
inline std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> 
getRoPETable(Qwen3Meta const *meta, infiniDevice_t device) {
    auto dctx = meta->dctx;
    auto dh = meta->dh;
    auto theta = meta->theta;
    
    std::vector<float> sin_data(dctx * dh);
    std::vector<float> cos_data(dctx * dh);
    
    for (size_t pos = 0; pos < dctx; ++pos) {
        for (size_t i = 0; i < dh / 2; ++i) {
            float freq = 1.0f / std::pow(theta, (float)(2 * i) / dh);
            float angle = pos * freq;
            sin_data[pos * dh + i] = std::sin(angle);
            sin_data[pos * dh + i + dh / 2] = std::sin(angle);
            cos_data[pos * dh + i] = std::cos(angle);
            cos_data[pos * dh + i + dh / 2] = std::cos(angle);
        }
    }
    
    auto sin_shape = std::vector<size_t>({dctx, dh});
    auto cos_shape = std::vector<size_t>({dctx, dh});
    
    auto sin_tensor = Tensor::weight((char*)sin_data.data(), INFINI_F32, sin_shape);
    auto cos_tensor = Tensor::weight((char*)cos_data.data(), INFINI_F32, cos_shape);
    
    return {sin_tensor, cos_tensor};
}

#endif
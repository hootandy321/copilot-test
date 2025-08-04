/*
 * Jiuge Model Weight Extraction Utilities
 * 
 * This header provides utility functions for extracting and reshaping model weights
 * from the global weight storage into device-specific tensor objects. Key features:
 * 
 * - Distributed weight partitioning for tensor parallelism across devices
 * - Automatic tensor shape handling and transposition for different storage formats
 * - RoPE (Rotary Position Embedding) table generation with precomputed trigonometric values
 * - Support for both transposed and non-transposed linear layer weight formats
 * 
 * Weight Distribution Strategy:
 * - Global tensors (embeddings, norms): replicated across all devices
 * - Attention weights: partitioned by attention heads across devices  
 * - FFN weights: partitioned by intermediate dimension across devices
 * - RoPE tables: generated on-demand with mathematical computation
 * 
 * All functions return shared_ptr<Tensor> for automatic memory management.
 */

#ifndef JIUGE_WEIGHT_HPP
#define JIUGE_WEIGHT_HPP

#include "jiuge_impl.hpp"

#include <cmath>
/*
 * Extract Input Embedding Table
 * 
 * Creates a tensor wrapper for the input token embedding lookup table.
 * This table maps token IDs to their corresponding dense vector representations.
 * 
 * Tensor Properties:
 * - Shape: [dvoc, d] where dvoc = vocabulary size, d = model dimension
 * - Data Type: meta->dt_logits (typically FP16, BF16, or FP32)
 * - Memory: References global weight storage (no copying)
 * - Distribution: Replicated on all devices (not partitioned)
 * 
 * Usage: During inference, token IDs are used to index into this table
 * to retrieve the initial hidden representations before transformer processing.
 * 
 * Parameters:
 * - meta: Model metadata containing dimensions and data types
 * - w: Global weight storage containing all model parameters
 * 
 * Returns: Shared tensor wrapper for input embedding table [dvoc, d]
 */
inline std::shared_ptr<Tensor> getInEmbd(
    JiugeMeta const *meta,
    JiugeWeights const *w) {
    auto shape = std::vector<size_t>({meta->dvoc, meta->d});
    return Tensor::weight((char *)w->input_embd, meta->dt_logits, shape);
}

/*
 * Extract Final Layer Normalization Weights
 * 
 * Creates a tensor wrapper for the final RMSNorm layer applied before
 * the language model head. This normalization stabilizes the final
 * hidden representations before vocabulary projection.
 * 
 * Tensor Properties:
 * - Shape: [d] where d = model hidden dimension
 * - Data Type: w->dt_norm (normalization parameter data type)
 * - Memory: References global weight storage
 * - Distribution: Replicated on all devices
 * 
 * RMSNorm Formula: y = x / √(mean(x²) + ε) * γ
 * Where γ is the scale parameter stored in this tensor.
 * 
 * Parameters:
 * - meta: Model metadata for dimension information
 * - w: Global weight storage
 * 
 * Returns: Shared tensor wrapper for output normalization weights [d]
 */
inline std::shared_ptr<Tensor> getOutNorm(
    JiugeMeta const *meta,
    JiugeWeights const *w) {
    auto shape = std::vector<size_t>({meta->d});
    return Tensor::weight((char *)w->output_norm, w->dt_norm, shape);
}

/*
 * Extract Output Embedding / Language Model Head Weights
 * 
 * Creates a tensor wrapper for the final linear projection that maps hidden
 * states to vocabulary logits for next token prediction. This layer is often
 * tied to the input embedding table for parameter efficiency.
 * 
 * Weight Format Handling:
 * - transpose_linear_weights = 0: Weights stored as [d, dvoc] (standard format)
 * - transpose_linear_weights ≠ 0: Weights stored as [dvoc, d] (transposed format)
 * 
 * When weights are stored transposed, we apply permute({1, 0}) to get the
 * correct [d, dvoc] shape for matrix multiplication: hidden_states @ weights.
 * 
 * Matrix Operation: logits = hidden_states @ weights
 * - hidden_states: [batch_size, d]
 * - weights: [d, dvoc] 
 * - logits: [batch_size, dvoc]
 * 
 * Tensor Properties:
 * - Final Shape: [d, dvoc] regardless of storage format
 * - Data Type: meta->dt_logits
 * - Distribution: Replicated on all devices
 * 
 * Parameters:
 * - meta: Model metadata for dimensions
 * - w: Global weight storage with format flags
 * 
 * Returns: Shared tensor wrapper for output projection weights [d, dvoc]
 */
inline std::shared_ptr<Tensor> getOutEmbd(
    JiugeMeta const *meta,
    JiugeWeights const *w) {
    if (w->transpose_linear_weights != 0) {
        // Weights stored as [dvoc, d], need to transpose to [d, dvoc]
        auto shape = std::vector<size_t>({meta->dvoc, meta->d});
        return Tensor::weight((char *)w->output_embd, meta->dt_logits, shape)
            ->permute({1, 0});  // Transpose: [dvoc, d] -> [d, dvoc]
    } else {
        // Weights already stored as [d, dvoc]
        auto shape = std::vector<size_t>({meta->d, meta->dvoc});
        return Tensor::weight((char *)w->output_embd, meta->dt_logits, shape);
    }
}

/*
 * Extract Attention Layer Normalization Weights for Specific Layer
 * 
 * Creates a tensor wrapper for the RMSNorm weights applied before the
 * attention mechanism in each transformer layer. This pre-attention
 * normalization is crucial for training stability and performance.
 * 
 * Tensor Properties:
 * - Shape: [d] where d = model hidden dimension
 * - Data Type: w->dt_norm (normalization data type)
 * - Distribution: Replicated on all devices (same for all devices)
 * - Usage: Applied before QKV projection in attention computation
 * 
 * RMSNorm Application: normalized = (x / √(mean(x²) + ε)) * scale_weights
 * 
 * Parameters:
 * - meta: Model metadata for dimension information
 * - w: Global weight storage
 * - layer: Layer index (0 to nlayer-1)
 * 
 * Returns: Shared tensor wrapper for attention norm weights [d]
 */
inline std::shared_ptr<Tensor> getAttnNorm(
    JiugeMeta const *meta,
    JiugeWeights const *w,
    size_t layer) {
    auto shape = std::vector<size_t>({meta->d});
    return Tensor::weight((char *)(w->attn_norm[layer]), w->dt_norm, shape);
}

/*
 * Extract QKV Projection Weights for Distributed Attention
 * 
 * Creates device-specific tensor wrapper for Query, Key, Value projection weights.
 * In distributed inference, attention heads are partitioned across devices,
 * so each device gets a slice of the total QKV projection matrix.
 * 
 * Weight Partitioning Strategy:
 * - Total attention heads: nh (query heads) + 2*nkvh (key + value heads)
 * - Per-device heads: (nh + 2*nkvh) / ndev
 * - Each device processes its assigned head slice independently
 * - Concatenated QKV format: [Q_heads, K_heads, V_heads] along head dimension
 * 
 * Memory Layout and Offset Calculation:
 * - Global weight shape: [d, (nh + 2*nkvh) * dh] or transposed
 * - Per-device slice: [d, (nh + 2*nkvh)/ndev * dh]
 * - Byte offset: idev * heads_per_device * dh * d * sizeof(data_type)
 * 
 * Tensor Properties:
 * - Input dimension: d (model hidden size)
 * - Output dimension: (nh + 2*nkvh)/ndev * dh (heads per device * head dimension)
 * - Final shape: [d, (nh + 2*nkvh)/ndev * dh] for matrix multiplication
 * - Distribution: Partitioned by attention heads across devices
 * 
 * Matrix Operation: QKV = hidden_states @ weights_slice
 * - hidden_states: [batch_size, d]
 * - weights_slice: [d, (nh + 2*nkvh)/ndev * dh]
 * - QKV: [batch_size, (nh + 2*nkvh)/ndev * dh]
 * 
 * Parameters:
 * - meta: Model metadata for dimensions
 * - w: Global weight storage  
 * - layer: Transformer layer index
 * - idev: Current device index (0 to ndev-1)
 * - ndev: Total number of devices
 * 
 * Returns: Device-specific QKV projection weights [(nh+2*nkvh)/ndev * dh, d] or [d, (nh+2*nkvh)/ndev * dh]
 */
inline std::shared_ptr<Tensor> getAttnQKV(
    JiugeMeta const *meta,
    JiugeWeights const *w,
    size_t layer, size_t idev, size_t ndev) {
    // Extract model dimensions
    auto nkvh = meta->nkvh;     // Total key-value heads
    auto nh = meta->nh;         // Total query heads  
    auto dh = meta->dh;         // Head dimension
    auto d = meta->d;           // Model hidden dimension
    
    /*
     * Calculate Memory Offset for Device Slice
     * 
     * Each device gets (nh + 2*nkvh)/ndev attention heads.
     * The offset calculation accounts for:
     * - Device index: idev
     * - Heads per device: (nh + 2*nkvh) / ndev
     * - Head dimension: dh  
     * - Model dimension: d
     * - Data type size: dsize(w->dt_mat)
     */
    size_t offset = idev * ((nkvh * 2 + nh) / ndev * dh) * d * dsize(w->dt_mat);
    /*
     * Handle Different Weight Storage Formats
     * 
     * The weights may be stored in two formats:
     * 1. Standard: [d, output_dim] - ready for matrix multiplication
     * 2. Transposed: [output_dim, d] - requires permutation for correct usage
     */
    if (w->transpose_linear_weights != 0) {
        // Weights stored as [(nh + 2*nkvh)/ndev * dh, d], transpose to [d, (nh + 2*nkvh)/ndev * dh]
        auto shape = std::vector<size_t>({(nh + 2 * nkvh) / ndev * dh, d});
        return Tensor::weight((char *)(w->attn_qkv[layer]) + offset, w->dt_mat, shape)
            ->permute({1, 0});  // Transpose to correct orientation
    } else {
        // Weights already in correct format: [d, (nh + 2*nkvh)/ndev * dh]
        auto shape = std::vector<size_t>({d, (nh + 2 * nkvh) / ndev * dh});
        return Tensor::weight((char *)(w->attn_qkv[layer]) + offset, w->dt_mat, shape);
    }
}

inline std::shared_ptr<Tensor> getAttnQKVBias(
    JiugeMeta const *meta,
    JiugeWeights const *w,
    size_t layer, size_t idev, size_t ndev) {
    auto nkvh = meta->nkvh;
    auto nh = meta->nh;
    auto dh = meta->dh;
    size_t offset = idev * ((nkvh * 2 + nh) / ndev * dh) * dsize(w->dt_mat);
    auto shape = std::vector<size_t>({(nh + 2 * nkvh) / ndev * dh});
    return Tensor::weight((char *)(w->attn_qkv_b[layer]) + offset, w->dt_mat, shape);
}

inline std::shared_ptr<Tensor> getAttnO(JiugeMeta const *meta,
                                        JiugeWeights const *w, size_t layer,
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
    JiugeMeta const *meta,
    JiugeWeights const *w,
    size_t layer) {
    auto shape = std::vector<size_t>({meta->d});
    return Tensor::weight((char *)(w->ffn_norm[layer]), w->dt_norm, shape);
}

inline std::shared_ptr<Tensor> getFFNGateUp(
    JiugeMeta const *meta,
    JiugeWeights const *w,
    size_t layer, size_t idev, size_t ndev) {
    auto di = meta->di;
    auto d = meta->d;
    size_t offset = idev * (2 * di / ndev) * d * dsize(w->dt_mat);
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({2 * di / ndev, d});
        return Tensor::weight((char *)(w->ffn_gate_up[layer]) + offset,
                              w->dt_mat, shape)
            ->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({d, 2 * di / ndev});
        return Tensor::weight((char *)(w->ffn_gate_up[layer]) + offset,
                              w->dt_mat, shape);
    }
}

inline std::shared_ptr<Tensor> getFFNDown(
    JiugeMeta const *meta,
    JiugeWeights const *w,
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

inline std::shared_ptr<Tensor> getSinTable(JiugeMeta const *meta) {
    auto half_dh = meta->dh / 2;
    auto unit = dsize(meta->dt_logits);
    void *table = std::malloc(meta->dctx * half_dh * unit);

    for (size_t i = 0; i < meta->dctx; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            float _sin = std::sin(
                static_cast<float>(i) / std::pow(meta->theta, static_cast<float>(j) / half_dh));
            if (meta->dt_logits == INFINI_DTYPE_F16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_f16(_sin);
            } else if (meta->dt_logits == INFINI_DTYPE_BF16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_bf16(_sin);
            } else if (meta->dt_logits == INFINI_DTYPE_F32) {
                ((float *)table)[i * half_dh + j] = _sin;
            } else {
                std::cout << "unsupported data type" << std::endl;
                exit(1);
            }
        }
    }
    auto shape = std::vector<size_t>({meta->dctx, half_dh});
    auto tensor = Tensor::weight(table, meta->dt_logits, shape);
    std::free(table);
    return tensor;
}

inline std::shared_ptr<Tensor> getCosTable(JiugeMeta const *meta) {
    auto half_dh = meta->dh / 2;
    auto unit = dsize(meta->dt_logits);
    void *table = std::malloc(meta->dctx * half_dh * unit);

    for (size_t i = 0; i < meta->dctx; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            float _cos = std::cos(
                static_cast<float>(i) / std::pow(meta->theta, static_cast<float>(j) / half_dh));
            if (meta->dt_logits == INFINI_DTYPE_F16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_f16(_cos);
            } else if (meta->dt_logits == INFINI_DTYPE_BF16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_bf16(_cos);
            } else if (meta->dt_logits == INFINI_DTYPE_F32) {
                ((float *)table)[i * half_dh + j] = _cos;
            } else {
                std::cout << "unsupported data type" << std::endl;
                exit(1);
            }
        }
    }
    auto shape = std::vector<size_t>({meta->dctx, half_dh});
    auto tensor = Tensor::weight(table, meta->dt_logits, shape);
    std::free(table);
    return tensor;
}

#endif

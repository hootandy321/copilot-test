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

/*
 * Extract QKV Bias Weights for Distributed Attention (Optional)
 * 
 * Creates device-specific tensor wrapper for QKV projection bias terms.
 * These bias terms are optional and only present in some model configurations.
 * Like the QKV weights, bias terms are partitioned across devices by attention heads.
 * 
 * Bias Partitioning:
 * - Total bias elements: (nh + 2*nkvh) * dh  
 * - Per-device elements: (nh + 2*nkvh)/ndev * dh
 * - Each device gets bias terms for its assigned attention heads
 * 
 * Memory Layout:
 * - Global bias shape: [(nh + 2*nkvh) * dh]
 * - Per-device slice: [(nh + 2*nkvh)/ndev * dh]
 * - Byte offset: idev * heads_per_device * dh * sizeof(data_type)
 * 
 * Usage: Added to QKV projection outputs before attention computation:
 * QKV_output = (hidden_states @ QKV_weights) + QKV_bias
 * 
 * Parameters:
 * - meta: Model metadata for dimensions
 * - w: Global weight storage
 * - layer: Transformer layer index  
 * - idev: Current device index
 * - ndev: Total number of devices
 * 
 * Returns: Device-specific QKV bias terms [(nh+2*nkvh)/ndev * dh]
 */
inline std::shared_ptr<Tensor> getAttnQKVBias(
    JiugeMeta const *meta,
    JiugeWeights const *w,
    size_t layer, size_t idev, size_t ndev) {
    // Extract model dimensions
    auto nkvh = meta->nkvh;     // Total key-value heads
    auto nh = meta->nh;         // Total query heads
    auto dh = meta->dh;         // Head dimension
    
    /*
     * Calculate Memory Offset for Device Slice
     * 
     * Bias terms are 1D vectors, so offset calculation is simpler than weights:
     * offset = device_index * heads_per_device * head_dimension * data_size
     */
    size_t offset = idev * ((nkvh * 2 + nh) / ndev * dh) * dsize(w->dt_mat);
    
    // Create tensor slice for this device's bias terms
    auto shape = std::vector<size_t>({(nh + 2 * nkvh) / ndev * dh});
    return Tensor::weight((char *)(w->attn_qkv_b[layer]) + offset, w->dt_mat, shape);
}

/*
 * Extract Attention Output Projection Weights for Distributed Inference
 * 
 * Creates device-specific tensor wrapper for the attention output projection
 * that combines multi-head attention outputs back to the model dimension.
 * In distributed inference, this projection takes the concatenated attention
 * outputs from this device's heads and projects them to the full model dimension.
 * 
 * Weight Partitioning Strategy:
 * - Each device processes nh/ndev attention heads
 * - Input dimension: nh/ndev * dh (attention heads on this device)
 * - Output dimension: d (full model dimension, same for all devices)
 * - Results are summed across devices via all-reduce operation
 * 
 * Memory Layout and Offset Calculation:
 * - Global weight shape: [nh * dh, d] or transposed
 * - Per-device slice: [nh/ndev * dh, d]  
 * - Byte offset: idev * (nh/ndev * dh) * d * sizeof(data_type)
 * 
 * Matrix Operation: output = attention_heads @ output_weights
 * - attention_heads: [batch_size, nh/ndev * dh] (device-specific)
 * - output_weights: [nh/ndev * dh, d] (device-specific slice)
 * - output: [batch_size, d] (full model dimension)
 * 
 * Distributed Computation:
 * 1. Each device computes partial output from its attention heads
 * 2. All-reduce sums partial outputs across devices
 * 3. Final result represents complete attention output
 * 
 * Parameters:
 * - meta: Model metadata for dimensions
 * - w: Global weight storage
 * - layer: Transformer layer index
 * - idev: Current device index
 * - ndev: Total number of devices
 * 
 * Returns: Device-specific attention output weights [nh/ndev * dh, d] or [d, nh/ndev * dh]
 */
inline std::shared_ptr<Tensor> getAttnO(JiugeMeta const *meta,
                                        JiugeWeights const *w, size_t layer,
                                        size_t idev, size_t ndev) {
    // Extract model dimensions
    auto nh = meta->nh;         // Total query heads
    auto dh = meta->dh;         // Head dimension  
    auto d = meta->d;           // Model hidden dimension
    
    /*
     * Calculate Memory Offset for Device Slice
     * 
     * Each device gets a slice of the output projection corresponding
     * to its assigned attention heads:
     * offset = device_index * heads_per_device * head_dim * model_dim * data_size
     */
    size_t offset = idev * d * (nh / ndev * dh) * dsize(w->dt_mat);
    
    /*
     * Handle Different Weight Storage Formats
     */
    if (w->transpose_linear_weights != 0) {
        // Weights stored as [d, nh/ndev * dh], transpose to [nh/ndev * dh, d]
        auto shape = std::vector<size_t>({d, nh / ndev * dh});
        return Tensor::weight((char *)(w->attn_o[layer]) + offset, w->dt_mat, shape)
            ->permute({1, 0});  // Transpose to correct orientation
    } else {
        // Weights already in correct format: [nh/ndev * dh, d]
        auto shape = std::vector<size_t>({nh / ndev * dh, d});
        return Tensor::weight((char *)(w->attn_o[layer]) + offset, w->dt_mat, shape);
    }
}

/*
 * Extract FFN Layer Normalization Weights for Specific Layer
 * 
 * Creates tensor wrapper for the RMSNorm weights applied before the
 * feed-forward network in each transformer layer. This pre-FFN 
 * normalization stabilizes training and improves performance.
 * 
 * Tensor Properties:
 * - Shape: [d] where d = model hidden dimension
 * - Data Type: w->dt_norm (normalization parameter data type)
 * - Distribution: Replicated on all devices (same for all devices)
 * - Usage: Applied before FFN gate/up projections
 * 
 * RMSNorm Formula: y = (x / √(mean(x²) + ε)) * scale_weights
 * This normalizes the post-attention hidden states before FFN processing.
 * 
 * Parameters:
 * - meta: Model metadata for dimension information
 * - w: Global weight storage
 * - layer: Layer index (0 to nlayer-1)
 * 
 * Returns: Shared tensor wrapper for FFN normalization weights [d]
 */
inline std::shared_ptr<Tensor> getFFNNorm(
    JiugeMeta const *meta,
    JiugeWeights const *w,
    size_t layer) {
    auto shape = std::vector<size_t>({meta->d});
    return Tensor::weight((char *)(w->ffn_norm[layer]), w->dt_norm, shape);
}

/*
 * Extract FFN Gate & Up Projection Weights for Distributed Inference
 * 
 * Creates device-specific tensor wrapper for the combined gate and up
 * projections used in the SwiGLU activation function. These projections
 * expand the model dimension to the intermediate FFN dimension.
 * 
 * SwiGLU Architecture:
 * - Gate projection: linear transformation for gating mechanism
 * - Up projection: linear transformation for value computation  
 * - Combined operation: gate_output = gate_proj(x), up_output = up_proj(x)
 * - SwiGLU activation: output = gate_output * swish(up_output)
 * 
 * Weight Partitioning Strategy:
 * - Total intermediate dimension: di (FFN expansion factor * model_dim)
 * - Per-device dimension: di/ndev (distributed across devices)
 * - Gate + Up combined: 2 * di/ndev (both projections concatenated)
 * - Each device handles a slice of the intermediate dimension
 * 
 * Memory Layout:
 * - Global weight shape: [d, 2*di] or transposed
 * - Per-device slice: [d, 2*di/ndev]
 * - Concatenated format: [gate_weights, up_weights] along output dimension
 * - Byte offset: idev * (2*di/ndev) * d * sizeof(data_type)
 * 
 * Matrix Operations:
 * [gate_output, up_output] = hidden_states @ [gate_weights, up_weights]
 * - hidden_states: [batch_size, d]
 * - combined_weights: [d, 2*di/ndev] 
 * - outputs: [batch_size, 2*di/ndev] = [gate_batch, up_batch]
 * 
 * Parameters:
 * - meta: Model metadata for dimensions
 * - w: Global weight storage
 * - layer: Transformer layer index
 * - idev: Current device index
 * - ndev: Total number of devices
 * 
 * Returns: Device-specific FFN gate&up weights [d, 2*di/ndev] or [2*di/ndev, d]
 */
inline std::shared_ptr<Tensor> getFFNGateUp(
    JiugeMeta const *meta,
    JiugeWeights const *w,
    size_t layer, size_t idev, size_t ndev) {
    // Extract model dimensions
    auto di = meta->di;         // Total intermediate dimension
    auto d = meta->d;           // Model hidden dimension
    
    /*
     * Calculate Memory Offset for Device Slice
     * 
     * Each device gets a slice of the intermediate dimension:
     * offset = device_index * (2*intermediate_per_device) * model_dim * data_size
     */
    size_t offset = idev * (2 * di / ndev) * d * dsize(w->dt_mat);
    
    /*
     * Handle Different Weight Storage Formats
     */
    if (w->transpose_linear_weights != 0) {
        // Weights stored as [2*di/ndev, d], transpose to [d, 2*di/ndev]
        auto shape = std::vector<size_t>({2 * di / ndev, d});
        return Tensor::weight((char *)(w->ffn_gate_up[layer]) + offset,
                              w->dt_mat, shape)
            ->permute({1, 0});  // Transpose to correct orientation
    } else {
        // Weights already in correct format: [d, 2*di/ndev]
        auto shape = std::vector<size_t>({d, 2 * di / ndev});
        return Tensor::weight((char *)(w->ffn_gate_up[layer]) + offset,
                              w->dt_mat, shape);
    }
}

/*
 * Extract FFN Down Projection Weights for Distributed Inference
 * 
 * Creates device-specific tensor wrapper for the down projection that
 * maps the intermediate FFN dimension back to the model dimension.
 * This completes the FFN computation after SwiGLU activation.
 * 
 * FFN Down Projection Role:
 * - Input: SwiGLU activated intermediate representations [batch, di/ndev]
 * - Output: Hidden states back to model dimension [batch, d]  
 * - Purpose: Project expanded intermediate features back to residual stream
 * 
 * Weight Partitioning Strategy:
 * - Input dimension: di/ndev (intermediate dimension per device)
 * - Output dimension: d (full model dimension, same for all devices)
 * - Each device processes its slice of intermediate dimension
 * - Results are summed across devices via all-reduce
 * 
 * Memory Layout:
 * - Global weight shape: [di, d] or transposed
 * - Per-device slice: [di/ndev, d]
 * - Byte offset: idev * (di/ndev) * d * sizeof(data_type)
 * 
 * Matrix Operation: output = intermediate_activated @ down_weights
 * - intermediate_activated: [batch_size, di/ndev] (after SwiGLU)
 * - down_weights: [di/ndev, d] (device-specific slice)
 * - output: [batch_size, d] (back to model dimension)
 * 
 * Distributed Computation:
 * 1. Each device computes partial output from its intermediate slice
 * 2. All-reduce sums partial outputs across devices  
 * 3. Final result represents complete FFN output
 * 
 * Parameters:
 * - meta: Model metadata for dimensions
 * - w: Global weight storage
 * - layer: Transformer layer index
 * - idev: Current device index
 * - ndev: Total number of devices
 * 
 * Returns: Device-specific FFN down weights [di/ndev, d] or [d, di/ndev]
 */
inline std::shared_ptr<Tensor> getFFNDown(
    JiugeMeta const *meta,
    JiugeWeights const *w,
    size_t layer, size_t idev, size_t ndev) {
    // Extract model dimensions
    auto di = meta->di;         // Total intermediate dimension
    auto d = meta->d;           // Model hidden dimension
    
    /*
     * Calculate Memory Offset for Device Slice
     * 
     * Each device gets a slice of the down projection corresponding
     * to its intermediate dimension slice:
     * offset = device_index * model_dim * (intermediate_per_device) * data_size
     */
    size_t offset = idev * d * (di / ndev) * dsize(w->dt_mat);
    
    /*
     * Handle Different Weight Storage Formats
     */
    if (w->transpose_linear_weights != 0) {
        // Weights stored as [d, di/ndev], transpose to [di/ndev, d]
        auto shape = std::vector<size_t>({d, di / ndev});
        return Tensor::weight((char *)(w->ffn_down[layer]) + offset, w->dt_mat, shape)
            ->permute({1, 0});  // Transpose to correct orientation
    } else {
        // Weights already in correct format: [di/ndev, d]
        auto shape = std::vector<size_t>({di / ndev, d});
        return Tensor::weight((char *)(w->ffn_down[layer]) + offset, w->dt_mat, shape);
    }
}

/*
 * Generate RoPE Sine Lookup Table
 * 
 * Creates a precomputed sine table for Rotary Position Embedding (RoPE).
 * RoPE encodes positional information by applying rotation matrices to
 * query and key vectors based on their positions in the sequence.
 * 
 * RoPE Mathematical Foundation:
 * For position pos and dimension pair (i, i+d/2):
 * - Rotation angle: θ_i = pos / (base^(2i/d)) where base = meta->theta (typically 10000)
 * - Sine component: sin(θ_i) = sin(pos / (base^(2i/d)))
 * 
 * Table Structure:
 * - Shape: [dctx, dh/2] where dctx = max context length, dh = head dimension
 * - Entry [pos][i]: sin(pos / (theta^(2i/dh))) for position pos and dimension i
 * - Covers all possible positions (0 to dctx-1) and half of head dimensions
 * 
 * Data Type Handling:
 * - Supports FP16, BF16, and FP32 based on meta->dt_logits
 * - Converts from float32 computation to target data type
 * - Uses appropriate conversion functions (f32_to_f16, f32_to_bf16)
 * 
 * Memory Management:
 * - Allocates temporary storage for computation
 * - Creates tensor wrapper with copy of computed values
 * - Frees temporary storage after tensor creation
 * 
 * Usage During Inference:
 * - Looked up by position IDs to get sine values for rotation
 * - Combined with cosine table to form complete rotation matrices
 * - Applied element-wise to query and key vectors
 * 
 * Parameters:
 * - meta: Model metadata containing context length, head dimension, theta, and data type
 * 
 * Returns: Shared tensor containing precomputed sine values [dctx, dh/2]
 */
inline std::shared_ptr<Tensor> getSinTable(JiugeMeta const *meta) {
    auto half_dh = meta->dh / 2;                    // Half of head dimension
    auto unit = dsize(meta->dt_logits);             // Size of data type in bytes
    void *table = std::malloc(meta->dctx * half_dh * unit);  // Allocate temporary storage

    /*
     * Compute Sine Values for All Positions and Dimensions
     * 
     * Nested loop structure:
     * - Outer loop: iterate over all possible positions (0 to dctx-1)
     * - Inner loop: iterate over half of head dimensions (0 to dh/2-1)
     */
    for (size_t i = 0; i < meta->dctx; i++) {      // Position loop
        for (size_t j = 0; j < half_dh; j++) {     // Dimension loop
            /*
             * RoPE Sine Calculation:
             * 
             * Formula: sin(position / (theta^(2*dim_index / head_dimension)))
             * - position: i (current sequence position)
             * - dim_index: j (current dimension index)  
             * - theta: meta->theta (rotation base, typically 10000)
             * - head_dimension: meta->dh (full head dimension)
             */
            float _sin = std::sin(
                static_cast<float>(i) / std::pow(meta->theta, static_cast<float>(j) / half_dh));
            
            /*
             * Convert and Store Based on Target Data Type
             * 
             * Compute in FP32 for precision, then convert to target format.
             */
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
    
    // Create tensor wrapper and clean up temporary storage
    auto shape = std::vector<size_t>({meta->dctx, half_dh});
    auto tensor = Tensor::weight(table, meta->dt_logits, shape);
    std::free(table);  // Free temporary storage (tensor has its own copy)
    return tensor;
}

/*
 * Generate RoPE Cosine Lookup Table
 * 
 * Creates a precomputed cosine table for Rotary Position Embedding (RoPE).
 * This complements the sine table to form complete rotation matrices for
 * encoding positional information in transformer attention mechanisms.
 * 
 * RoPE Mathematical Foundation:
 * For position pos and dimension pair (i, i+d/2):
 * - Rotation angle: θ_i = pos / (base^(2i/d)) where base = meta->theta
 * - Cosine component: cos(θ_i) = cos(pos / (base^(2i/d)))
 * 
 * Table Structure:
 * - Shape: [dctx, dh/2] where dctx = max context length, dh = head dimension
 * - Entry [pos][i]: cos(pos / (theta^(2i/dh))) for position pos and dimension i
 * - Identical structure to sine table for efficient paired lookup
 * 
 * RoPE Rotation Matrix Application:
 * For each position and dimension pair (i, i+dh/2):
 * - q'[i] = q[i] * cos(θ) - q[i+dh/2] * sin(θ)
 * - q'[i+dh/2] = q[i] * sin(θ) + q[i+dh/2] * cos(θ)
 * 
 * This creates a 2D rotation in the (i, i+dh/2) plane, encoding relative
 * position information directly into the attention computation.
 * 
 * Data Type Support:
 * - FP16: Half precision for memory efficiency
 * - BF16: Brain float for better numeric range  
 * - FP32: Full precision for maximum accuracy
 * - Computed in FP32 then converted to target format
 * 
 * Memory Management:
 * - Temporary allocation during computation
 * - Tensor copy for persistent storage
 * - Automatic cleanup of temporary memory
 * 
 * Performance Considerations:
 * - Precomputed tables avoid expensive trigonometric operations during inference
 * - Cache-friendly access patterns during attention computation
 * - Shared across all layers and attention heads
 * 
 * Parameters:
 * - meta: Model metadata with context length, dimensions, rotation base, and data type
 * 
 * Returns: Shared tensor containing precomputed cosine values [dctx, dh/2]
 */
inline std::shared_ptr<Tensor> getCosTable(JiugeMeta const *meta) {
    auto half_dh = meta->dh / 2;                    // Half of head dimension
    auto unit = dsize(meta->dt_logits);             // Size of data type in bytes
    void *table = std::malloc(meta->dctx * half_dh * unit);  // Allocate temporary storage

    /*
     * Compute Cosine Values for All Positions and Dimensions
     * 
     * Same loop structure as sine table for consistency and paired access.
     */
    for (size_t i = 0; i < meta->dctx; i++) {      // Position loop
        for (size_t j = 0; j < half_dh; j++) {     // Dimension loop  
            /*
             * RoPE Cosine Calculation:
             * 
             * Formula: cos(position / (theta^(2*dim_index / head_dimension)))
             * - Identical to sine calculation but using cosine function
             * - Ensures sin² + cos² = 1 for proper rotation matrices
             */
            float _cos = std::cos(
                static_cast<float>(i) / std::pow(meta->theta, static_cast<float>(j) / half_dh));
            
            /*
             * Convert and Store Based on Target Data Type
             * 
             * Same data type handling as sine table for consistency.
             */
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
    
    // Create tensor wrapper and clean up temporary storage
    auto shape = std::vector<size_t>({meta->dctx, half_dh});
    auto tensor = Tensor::weight(table, meta->dt_logits, shape);
    std::free(table);  // Free temporary storage (tensor has its own copy)
    return tensor;
}

#endif

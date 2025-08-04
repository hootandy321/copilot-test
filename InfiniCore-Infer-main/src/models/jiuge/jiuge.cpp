/*
 * Jiuge Model Implementation - InfiniCore Inference Engine
 * 
 * This file implements the core inference logic for the Jiuge transformer model
 * using InfiniCore's high-performance computing APIs. It handles:
 * - Multi-device distributed inference with tensor parallelism
 * - Memory-efficient attention mechanism with KV-caching
 * - Optimized matrix operations and tensor transformations
 * - Asynchronous execution with proper synchronization
 */

#include "jiuge_impl.hpp"
#include "jiuge_weight.hpp"

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "infinicore_infer.h"

#include <random>
#include <thread>
#include <vector>

/*
 * Device Resource Creation and Initialization
 * 
 * Creates and initializes all GPU/device resources needed for inference including:
 * - InfiniCore device context and operation handles
 * - Distributed tensor weights for multi-device parallelism
 * - Memory pools for efficient buffer management
 * - Communication contexts for inter-device synchronization
 * 
 * Parameters:
 * - rsrc: Output device resource structure to populate
 * - meta: Model metadata (layers, dimensions, data types)
 * - weights: Model weight tensors
 * - device: InfiniCore device type (GPU/CPU)
 * - idev: Current device index in distributed setup (0 to ndev-1)
 * - ndev: Total number of devices for tensor parallelism
 * - dev_id: Physical device ID
 * - comm: InfiniCCL communicator for multi-device operations
 */
void createDeviceResource(DeviceResource *rsrc, const JiugeMeta *meta,
                          const JiugeWeights *weights,
                          infiniDevice_t device, int idev,
                          int ndev, int dev_id,
                          infinicclComm_t comm) {
    // Initialize InfiniCore device context and create operation handle
    // This sets the active device for subsequent InfiniCore API calls
    RUN_INFINI(infinirtSetDevice(device, dev_id));
    
    // Create operation handle for this device - used for all compute operations
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle);
    
    // Create execution stream for asynchronous operations
    infinirtStream_t stream;
    infinirtStreamCreate(&stream);

    /*
     * Weight Tensor Extraction for Distributed Inference
     * 
     * Extract model weights from global weight storage and partition them
     * across devices for tensor parallelism. Each device gets a slice of:
     * - Attention projection weights (QKV, output): partitioned by attention heads
     * - FFN weights: partitioned by intermediate dimension
     * - Normalization weights: replicated across all devices
     * 
     * Tensor shapes:
     * - w_attn_norm: [d] - layer normalization weights, replicated
     * - w_attn_qkv: [d, (nh + 2*nkvh)/ndev * dh] - QKV projection, head-partitioned  
     * - b_attn_qkv: [(nh + 2*nkvh)/ndev * dh] - QKV bias (optional), head-partitioned
     * - w_attn_out: [nh/ndev * dh, d] - output projection, head-partitioned
     * - w_ffn_norm: [d] - FFN normalization weights, replicated
     * - w_ffn_gate_up: [d, 2*di/ndev] - gate & up projections, dim-partitioned
     * - w_ffn_down: [di/ndev, d] - down projection, dim-partitioned
     */
    std::vector<std::shared_ptr<Tensor>> w_attn_norm, w_attn_qkv, b_attn_qkv, w_attn_out,
        w_ffn_norm, w_ffn_gate_up, w_ffn_down;
    for (size_t layer = 0; layer < meta->nlayer; layer++) {
        // Extract attention normalization weights [d] - same on all devices
        w_attn_norm.push_back(
            getAttnNorm(meta, weights, layer));
        
        // Extract QKV projection weights [d, (nh + 2*nkvh)/ndev * dh] 
        // Partitioned by attention heads across devices
        w_attn_qkv.push_back(
            getAttnQKV(meta, weights, layer, idev, ndev));
        
        // Extract QKV bias if present [(nh + 2*nkvh)/ndev * dh]
        if (weights->attn_qkv_b != nullptr) {
            b_attn_qkv.push_back(
                getAttnQKVBias(meta, weights, layer, idev, ndev));
        }
        
        // Extract attention output projection [nh/ndev * dh, d]
        // Partitioned by input dimension (attention heads)
        w_attn_out.push_back(
            getAttnO(meta, weights, layer, idev, ndev));
            
        // Extract FFN normalization weights [d] - same on all devices  
        w_ffn_norm.push_back(
            getFFNNorm(meta, weights, layer));
            
        // Extract FFN gate & up projections [d, 2*di/ndev]
        // Partitioned by intermediate dimension across devices
        w_ffn_gate_up.push_back(
            getFFNGateUp(meta, weights, layer, idev, ndev));
            
        // Extract FFN down projection [di/ndev, d] 
        // Partitioned by input dimension across devices
        w_ffn_down.push_back(
            getFFNDown(meta, weights, layer, idev, ndev));
    }

    // Create memory pool for efficient buffer allocation (128MB)
    // This pool manages temporary tensors during inference to avoid frequent malloc/free
    auto memory_pool = std::make_shared<MemoryPool>(128 * 1024 * 1024);

    // Populate device resource structure with all initialized components
    // This structure contains everything needed for inference on this device
    *rsrc = DeviceResource{
        device,                    // InfiniCore device type (GPU/CPU)
        dev_id,                   // Physical device ID
        handle,                   // InfiniCore operation handle
        getInEmbd(meta, weights), // Input embedding table [dvoc, d]
        getOutNorm(meta, weights),// Output normalization weights [d] 
        getOutEmbd(meta, weights),// Output embedding/LM head [d, dvoc]
        getSinTable(meta),        // RoPE sine table [dctx, dh/2]
        getCosTable(meta),        // RoPE cosine table [dctx, dh/2]
        w_attn_norm,             // Attention norm weights per layer
        w_attn_qkv,              // QKV projection weights per layer  
        b_attn_qkv,              // QKV bias weights per layer (optional)
        w_attn_out,              // Attention output weights per layer
        w_ffn_norm,              // FFN norm weights per layer
        w_ffn_gate_up,           // FFN gate & up weights per layer
        w_ffn_down,              // FFN down weights per layer
        stream,                  // Execution stream for async ops
        comm,                    // Inter-device communication context
        memory_pool,             // Memory pool for temporary buffers
    };
    
    // Synchronize device to ensure all initialization is complete
    RUN_INFINI(infinirtDeviceSynchronize());
}

/*
 * Device Resource Cleanup and Memory Deallocation
 * 
 * Properly releases all device resources in reverse order of allocation:
 * 1. Synchronize device to complete all pending operations
 * 2. Release tensor memory (shared_ptr automatically handles reference counting)
 * 3. Destroy InfiniCore handles and streams
 * 4. Clean up communication contexts
 * 
 * This prevents memory leaks and ensures proper cleanup of GPU resources
 */
void releaseDeviceResource(DeviceResource &res) {
    // Wait for all pending operations to complete before cleanup
    infinirtDeviceSynchronize();
    
    // Release tensor memory by resetting shared_ptr references
    // The underlying memory will be freed when reference count reaches zero
    // Release global model tensors (input/output embeddings, normalization, RoPE tables)
    res.w_in_embd.reset();     // Input embedding table [dvoc, d]
    res.w_out_norm.reset();    // Final layer norm [d]
    res.w_out_embd.reset();    // Output projection/LM head [d, dvoc] 
    res.sin_table.reset();     // RoPE sine lookup table [dctx, dh/2]
    res.cos_table.reset();     // RoPE cosine lookup table [dctx, dh/2]
    
    // Release per-layer attention weights and clear vectors
    for (auto &t : res.w_attn_norm) {
        t.reset();             // Attention layer norm weights [d]
    }
    res.w_attn_norm.clear();
    
    for (auto &t : res.w_attn_qkv) {
        t.reset();             // QKV projection weights [d, (nh+2*nkvh)/ndev * dh]
    }
    res.w_attn_qkv.clear();
    
    for (auto &t : res.b_attn_qkv) {
        t.reset();             // QKV bias weights [(nh+2*nkvh)/ndev * dh]
    }
    res.b_attn_qkv.clear();
    
    for (auto &t : res.w_attn_out) {
        t.reset();             // Attention output weights [nh/ndev * dh, d]
    }
    res.w_attn_out.clear();
    
    // Release per-layer FFN weights and clear vectors
    for (auto &t : res.w_ffn_norm) {
        t.reset();             // FFN layer norm weights [d]
    }
    res.w_ffn_norm.clear();
    
    for (auto &t : res.w_ffn_gate_up) {
        t.reset();             // FFN gate & up weights [d, 2*di/ndev]
    }
    res.w_ffn_gate_up.clear();
    
    for (auto &t : res.w_ffn_down) {
        t.reset();             // FFN down weights [di/ndev, d]
    }
    res.w_ffn_down.clear();
    
    // Destroy InfiniCore handles and contexts
    infiniopDestroyHandle(res.handle);    // Release operation handle
    res.handle = nullptr;
    
    infinirtStreamDestroy(res.stream);    // Release execution stream  
    res.stream = nullptr;
    
    infinicclCommDestroy(res.comm);       // Release communication context
    res.comm = nullptr;
}

/*
 * Device-Level Batch Inference Function
 * 
 * Performs transformer inference for a batch of sequences on a single device.
 * Implements the complete forward pass including:
 * 1. Input embedding lookup and RoPE position encoding
 * 2. Multi-layer transformer blocks (attention + FFN)
 * 3. Output normalization and probability distribution
 * 4. Token sampling with temperature/top-k/top-p
 * 
 * This function handles distributed inference via tensor parallelism where
 * each device processes a slice of the model parameters.
 * 
 * Input Parameters:
 * - meta: Model architecture metadata (dimensions, layer count, etc.)
 * - rsrc: Device resources (weights, handles, memory pools)
 * - idev/ndev: Device index and total device count for distributed inference
 * - tokens: Input token IDs to process [ntok]
 * - ntok: Total number of tokens across all requests in batch
 * - req_lens: Length of each request [nreq] 
 * - nreq: Number of requests in the batch
 * - req_pos: Starting position for each request in KV cache [nreq]
 * - kv_caches: KV cache storage for each request [nreq][ndev][nlayer]
 * - temperature/topk/topp: Sampling parameters [nreq]
 * - output: Generated token IDs [nreq]
 * 
 * Tensor Dimension Notation:
 * - ntok: Total tokens in batch
 * - nreq: Number of requests  
 * - d: Model hidden dimension
 * - nh: Total attention heads
 * - nkvh: Total key-value heads  
 * - dh: Head dimension (d/nh)
 * - di: FFN intermediate dimension
 * - dvoc: Vocabulary size
 * - dctx: Maximum context length
 */
void inferDeviceBatch(const JiugeMeta &meta, DeviceResource &rsrc,
                      uint32_t idev, uint32_t ndev,
                      const uint32_t *tokens, uint32_t ntok,
                      const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                      struct KVCache **kv_caches,
                      const float *temperature, const uint32_t *topk, const float *topp,
                      uint32_t *output) {
    /*
     * Extract Model Dimensions and Configure for Distributed Inference
     * 
     * Key dimension calculations for tensor parallelism:
     * - nkvh: Key-value heads per device = total_kv_heads / ndev
     * - nh: Query heads per device = total_heads / ndev  
     * - ngroup: Grouped query attention ratio = nh / nkvh
     * - di: FFN intermediate dim per device = total_intermediate / ndev
     * 
     * This ensures each device handles a slice of the attention heads
     * and FFN dimensions while maintaining the same sequence processing.
     */
    auto nlayer = meta.nlayer;          // Number of transformer layers
    auto nkvh = meta.nkvh / ndev;       // KV heads per device (distributed)
    auto nh = meta.nh / ndev;           // Query heads per device (distributed) 
    auto ngroup = nh / nkvh;            // Grouped-query attention factor
    // auto dctx = meta.dctx;           // Maximum context length (unused)
    auto dh = meta.dh;                  // Head dimension
    auto d = meta.d;                    // Model hidden dimension  
    auto dt_logits = meta.dt_logits;    // Data type for logits (FP16/BF16/FP32)
    auto di = meta.di / ndev;           // FFN intermediate dim per device
    auto dvoc = meta.dvoc;              // Vocabulary size
    auto stream = rsrc.stream;          // Execution stream for async operations
    bool has_qkv_bias = rsrc.b_attn_qkv.size() > 0;  // Whether QKV has bias terms

    /*
     * Memory Buffer Allocation for Inference Pipeline
     * 
     * Allocate temporary buffers for intermediate computations.
     * All buffers use the device memory pool for efficient allocation/deallocation.
     * 
     * Buffer Tensor Shapes:
     * - logits_in/out: [ntok, d] - hidden states flowing through layers
     * - qkv_buf: [ntok, (nh + nkvh*2) * dh] - Q, K, V projections concatenated
     * - gate_up_buf: [ntok, 2*di] - FFN gate and up projections concatenated  
     * - o_buf: [ntok, nh*dh] - attention output before output projection
     * - prob_buf: [nreq, dvoc] - output probability distributions
     * - result_buf: [nreq] - sampled token IDs (device memory)
     * - result_cpu: [nreq] - sampled token IDs (host memory for output)
     */
    // Allocate buffers
    auto logits_in = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto logits_out = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto qkv_buf = Tensor::buffer(dt_logits, {ntok, (nh + nkvh * 2) * dh}, rsrc.memory_pool);
    auto gate_up_buf = Tensor::buffer(dt_logits, {ntok, 2 * di}, rsrc.memory_pool);
    auto o_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, rsrc.memory_pool);
    auto prob_buf = Tensor::buffer(dt_logits, {nreq, dvoc}, rsrc.memory_pool);
    auto result_buf = Tensor::buffer(INFINI_DTYPE_I64, {nreq}, rsrc.memory_pool);
    auto result_cpu = std::vector<int64_t>(nreq);

    /*
     * Input Preparation and Token Embedding Lookup
     * 
     * 1. Create position IDs for each token based on request positions
     * 2. Copy position IDs to device memory if needed
     * 3. Look up input embeddings for each token ID
     * 
     * Position ID Calculation:
     * For each request, position IDs are: [req_pos[i], req_pos[i]+1, ..., req_pos[i]+req_lens[i]-1]
     * This allows proper attention masking and RoPE position encoding.
     * 
     * Embedding Lookup: logits_in[i] = w_in_embd[tokens[i]] for i in [0, ntok)
     * Shape: [ntok, d] where each row is the embedding vector for token tokens[i]
     */
    // Prepare inputs
    auto batch_pos_ids = std::vector<uint32_t>(ntok);
    size_t req_start = 0;
    
    // Build position ID array: concatenate position sequences for each request
    for (uint32_t req = 0; req < nreq; req++) {
        for (uint32_t i = 0; i < req_lens[req]; i++) {
            batch_pos_ids[req_start + i] = req_pos[req] + i;
        }
        req_start += req_lens[req];
    }

    // Copy position IDs to device memory (CPU device can use host pointer directly)
    std::shared_ptr<Tensor> pos_ids_buf;
    if (rsrc.device == INFINI_DEVICE_CPU) {
        pos_ids_buf = Tensor::weight(batch_pos_ids.data(), INFINI_DTYPE_U32, {ntok});
    } else {
        pos_ids_buf = Tensor::buffer(INFINI_DTYPE_U32, {ntok}, rsrc.memory_pool);
        // Asynchronous host-to-device copy of position IDs [ntok]
        RUN_INFINI(infinirtMemcpyAsync(pos_ids_buf->data(), batch_pos_ids.data(), sizeof(uint32_t) * ntok,
                                       INFINIRT_MEMCPY_H2D, stream));
    }
    
    // Look up input embeddings: logits_in[i] = w_in_embd[tokens[i]]
    // This performs embedding table lookup for each input token
    // Shape transformation: [ntok] token IDs -> [ntok, d] embedding vectors
    for (uint32_t i = 0; i < ntok; i++) {
        RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d),                // Destination: row i of logits_in
                                       rsrc.w_in_embd->data(tokens[i] * d),   // Source: embedding for tokens[i]
                                       dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
    }

    /*
     * InfiniCore Operator Descriptor Creation and Workspace Size Calculation
     * 
     * This section creates descriptors for all compute operations needed in the inference pipeline.
     * Descriptors define the operation parameters, tensor shapes, and algorithms to use.
     * InfiniCore will use these descriptors to:
     * 1. Optimize kernel selection based on hardware capabilities
     * 2. Calculate required workspace memory for temporary computations
     * 3. Enable efficient operator reuse across multiple calls
     * 
     * The workspace size is calculated as the maximum across all operations to ensure
     * a single allocation can handle any intermediate computation.
     */
    // Prepare operators and workspace
    size_t workspace_size = 0, temp_size = 0;
    
    /*
     * RMS Normalization Descriptor for Attention and FFN Layers
     * 
     * RMSNorm formula: y = x / √(mean(x²) + ε) * γ
     * Where γ is the learned scale parameter (weight)
     * 
     * Input/Output shapes: [ntok, d] -> [ntok, d]
     * Weight shape: [d]
     * 
     * This descriptor is reused for all layer normalizations in the model.
     */
    infiniopRMSNormDescriptor_t desc_norm;
    RUN_INFINI(infiniopCreateRMSNormDescriptor(
        rsrc.handle, &desc_norm, logits_in->desc(),     // Input: [ntok, d]
        logits_out->desc(), rsrc.w_attn_norm[0]->desc(), // Output: [ntok, d], Weight: [d]
        meta.epsilon));                                   // Normalization epsilon
    RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_norm, &workspace_size));
    workspace_size = std::max(workspace_size, temp_size);
    /*
     * Attention Mechanism Descriptors
     * 
     * The attention computation involves several matrix operations:
     * 1. QKV projection: X -> Q, K, V via matrix multiplication
     * 2. RoPE position encoding on Q and K  
     * 3. Attention computation: softmax(QK^T/√d_k) * V
     * 4. Output projection: O -> final attention output
     */
    
    // GEMM descriptors for QKV projection and attention output
    infiniopGemmDescriptor_t desc_attn_qkv, desc_attn_o;
    infiniopRearrangeDescriptor_t desc_qkv_bias;
    
    // QKV bias addition if present (optional operation)
    if (has_qkv_bias) {
        // Rearrange/broadcast bias to match QKV buffer shape
        // Bias shape: [(nh + 2*nkvh)/ndev * dh] -> [ntok, (nh + 2*nkvh)/ndev * dh]
        RUN_INFINI(infiniopCreateRearrangeDescriptor(
            rsrc.handle, &desc_qkv_bias, qkv_buf->desc(),
            TensorDesc::create(dt_logits, {ntok, (nh + nkvh * 2) * dh}, {0, 1})->desc()));
    }
    
    /*
     * QKV Projection GEMM: logits_in * w_attn_qkv -> qkv_buf
     * 
     * Matrix multiplication: Y = X * W  
     * Input X: [ntok, d] - normalized hidden states
     * Weight W: [d, (nh + 2*nkvh)/ndev * dh] - QKV projection weights  
     * Output Y: [ntok, (nh + 2*nkvh)/ndev * dh] - concatenated Q, K, V projections
     * 
     * The output contains Q, K, V projections concatenated along the last dimension:
     * - Q: [ntok, nh/ndev * dh]      (query projections)
     * - K: [ntok, nkvh/ndev * dh]    (key projections)  
     * - V: [ntok, nkvh/ndev * dh]    (value projections)
     */
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_attn_qkv, qkv_buf->desc(),
        logits_in->desc(), rsrc.w_attn_qkv[0]->desc()));
        
    /*
     * Attention Output Projection GEMM: o_buf * w_attn_out -> logits_in
     * 
     * Matrix multiplication: Y = X * W
     * Input X: [ntok, nh/ndev * dh] - attention output from all heads on this device
     * Weight W: [nh/ndev * dh, d] - output projection weights
     * Output Y: [ntok, d] - projected attention output (will be accumulated across devices)
     */
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_attn_o, logits_in->desc(),
        o_buf->desc(), rsrc.w_attn_out[0]->desc()));
        
    // Calculate workspace requirements for both GEMM operations
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_attn_qkv, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_attn_o, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    /*
     * RoPE (Rotary Position Embedding) Descriptors
     * 
     * RoPE applies rotational position encoding to query and key vectors:
     * For each position pos and dimension pair (i, i+d/2):
     * q'[i] = q[i] * cos(pos/θ^(i/d)) - q[i+d/2] * sin(pos/θ^(i/d))
     * q'[i+d/2] = q[i] * sin(pos/θ^(i/d)) + q[i+d/2] * cos(pos/θ^(i/d))
     * 
     * This encoding allows the model to understand relative positions between tokens.
     */
    infiniopRoPEDescriptor_t desc_rope_q, desc_rope_k;
    
    // Split QKV buffer to access Q and K separately
    // qkv_buf shape: [ntok, (nh + 2*nkvh) * dh] -> [ntok, nh + 2*nkvh, dh]
    qkv_buf->dimSplit(1, {nh + nkvh * 2, dh}); 
    
    // Extract query and key projections from concatenated QKV buffer
    auto qkv_buf_q = qkv_buf->slice(1, 0, nh);           // Q: [ntok, nh, dh]
    auto qkv_buf_k = qkv_buf->slice(1, nh, nkvh);        // K: [ntok, nkvh, dh]
    
    /*
     * RoPE descriptor for Query vectors
     * 
     * Applies position encoding to queries based on position IDs and precomputed sin/cos tables
     * Input/Output: [ntok, nh, dh] -> [ntok, nh, dh]
     * Position IDs: [ntok] - position of each token in its sequence
     * Sin/Cos tables: [dctx, dh/2] - precomputed trigonometric values
     */
    RUN_INFINI(infiniopCreateRoPEDescriptor(
        rsrc.handle, &desc_rope_q, qkv_buf_q->desc(), qkv_buf_q->desc(),
        pos_ids_buf->desc(), rsrc.sin_table->desc(),
        rsrc.cos_table->desc()));
    RUN_INFINI(infiniopGetRoPEWorkspaceSize(desc_rope_q, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    
    /*
     * RoPE descriptor for Key vectors
     * 
     * Applies the same position encoding to keys
     * Input/Output: [ntok, nkvh, dh] -> [ntok, nkvh, dh]
     */
    RUN_INFINI(infiniopCreateRoPEDescriptor(
        rsrc.handle, &desc_rope_k, qkv_buf_k->desc(), qkv_buf_k->desc(),
        pos_ids_buf->desc(), rsrc.sin_table->desc(),
        rsrc.cos_table->desc()));
    RUN_INFINI(infiniopGetRoPEWorkspaceSize(desc_rope_k, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    /*
     * Per-Request Attention Inner Loop Descriptors
     * 
     * Since each request in the batch may have different sequence lengths and KV cache states,
     * we need separate descriptors for each request's attention computation.
     * 
     * The attention mechanism for each request involves:
     * 1. Rearranging Q and K for grouped-query attention (GQA)
     * 2. Computing attention scores: QK^T (scaled dot-product attention)
     * 3. Applying causal softmax with masking
     * 4. Computing attention output: attention_weights * V
     * 5. Rearranging output back to standard format
     * 
     * Key optimizations:
     * - Grouped Query Attention: Multiple query heads can share KV heads (ngroup = nh/nkvh)
     * - KV caching: Past key-value pairs are stored and reused
     * - Causal masking: Future tokens cannot attend to past tokens
     */
    // attention inner
    auto desc_kv_rearranges = std::vector<infiniopRearrangeDescriptor_t>(nreq);
    auto desc_q_rearranges = std::vector<infiniopRearrangeDescriptor_t>(nreq);
    auto desc_qk_gemms = std::vector<infiniopGemmDescriptor_t>(nreq);
    auto desc_qk_softmaxs = std::vector<infiniopCausalSoftmaxDescriptor_t>(nreq);
    auto desc_attn_v_gemms = std::vector<infiniopGemmDescriptor_t>(nreq);
    auto desc_attn_v_rearranges = std::vector<infiniopRearrangeDescriptor_t>(nreq);
    
    size_t token_offset = 0;   // Track position in batch for current request
    size_t max_qk_size = 0;    // Maximum QK matrix size for buffer allocation
    size_t max_seq_len = 0;    // Maximum sequence length for buffer allocation
    
    // Prepare output buffer for attention heads: [ntok, nh, dh]
    o_buf->dimSplit(1, {nh, dh});
    /*
     * Create Descriptors for Each Request's Attention Computation
     * 
     * Each request may have different sequence lengths and past KV cache lengths,
     * requiring individual descriptors for optimal memory layout and computation.
     */
    for (uint32_t req = 0; req < nreq; req++) {
        auto past_len = req_pos[req];        // Number of tokens already in KV cache
        auto seq_len = req_lens[req];        // Current sequence length to process
        auto total_len = past_len + seq_len; // Total sequence length in KV cache
        
        /*
         * Extract per-request tensor slices from batch tensors
         * 
         * Tensor shapes for this request:
         * - o: [seq_len, nh, dh] - attention output for this request
         * - q: [seq_len, nh, dh] - query vectors for this request  
         * - k: [seq_len, nkvh, dh] - key vectors for this request
         * - v: [seq_len, nkvh, dh] - value vectors for this request (used later)
         */
        auto o = o_buf->slice({{0, token_offset, seq_len}});
        auto q = qkv_buf->slice({{0, token_offset, seq_len}, {1, 0, nh}});
        auto k = qkv_buf->slice({{0, token_offset, seq_len}, {1, nh, nkvh}});
        // auto v = qkv_buf->slice({{0, token_offset, seq_len}, {1, nh + nkvh, nkvh}});
        
        /*
         * KV Cache Tensor Configuration
         * 
         * KV cache stores past key-value pairs for efficient autoregressive generation.
         * Shape: [total_len, nkvh, dh] stored as [nkvh, dh, total_len] in memory
         * 
         * full_kv: Complete KV cache including past + current tokens [nkvh, dh, total_len]
         * cache_kv: Slice for storing current keys/values [nkvh, dh, seq_len]
         */
        // kv cache tensors can share the same descriptor
        // [nkvh, dh, total_len]
        auto full_kv = kv_caches[req]->k[idev][0]->slice(0, 0, total_len)->permute({1, 2, 0});
        auto cache_kv = kv_caches[req]->k[idev][0]->slice(0, past_len, seq_len);

        /*
         * KV Rearrange Descriptor: Store current K/V in cache
         * 
         * Transforms current keys/values to cache storage format:
         * k: [seq_len, nkvh, dh] -> cache_kv: [seq_len, nkvh, dh] (different memory layout)
         */
        RUN_INFINI(infiniopCreateRearrangeDescriptor(rsrc.handle, &desc_kv_rearranges[req],
                                                     cache_kv->desc(), k->desc()));

        /*
         * Query Rearrange for Grouped Query Attention (GQA)
         * 
         * Reshape queries to enable efficient GQA computation:
         * q: [seq_len, nh, dh] -> [seq_len, nkvh, ngroup, dh] -> [nkvh, ngroup, seq_len, dh]
         * 
         * This layout allows each KV head to attend to multiple query heads (ngroup heads per KV head)
         */
        // [nkvh, ngroup, seq_len, dh]
        q->dimSplit(1, {nkvh, ngroup})->permute({1, 2, 0, 3});
        auto q_t = TensorDesc::create(dt_logits, {nkvh, ngroup, seq_len, dh});
        // [seq_len, nkvh, ngroup, dh] -> [nkvh, ngroup, seq_len, dh]
        RUN_INFINI(infiniopCreateRearrangeDescriptor(rsrc.handle, &desc_q_rearranges[req],
                                                     q_t->desc(), q->desc()));
        
        /*
         * Attention Value Rearrange Descriptor
         * 
         * After computing attention weights * values, rearrange back to standard format:
         * [nkvh, ngroup, seq_len, dh] -> [seq_len, nkvh, ngroup, dh] -> [seq_len, nh, dh]
         */
        // [nkvh, ngroup, seq_len, dh] -> [seq_len, nkvh, ngroup, dh]
        auto attn_v_t = q_t;
        auto attn_v = TensorDesc::createWithOrder(dt_logits, {nkvh, ngroup, seq_len, dh}, {1, 2, 0, 3});
        RUN_INFINI(infiniopCreateRearrangeDescriptor(rsrc.handle, &desc_attn_v_rearranges[req],
                                                     attn_v->desc(), attn_v_t->desc()));
        
        /*
         * QK Attention Score Computation: Q * K^T / √d_k
         * 
         * Matrix multiplication to compute attention scores:
         * Q: [nkvh, ngroup * seq_len, dh] (reshaped for batched computation)
         * K^T: [nkvh, dh, total_len] (full KV cache transposed)
         * QK: [nkvh, ngroup * seq_len, total_len] (attention scores before softmax)
         * 
         * The scaling factor 1/√d_k is applied later during the GEMM operation.
         */
        q_t = TensorDesc::create(dt_logits, {nkvh, ngroup * seq_len, dh});
        auto qk = TensorDesc::create(dt_logits, {nkvh, ngroup * seq_len, total_len});
        max_qk_size = std::max(max_qk_size, size_t(seq_len * total_len));
        max_seq_len = std::max(max_seq_len, size_t(seq_len));
        RUN_INFINI(infiniopCreateGemmDescriptor(
            rsrc.handle, &desc_qk_gemms[req], qk->desc(), q_t->desc(), full_kv->desc()));
        RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_qk_gemms[req], &temp_size));
        workspace_size = std::max(workspace_size, temp_size);

        /*
         * Attention Value Computation: attention_weights * V
         * 
         * Matrix multiplication with attention-weighted values:
         * attention_weights: [nkvh, ngroup * seq_len, total_len] (after softmax)
         * V: [nkvh, total_len, dh] (full value cache)
         * output: [nkvh, ngroup * seq_len, dh] (attention output)
         */
        // [nkvh, total_len, dh]
        auto full_v = kv_caches[req]->v[idev][0]->slice(0, 0, total_len)->permute({1, 0, 2});
        RUN_INFINI(infiniopCreateGemmDescriptor(
            rsrc.handle, &desc_attn_v_gemms[req], q_t->desc(), qk->desc(), full_v->desc()));
        RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_attn_v_gemms[req], &temp_size));
        workspace_size = std::max(workspace_size, temp_size);

        /*
         * Causal Softmax with Attention Masking
         * 
         * Applies softmax to attention scores with causal masking (lower triangular).
         * Shape: [nkvh * ngroup, seq_len, total_len]
         * 
         * Causal mask ensures each token can only attend to previous tokens and itself,
         * preventing information leakage from future tokens during autoregressive generation.
         */
        qk = TensorDesc::create(dt_logits, {nkvh * ngroup, seq_len, total_len});
        RUN_INFINI(infiniopCreateCausalSoftmaxDescriptor(
            rsrc.handle, &desc_qk_softmaxs[req], qk->desc(), qk->desc()));
        RUN_INFINI(infiniopGetCausalSoftmaxWorkspaceSize(desc_qk_softmaxs[req], &temp_size));
        workspace_size = std::max(workspace_size, temp_size);

        token_offset += seq_len;
    }
    /*
     * Allocate Attention Intermediate Buffers
     * 
     * These buffers store intermediate results during attention computation.
     * Sizes are based on maximum requirements across all requests in the batch.
     * 
     * Buffer shapes:
     * - qk_buf: [nh, max_qk_size] - attention scores (QK^T) for largest request
     * - rearrange_q_buf: [nkvh, ngroup * max_seq_len, dh] - rearranged queries
     * - attn_val_buf: [nh, max_seq_len, dh] - attention output values
     */
    auto qk_buf = Tensor::buffer(dt_logits, {nh, max_qk_size}, rsrc.memory_pool);
    auto rearrange_q_buf = Tensor::buffer(dt_logits, {nkvh, ngroup * max_seq_len, dh}, rsrc.memory_pool);
    auto attn_val_buf = Tensor::buffer(dt_logits, {nh, max_seq_len, dh}, rsrc.memory_pool);

    /*
     * Feed-Forward Network (FFN) Descriptors
     * 
     * The FFN block implements the SwiGLU activation function:
     * FFN(x) = (Swish(x * W_gate) ⊙ (x * W_up)) * W_down
     * Where Swish(x) = x * sigmoid(x) and ⊙ is element-wise multiplication
     * 
     * This involves three matrix multiplications:
     * 1. Gate & Up projections: x -> [gate, up] (computed together)
     * 2. SwiGLU activation: gate * swish(up) 
     * 3. Down projection: activated -> output
     */
    // MLP descriptors
    infiniopGemmDescriptor_t desc_ffn_gate_up, desc_ffn_down;
    infiniopSwiGLUDescriptor_t desc_swiglu;
    
    /*
     * FFN Gate & Up Projection GEMM: logits_out * w_ffn_gate_up -> gate_up_buf
     * 
     * Matrix multiplication: Y = X * W
     * Input X: [ntok, d] - normalized hidden states from attention
     * Weight W: [d, 2*di/ndev] - concatenated gate and up projection weights
     * Output Y: [ntok, 2*di/ndev] - gate and up projections concatenated
     * 
     * The output contains both gate and up projections:
     * - gate: [ntok, di/ndev] - gating values for SwiGLU
     * - up: [ntok, di/ndev] - up-projected values for SwiGLU
     */
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_ffn_gate_up, gate_up_buf->desc(),
        logits_out->desc(), rsrc.w_ffn_gate_up[0]->desc()));
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_ffn_gate_up, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    
    // Split gate_up_buf into gate and up components for SwiGLU
    auto gate_buf = gate_up_buf->slice(1, 0, di);      // Gate: [ntok, di/ndev]
    auto up_buf = gate_up_buf->slice(1, di, di);       // Up: [ntok, di/ndev]
    
    /*
     * SwiGLU Activation Function
     * 
     * Computes: output = gate * swish(up) = gate * (up * sigmoid(up))
     * Input gate: [ntok, di/ndev] - gating values
     * Input up: [ntok, di/ndev] - values to be gated
     * Output: [ntok, di/ndev] - activated values (stored back in gate_buf)
     * 
     * SwiGLU provides better performance than standard ReLU activations
     * in transformer FFN blocks by using learnable gating mechanisms.
     */
    RUN_INFINI(infiniopCreateSwiGLUDescriptor(
        rsrc.handle, &desc_swiglu, gate_buf->desc(), up_buf->desc(), gate_buf->desc()));
    RUN_INFINI(infiniopGetSwiGLUWorkspaceSize(desc_swiglu, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    
    /*
     * FFN Down Projection GEMM: gate_buf * w_ffn_down -> logits_in
     * 
     * Matrix multiplication: Y = X * W  
     * Input X: [ntok, di/ndev] - activated values from SwiGLU
     * Weight W: [di/ndev, d] - down projection weights
     * Output Y: [ntok, d] - projected back to model dimension
     * 
     * This completes the FFN computation and the result will be added
     * to the residual connection from the attention block.
     */
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_ffn_down, logits_in->desc(),
        gate_buf->desc(), rsrc.w_ffn_down[0]->desc()));
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_ffn_down, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);

    /*
     * Output Generation and Token Sampling Descriptors
     * 
     * After all transformer layers, we need to:
     * 1. Apply final layer normalization to the last token of each request
     * 2. Project to vocabulary space to get logits for next token prediction
     * 3. Apply sampling (temperature, top-k, top-p) to select next tokens
     */
    // Output and sample
    infiniopRMSNormDescriptor_t desc_norm_out;
    
    /*
     * Final Output Normalization
     * 
     * Apply RMSNorm to the last token's hidden state for each request.
     * This normalizes the final representation before projecting to vocabulary.
     * 
     * Input/Output shape: [1, d] -> [1, d] (processed one request at a time)
     * Weight shape: [d] - final layer normalization parameters
     */
    RUN_INFINI(infiniopCreateRMSNormDescriptor(
        rsrc.handle, &desc_norm_out, logits_out->slice(0, 0, 1)->desc(),
        logits_out->slice(0, 0, 1)->desc(),
        rsrc.w_out_norm->desc(), meta.epsilon));
    RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_norm_out, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    
    /*
     * Language Model Head Projection: hidden_states -> vocabulary_logits
     * 
     * Project normalized hidden states to vocabulary space for next token prediction.
     * 
     * Matrix multiplication: Y = X * W
     * Input X: [nreq, d] - final hidden states for each request
     * Weight W: [d, dvoc] - language model head weights (often tied to input embeddings)
     * Output Y: [nreq, dvoc] - logits over vocabulary for each request
     * 
     * These logits represent unnormalized log probabilities for each possible next token.
     */
    infiniopGemmDescriptor_t desc_out_embd;
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_out_embd, prob_buf->desc(),
        logits_out->slice(0, 0, nreq)->desc(),
        rsrc.w_out_embd->desc()));
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_out_embd, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    
    /*
     * Random Sampling Descriptor
     * 
     * Performs temperature scaling, top-k filtering, top-p (nucleus) sampling
     * to select the next token from the probability distribution.
     * 
     * Sampling process:
     * 1. Apply temperature scaling: logits = logits / temperature
     * 2. Apply top-k filtering: keep only k highest probability tokens
     * 3. Apply top-p filtering: keep tokens until cumulative probability >= p
     * 4. Sample from the filtered distribution
     * 
     * Input: [dvoc] - logits over vocabulary for one request
     * Output: scalar int64 - selected token ID
     */
    infiniopRandomSampleDescriptor_t desc_sample;
    RUN_INFINI(infiniopCreateRandomSampleDescriptor(
        rsrc.handle, &desc_sample,
        TensorDesc::create(INFINI_DTYPE_I64, {}, {})->desc(),     // Output: scalar token ID
        TensorDesc::create(dt_logits, {dvoc}, {1})->desc()));     // Input: [dvoc] logits
    RUN_INFINI(infiniopGetRandomSampleWorkspaceSize(desc_sample, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    
    /*
     * Workspace Allocation
     * 
     * Allocate a single workspace buffer that can handle the maximum memory
     * requirement across all operations. This avoids frequent allocations
     * during inference and ensures efficient memory usage.
     */
    // Allocate workspace
    std::shared_ptr<Storage> workspace_storage = Storage::createFromPool(workspace_size, rsrc.memory_pool);
    void *workspace = workspace_storage->memory();

    /*
     * ==================================================================================
     * MAIN TRANSFORMER INFERENCE COMPUTATION LOOP
     * ==================================================================================
     * 
     * This section executes the actual forward pass through all transformer layers.
     * Each layer consists of:
     * 1. Multi-head attention with residual connection
     * 2. Feed-forward network with residual connection
     * 
     * The computation follows the standard transformer architecture:
     * x = x + Attention(LayerNorm(x))
     * x = x + FFN(LayerNorm(x))
     * 
     * For distributed inference, attention and FFN outputs are accumulated
     * across devices via all-reduce operations.
     */
    // Compute
    for (uint32_t layer = 0; layer < nlayer; layer++) {
        /*
         * ============================================================================
         * MULTI-HEAD ATTENTION BLOCK
         * ============================================================================
         */
        // 1. Attention
        
        /*
         * Pre-Attention Layer Normalization
         * 
         * Apply RMSNorm to input hidden states before attention computation.
         * This follows the "Pre-LN" transformer architecture for better training stability.
         * 
         * Formula: y = x / √(mean(x²) + ε) * γ
         * Input: logits_in [ntok, d] - hidden states from previous layer/embeddings
         * Output: logits_out [ntok, d] - normalized hidden states for attention
         * Weight: w_attn_norm[layer] [d] - learnable scale parameters
         */
        // rms norm
        RUN_INFINI(infiniopRMSNorm(
            desc_norm, workspace, workspace_size,
            logits_out->data(), logits_in->data(),
            rsrc.w_attn_norm[layer]->data(), stream));
            
        /*
         * QKV Projection with Optional Bias
         * 
         * Transform normalized hidden states to query, key, and value projections.
         * If bias is present, it's added via a rearrange operation before the GEMM.
         * 
         * Matrix operation: QKV = X * W_qkv + b_qkv (if bias present)
         * Input: logits_out [ntok, d] - normalized hidden states
         * Weight: w_attn_qkv[layer] [d, (nh + 2*nkvh)/ndev * dh] - QKV projection weights
         * Bias: b_attn_qkv[layer] [(nh + 2*nkvh)/ndev * dh] - optional bias (broadcasted)
         * Output: qkv_buf [ntok, (nh + 2*nkvh)/ndev * dh] - concatenated Q, K, V
         */
        // qkv_proj
        if (has_qkv_bias) {
            // Broadcast bias to match batch dimension: [heads*dh] -> [ntok, heads*dh]
            RUN_INFINI(infiniopRearrange(
                desc_qkv_bias,
                qkv_buf->data(), rsrc.b_attn_qkv[layer]->data(), stream));
        }
        // QKV projection: X * W + bias (bias beta=1.0 if present, 0.0 otherwise)
        RUN_INFINI(infiniopGemm(
            desc_attn_qkv, workspace, workspace_size,
            qkv_buf->data(), logits_out->data(),
            rsrc.w_attn_qkv[layer]->data(), 1.0, has_qkv_bias ? 1.0 : 0.0, stream));
            
        /*
         * Rotary Position Embedding (RoPE) Application
         * 
         * Apply position-dependent rotations to query and key vectors.
         * This encodes relative position information directly into the attention mechanism.
         * 
         * RoPE formula for each position pos and dimension pair (i, i+d/2):
         * q'[i] = q[i] * cos(pos/θ^(i/d)) - q[i+d/2] * sin(pos/θ^(i/d))
         * q'[i+d/2] = q[i] * sin(pos/θ^(i/d)) + q[i+d/2] * cos(pos/θ^(i/d))
         * 
         * Applied separately to queries and keys using precomputed sin/cos tables.
         */
        // rope
        RUN_INFINI(infiniopRoPE(
            desc_rope_q, workspace, workspace_size,
            qkv_buf->data(), qkv_buf->data(),                // Q in-place: [ntok, nh, dh]
            pos_ids_buf->data(),                             // Position IDs: [ntok]
            rsrc.sin_table->data(),                          // Sin table: [dctx, dh/2]
            rsrc.cos_table->data(), stream));                // Cos table: [dctx, dh/2]
        RUN_INFINI(infiniopRoPE(
            desc_rope_k, workspace, workspace_size,
            qkv_buf->data(nh * dh), qkv_buf->data(nh * dh), // K in-place: [ntok, nkvh, dh]
            pos_ids_buf->data(),
            rsrc.sin_table->data(),
            rsrc.cos_table->data(),
            stream));

        /*
         * Per-Request Attention Computation with KV Caching
         * 
         * Process each request individually due to different sequence lengths
         * and KV cache states. This implements efficient autoregressive generation
         * with incremental KV cache updates.
         */
        size_t token_offset = 0;
        for (uint32_t req = 0; req < nreq; req++) {
            auto past_len = req_pos[req];    // Tokens already in KV cache
            auto seq_len = req_lens[req];    // Current sequence length to process
            
            // Extract per-request tensor slices from batch
            auto o = o_buf->slice({{0, token_offset, seq_len}});                              // [seq_len, nh, dh]
            auto q = qkv_buf->slice({{0, token_offset, seq_len}, {1, 0, nh}});               // [seq_len, nh, dh]
            auto k = qkv_buf->slice({{0, token_offset, seq_len}, {1, nh, nkvh}});            // [seq_len, nkvh, dh]
            auto v = qkv_buf->slice({{0, token_offset, seq_len}, {1, nh + nkvh, nkvh}});     // [seq_len, nkvh, dh]
            
            /*
             * ================================================================
             * SCALED DOT-PRODUCT ATTENTION WITH KV CACHING
             * ================================================================
             * 
             * Implements the attention mechanism: Attention(Q,K,V) = softmax(QK^T/√d_k)V
             * with efficient KV caching for autoregressive generation.
             */
            // self attention
            
            /*
             * KV Cache Update
             * 
             * Store current keys and values in the KV cache for future use.
             * The cache allows reusing computations from previous tokens
             * during autoregressive generation.
             * 
             * Cache storage format: [total_len, nkvh, dh] where total_len
             * includes both past and current tokens.
             */
            // concat
            RUN_INFINI(infiniopRearrange(
                desc_kv_rearranges[req],
                kv_caches[req]->k[idev][layer]->data(past_len * nkvh * dh),  // K cache: append at past_len
                k->data(), stream));                                          // Source: current K [seq_len, nkvh, dh]
            RUN_INFINI(infiniopRearrange(
                desc_kv_rearranges[req],
                kv_caches[req]->v[idev][layer]->data(past_len * nkvh * dh),  // V cache: append at past_len  
                v->data(), stream));                                          // Source: current V [seq_len, nkvh, dh]
                
            /*
             * Query Rearrangement for Grouped Query Attention
             * 
             * Reshape queries to enable efficient GQA computation where
             * multiple query heads share each key-value head.
             * 
             * Transformation: [seq_len, nh, dh] -> [nkvh, ngroup, seq_len, dh]
             * where ngroup = nh/nkvh (queries per KV head)
             */
            // qk
            RUN_INFINI(infiniopRearrange(desc_q_rearranges[req], rearrange_q_buf->data(), q->data(), stream));
            
            /*
             * Attention Score Computation: Q * K^T / √d_k
             * 
             * Compute scaled dot-product attention scores between queries and all keys
             * in the cache (including past + current keys).
             * 
             * Matrix multiplication:
             * Q: [nkvh, ngroup * seq_len, dh] - rearranged queries
             * K^T: [nkvh, dh, total_len] - all keys in cache (transposed)
             * Output: [nkvh, ngroup * seq_len, total_len] - attention scores
             * 
             * Scale factor: 1/√d_k for numerical stability
             */
            RUN_INFINI(infiniopGemm(
                desc_qk_gemms[req], workspace, workspace_size,
                qk_buf->data(), rearrange_q_buf->data(), kv_caches[req]->k[idev][layer]->data(), 
                1. / sqrt(dh), 0.0, stream));  // Scale by 1/√d_k
                
            /*
             * Causal Softmax with Attention Masking
             * 
             * Apply softmax to attention scores with causal masking to prevent
             * attending to future tokens. The causal mask ensures that token at
             * position i can only attend to tokens at positions 0...i.
             * 
             * Input/Output: [nkvh * ngroup, seq_len, total_len]
             * Causal mask: lower triangular matrix of 1s (attend) and 0s (mask)
             */
            // softmax
            RUN_INFINI(infiniopCausalSoftmax(
                desc_qk_softmaxs[req], workspace, workspace_size,
                qk_buf->data(), qk_buf->data(), stream));
                
            /*
             * Attention Output Computation: attention_weights * V
             * 
             * Apply attention weights to value vectors to compute final attention output.
             * 
             * Matrix multiplication:
             * attention_weights: [nkvh, ngroup * seq_len, total_len] - softmaxed scores
             * V: [nkvh, total_len, dh] - all values in cache
             * Output: [nkvh, ngroup * seq_len, dh] - weighted value combinations
             */
            // attn val
            RUN_INFINI(infiniopGemm(
                desc_attn_v_gemms[req], workspace, workspace_size,
                attn_val_buf->data(), qk_buf->data(), kv_caches[req]->v[idev][layer]->data(), 
                1.0, 0.0, stream));
                
            /*
             * Output Rearrangement
             * 
             * Transform attention output back to standard format for downstream processing.
             * 
             * Transformation: [nkvh, ngroup * seq_len, dh] -> [seq_len, nh, dh]
             * This undoes the GQA reshaping and prepares output for the next layer.
             */
            // rearrange attn val
            RUN_INFINI(infiniopRearrange(
                desc_attn_v_rearranges[req],
                o->data(),                    // Output: [seq_len, nh, dh]
                attn_val_buf->data(), stream)); // Input: [nkvh, ngroup * seq_len, dh]

            token_offset += seq_len;
        }
        /*
         * Attention Output Projection and Residual Connection
         * 
         * Project attention outputs back to model dimension and add residual connection.
         * In distributed inference, only device 0 adds the residual connection to avoid
         * double-counting across devices.
         * 
         * Matrix operation: Y = X * W + (residual if idev == 0 else 0)
         * Input: o_buf [ntok, nh/ndev * dh] - attention outputs from this device
         * Weight: w_attn_out[layer] [nh/ndev * dh, d] - output projection weights  
         * Output: logits_in [ntok, d] - projected output with residual connection
         */
        // o_proj
        RUN_INFINI(infiniopGemm(
            desc_attn_o, workspace, workspace_size,
            logits_in->data(), o_buf->data(),
            rsrc.w_attn_out[layer]->data(), 
            1.0, idev == 0 ? 1.0 : 0.0, stream)); // Residual: only rank 0 adds original input

        /*
         * Distributed All-Reduce for Multi-Device Inference
         * 
         * Sum attention outputs across all devices to complete the distributed computation.
         * Each device computed a slice of the attention heads, and the results must be
         * combined to get the complete attention output.
         * 
         * Operation: logits_in = sum(logits_in_device_i) for i in [0, ndev)
         * This synchronizes all devices and ensures consistent state across the cluster.
         */
        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));  // Synchronize after communication
        }
        /*
         * ============================================================================
         * FEED-FORWARD NETWORK (FFN) BLOCK WITH SwiGLU ACTIVATION
         * ============================================================================
         */
        // 2. FFN
        
        /*
         * Pre-FFN Layer Normalization
         * 
         * Apply RMSNorm to attention outputs before FFN computation.
         * 
         * Formula: y = x / √(mean(x²) + ε) * γ
         * Input: logits_in [ntok, d] - attention output + residual
         * Output: logits_out [ntok, d] - normalized for FFN processing
         * Weight: w_ffn_norm[layer] [d] - learnable scale parameters
         */
        // rms_norm
        RUN_INFINI(infiniopRMSNorm(
            desc_norm, workspace, workspace_size,
            logits_out->data(), logits_in->data(),
            rsrc.w_ffn_norm[layer]->data(), stream));
            
        /*
         * FFN Gate & Up Projections
         * 
         * Simultaneously compute gate and up projections for SwiGLU activation.
         * The weight matrix contains both projections concatenated.
         * 
         * Matrix operation: [gate, up] = X * W_gate_up
         * Input: logits_out [ntok, d] - normalized hidden states
         * Weight: w_ffn_gate_up[layer] [d, 2*di/ndev] - gate & up weights concatenated
         * Output: gate_up_buf [ntok, 2*di/ndev] - [gate_proj, up_proj] concatenated
         */
        RUN_INFINI(infiniopGemm(
            desc_ffn_gate_up, workspace, workspace_size,
            gate_up_buf->data(), logits_out->data(), rsrc.w_ffn_gate_up[layer]->data(),
            1.0, 0.0, stream));
            
        /*
         * SwiGLU Activation Function
         * 
         * Apply SwiGLU activation: output = gate * swish(up) = gate * (up * sigmoid(up))
         * This gated activation provides better performance than standard ReLU.
         * 
         * Input gate: [ntok, di/ndev] - gating values
         * Input up: [ntok, di/ndev] - values to be gated
         * Output: [ntok, di/ndev] - activated values (stored in gate_buf)
         * 
         * The swish function (x * sigmoid(x)) provides smooth, differentiable gating.
         */
        RUN_INFINI(infiniopSwiGLU(
            desc_swiglu, workspace, workspace_size,
            gate_buf->data(), up_buf->data(), gate_buf->data(), stream));
            
        /*
         * FFN Down Projection and Residual Connection
         * 
         * Project activated FFN output back to model dimension and add residual.
         * Like attention, only device 0 adds the residual in distributed inference.
         * 
         * Matrix operation: Y = X * W + (residual if idev == 0 else 0)
         * Input: gate_buf [ntok, di/ndev] - SwiGLU activated values
         * Weight: w_ffn_down[layer] [di/ndev, d] - down projection weights
         * Output: logits_in [ntok, d] - FFN output with residual connection
         */
        RUN_INFINI(infiniopGemm(
            desc_ffn_down, workspace, workspace_size,
            logits_in->data(), gate_buf->data(),
            rsrc.w_ffn_down[layer]->data(), 
            1.0, idev == 0 ? 1.0 : 0.0, stream)); // Residual: only rank 0 adds original input

        /*
         * Distributed All-Reduce for FFN Outputs
         * 
         * Sum FFN outputs across all devices to complete the distributed computation.
         * Each device computed a slice of the intermediate FFN dimension.
         */
        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));  // Synchronize after communication
        }
    }
    
    /*
     * ==================================================================================
     * OUTPUT GENERATION AND TOKEN SAMPLING
     * ==================================================================================
     * 
     * After processing through all transformer layers, generate next tokens by:
     * 1. Applying final layer normalization to last token of each request
     * 2. Projecting to vocabulary space to get logits
     * 3. Sampling next tokens using temperature, top-k, and top-p filtering
     * 
     * Only device 0 performs sampling to avoid duplicate computations.
     */
    // Sample and Output
    if (idev == 0) {
        /*
         * Final Layer Normalization for Each Request
         * 
         * Apply RMSNorm to the last token's hidden state for each request.
         * The last token is used for next token prediction in autoregressive generation.
         * 
         * For each request, extract the last token's hidden state:
         * - token_offset tracks the cumulative position in the batch
         * - The last token is at position (token_offset - 1) after processing req_lens[req] tokens
         */
        size_t token_offset = 0;
        for (uint32_t req = 0; req < nreq; req++) {
            auto seq_len = req_lens[req];
            token_offset += seq_len;
            
            /*
             * Normalize the last token's hidden state for this request
             * 
             * Input: logits_in[(token_offset-1)*d : (token_offset)*d] - last token's hidden state [d]
             * Output: logits_out[req*d : (req+1)*d] - normalized state for vocab projection [d]
             * Weight: w_out_norm [d] - final layer norm parameters
             */
            RUN_INFINI(infiniopRMSNorm(
                desc_norm_out, workspace, workspace_size,
                logits_out->data(req * d),                      // Output: [d] for request req
                logits_in->data((token_offset - 1) * d),        // Input: last token [d]
                rsrc.w_out_norm->data(), stream));
        }
        
        /*
         * Language Model Head Projection
         * 
         * Project normalized final hidden states to vocabulary space to get logits
         * for next token prediction.
         * 
         * Matrix operation: logits = hidden_states * W_lm_head
         * Input: logits_out [nreq, d] - normalized final hidden states
         * Weight: w_out_embd [d, dvoc] - language model head (often tied to input embeddings)
         * Output: prob_buf [nreq, dvoc] - unnormalized logits over vocabulary
         */
        RUN_INFINI(infiniopGemm(
            desc_out_embd, workspace, workspace_size,
            prob_buf->data(), logits_out->data(),
            rsrc.w_out_embd->data(), 1.0, 0.0, stream));
            
        /*
         * Token Sampling with Temperature and Filtering
         * 
         * For each request, sample the next token from the probability distribution
         * using temperature scaling, top-k filtering, and top-p (nucleus) sampling.
         * 
         * Sampling process:
         * 1. Apply temperature scaling: logits = logits / temperature
         * 2. Apply top-k: keep only the k highest probability tokens
         * 3. Apply top-p: keep tokens until cumulative probability >= p
         * 4. Sample from the filtered distribution using random value
         */
        std::random_device _rd;
        std::mt19937 gen(_rd());
        token_offset = 0;
        
        for (uint32_t req = 0; req < nreq; req++) {
            auto seq_len = req_lens[req];
            
            // Generate random value for sampling [0, 1)
            float random_val = std::uniform_real_distribution<float>(0, 1)(gen);
            
            /*
             * Sample next token for this request
             * 
             * Input: prob_buf[req*dvoc : (req+1)*dvoc] - logits over vocabulary [dvoc]
             * Output: result_buf[req] - sampled token ID
             * Parameters:
             * - random_val: random seed for sampling
             * - topp[req]: nucleus sampling threshold (cumulative probability)
             * - topk[req]: top-k filtering (keep top k tokens)
             * - temperature[req]: scaling factor for logits (higher = more random)
             */
            // prob_buf->debug();
            RUN_INFINI(infiniopRandomSample(
                desc_sample, workspace, workspace_size,
                result_buf->data(req),              // Output: sampled token ID
                prob_buf->data(req * dvoc),         // Input: logits for this request [dvoc]
                random_val,                         // Random seed
                topp[req], topk[req], temperature[req],  // Sampling parameters
                stream));
            // result_buf->debug();
            token_offset += seq_len;
        }
        
        /*
         * Copy Results to Host Memory
         * 
         * Transfer sampled token IDs from device to host memory for return to caller.
         * Synchronize stream to ensure all computations are complete before copy.
         */
        RUN_INFINI(infinirtStreamSynchronize(stream));
        RUN_INFINI(infinirtMemcpy(result_cpu.data(), result_buf->data(),
                                  sizeof(int64_t) * nreq, INFINIRT_MEMCPY_D2H));
                                  
        // Store results in output array
        for (uint32_t req = 0; req < nreq; req++) {
            output[req] = result_cpu[req];
        }
    }

    /*
     * ==================================================================================
     * DESCRIPTOR CLEANUP AND RESOURCE DEALLOCATION
     * ==================================================================================
     * 
     * Properly release all InfiniCore descriptors to prevent memory leaks.
     * Descriptors must be destroyed in reverse order of dependencies.
     */
    // Clean up
    infiniopDestroyRMSNormDescriptor(desc_norm);              // Layer normalization
    if (has_qkv_bias) {
        infiniopDestroyRearrangeDescriptor(desc_qkv_bias);    // QKV bias rearrangement
    }
    infiniopDestroyGemmDescriptor(desc_attn_qkv);             // QKV projection
    infiniopDestroyGemmDescriptor(desc_attn_o);               // Attention output projection
    infiniopDestroyRoPEDescriptor(desc_rope_q);               // RoPE for queries
    infiniopDestroyRoPEDescriptor(desc_rope_k);               // RoPE for keys
    
    // Clean up per-request attention descriptors
    for (uint32_t req = 0; req < nreq; req++) {
        infiniopDestroyRearrangeDescriptor(desc_kv_rearranges[req]);    // KV cache storage
        infiniopDestroyRearrangeDescriptor(desc_q_rearranges[req]);     // Query rearrangement
        infiniopDestroyGemmDescriptor(desc_qk_gemms[req]);              // QK attention scores
        infiniopDestroyCausalSoftmaxDescriptor(desc_qk_softmaxs[req]);  // Causal softmax
        infiniopDestroyGemmDescriptor(desc_attn_v_gemms[req]);          // Attention-value multiplication
        infiniopDestroyRearrangeDescriptor(desc_attn_v_rearranges[req]); // Output rearrangement
    }
    
    // Clean up FFN descriptors
    infiniopDestroyGemmDescriptor(desc_ffn_gate_up);          // FFN gate & up projections
    infiniopDestroySwiGLUDescriptor(desc_swiglu);             // SwiGLU activation
    infiniopDestroyGemmDescriptor(desc_ffn_down);             // FFN down projection
    
    // Clean up output descriptors
    infiniopDestroyRMSNormDescriptor(desc_norm_out);          // Final layer normalization
    infiniopDestroyGemmDescriptor(desc_out_embd);             // Language model head
    infiniopDestroyRandomSampleDescriptor(desc_sample);       // Token sampling
}

/*
 * Batch Inference API Function (C Interface)
 * 
 * Thread-safe wrapper for distributed batch inference across multiple devices.
 * This function coordinates inference across all devices in the model using
 * a producer-consumer pattern with condition variables for synchronization.
 * 
 * Parameters:
 * - model: JiugeModel instance containing device resources and worker threads
 * - tokens: Input token IDs [ntok] - concatenated tokens from all requests
 * - ntok: Total number of tokens across all requests  
 * - req_lens: Length of each request [nreq]
 * - nreq: Number of requests in the batch
 * - req_pos: Starting position for each request in KV cache [nreq]
 * - kv_caches: KV cache storage for each request [nreq]
 * - temperature/topk/topp: Sampling parameters [nreq]
 * - output: Generated token IDs [nreq] - filled by this function
 * 
 * Thread Synchronization:
 * 1. Main thread signals all worker threads to start inference
 * 2. Worker threads process their assigned device slices in parallel
 * 3. Main thread waits for all workers to complete before returning
 */
__C void
inferBatch(struct JiugeModel *model,
           const uint32_t *tokens, uint32_t ntok,
           const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
           struct KVCache **kv_caches,
           const float *temperature, const uint32_t *topk, const float *topp,
           uint32_t *output) {
    /*
     * Copy inference parameters to model's request structure
     * This allows worker threads to access the request data safely.
     */
    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.kv_caches = kv_caches;
    model->req.output = output;
    model->req.temperature = temperature;
    model->req.topk = topk;
    model->req.topp = topp;

    /*
     * Signal all worker threads to start inference
     * 
     * Each device has a dedicated worker thread waiting on a condition variable.
     * Setting proceed=true and notifying wakes up the worker to process this batch.
     */
    for (size_t idev = 0; idev < model->dev_ids.size(); idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].proceed = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }
    
    /*
     * Wait for all worker threads to complete inference
     * 
     * Wait in reverse order to handle any potential dependencies.
     * Each worker will set proceed=false when done and notify cv_done.
     */
    for (size_t i = model->dev_ids.size(); i > 0; i--) {
        auto idev = i - 1;
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].cv_done.wait(lock, [&] { return !(model->states[idev].proceed); });
        lock.unlock();
    }
}

/*
 * Device Worker Thread Function
 * 
 * Each device runs this function in a dedicated thread for asynchronous inference.
 * The thread lifecycle:
 * 1. Initialize device resources and signal readiness
 * 2. Wait for inference requests on condition variable
 * 3. Execute device-specific inference when signaled
 * 4. Signal completion and wait for next request
 * 5. Clean up resources when exit flag is set
 * 
 * This design enables efficient pipeline parallelism and device utilization.
 * 
 * Parameters:
 * - meta: Model architecture metadata
 * - weights: Model weight tensors  
 * - rsrc: Device resource structure to populate
 * - state: Thread synchronization state
 * - req: Shared request data structure
 * - device: InfiniCore device type
 * - idev/ndev: Device index and total device count
 * - dev_id: Physical device ID
 * - comm: Inter-device communication context
 */
void launchDevice(const JiugeMeta &meta, const JiugeWeights *weights, DeviceResource *rsrc, InferState &state, InferRequest &req,
                  infiniDevice_t device, int idev, int ndev, int dev_id, infinicclComm_t comm) {
    /*
     * Device Resource Initialization
     * 
     * Create all device-specific resources needed for inference.
     * This includes weights, handles, streams, and memory pools.
     */
    // Create Device Resource
    createDeviceResource(rsrc, &meta, weights, device, idev, ndev, dev_id, comm);
    
    /*
     * Signal Device Readiness
     * 
     * Notify the main thread that this device is ready for inference.
     * The main thread waits for all devices to be loaded before proceeding.
     */
    {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.loaded = true;
        lock.unlock();
        state.cv_load.notify_one();
    }

    /*
     * Main Worker Thread Loop
     * 
     * Wait for inference requests and process them until exit is requested.
     * This implements a producer-consumer pattern where the main thread
     * produces inference requests and worker threads consume them.
     */
    // Infer Loop
    while (true) {
        /*
         * Wait for Inference Request or Exit Signal
         * 
         * Block until either:
         * - proceed=true: new inference request is available
         * - exit_flag=true: shutdown requested
         */
        std::unique_lock<std::mutex> lock(state.mtx);
        state.cv_start.wait(lock, [&] { return state.proceed || state.exit_flag; });
        
        // Exit gracefully if shutdown requested
        if (state.exit_flag) {
            break;
        }

        /*
         * Execute Device-Specific Inference
         * 
         * Process the current batch on this device using tensor parallelism.
         * The function handles this device's slice of the computation.
         */
        inferDeviceBatch(meta, *rsrc, idev, ndev, req.tokens, req.ntok, req.req_lens, req.nreq, req.req_pos, req.kv_caches, req.temperature, req.topk, req.topp, req.output);

        /*
         * Signal Completion
         * 
         * Mark this device as finished and notify the main thread.
         * The main thread waits for all devices before returning results.
         */
        state.proceed = false;
        lock.unlock();
        state.cv_done.notify_one();
    }

    /*
     * Resource Cleanup
     * 
     * Release all device resources when thread exits.
     * This ensures proper cleanup during model destruction.
     */
    // Clean-Up
    releaseDeviceResource(*rsrc);
}

/*
 * JiugeModel Constructor
 * 
 * Initializes a distributed inference model with multiple devices.
 * Sets up worker threads, communication contexts, and device resources.
 * 
 * Parameters:
 * - _meta: Model architecture metadata (layers, dimensions, data types)
 * - weights: Model weight tensors
 * - device_: InfiniCore device type (GPU/CPU)
 * - device_ids: List of physical device IDs to use for distributed inference
 * 
 * Distributed Setup:
 * - Creates one worker thread per device for parallel inference
 * - Initializes InfiniCCL communication for multi-device synchronization
 * - Waits for all devices to complete initialization before returning
 */
JiugeModel::JiugeModel(const JiugeMeta *_meta, const JiugeWeights *weights, infiniDevice_t device_, std::vector<int> device_ids) : meta(*_meta) {
    int ndev = int(device_ids.size());
    device = device_;
    dev_ids = device_ids;
    dev_resources = std::vector<DeviceResource>(ndev);
    states = std::vector<InferState>(ndev);
    threads.resize(ndev);
    
    /*
     * Initialize InfiniCore Runtime
     * 
     * Set up the InfiniCore runtime environment for device management
     * and operation execution.
     */
    RUN_INFINI(infinirtInit());
    
    /*
     * Initialize Multi-Device Communication
     * 
     * Create InfiniCCL communicators for distributed inference if using multiple devices.
     * Communication enables synchronization and data exchange between devices.
     */
    auto comms = std::vector<infinicclComm_t>(ndev, nullptr);
    if (ndev > 1) {
        RUN_INFINI(infinicclCommInitAll(device, comms.data(), ndev, dev_ids.data()));
    }

    /*
     * Launch Worker Threads
     * 
     * Create one worker thread per device to handle asynchronous inference.
     * Each thread initializes its device resources and waits for inference requests.
     */
    for (int i = 0; i < ndev; i++) {
        threads[i] = std::thread(launchDevice, std::cref(meta), weights, &dev_resources[i], std::ref(states[i]), std::ref(req), device, i, ndev, dev_ids[i], comms[i]);
    }
    
    /*
     * Wait for All Devices to Initialize
     * 
     * Block until all worker threads have completed device resource initialization.
     * This ensures the model is fully ready before constructor returns.
     */
    for (int i = 0; i < ndev; i++) {
        std::unique_lock<std::mutex> lock(states[i].mtx);
        states[i].cv_load.wait(lock, [&] { return states[i].loaded; });
        lock.unlock();
    }
}

/*
 * JiugeModel Creation Function (C Interface)
 * 
 * Creates a new JiugeModel instance for distributed inference.
 * This is the main entry point for creating models from C/Python code.
 * 
 * Parameters:
 * - meta: Model architecture metadata
 * - weights: Model weight tensors
 * - device: InfiniCore device type
 * - ndev: Number of devices for distributed inference
 * - dev_ids: Array of physical device IDs [ndev]
 * 
 * Returns: Pointer to newly created JiugeModel instance
 */
__C struct JiugeModel *
createJiugeModel(const JiugeMeta *meta,
                 const JiugeWeights *weights,
                 infiniDevice_t device,
                 int ndev,
                 const int *dev_ids) {
    // Convert C array to C++ vector for constructor
    std::vector<int> device_ids(ndev);
    std::copy(dev_ids, dev_ids + ndev, device_ids.begin());
    
    // Create and return new model instance
    JiugeModel *model = new JiugeModel(meta, weights, device, device_ids);
    return model;
}

/*
 * JiugeModel Destruction Function (C Interface)
 * 
 * Safely destroys a JiugeModel instance and cleans up all resources.
 * Ensures all worker threads are properly terminated before deallocation.
 * 
 * Shutdown Process:
 * 1. Signal all worker threads to exit via exit_flag
 * 2. Notify all threads waiting on condition variables
 * 3. Join all threads to ensure clean termination
 * 4. Deallocate model instance
 * 
 * This prevents resource leaks and ensures graceful shutdown.
 */
__C void destroyJiugeModel(struct JiugeModel *model) {
    auto ndev = model->dev_resources.size();

    /*
     * Signal All Worker Threads to Exit
     * 
     * Set exit_flag for each device and notify the worker threads.
     * This breaks them out of their inference loops.
     */
    for (size_t idev = 0; idev < ndev; idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].exit_flag = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }

    /*
     * Wait for All Threads to Terminate
     * 
     * Join each worker thread to ensure clean shutdown.
     * This guarantees all device resources are properly released.
     */
    for (size_t idev = 0; idev < ndev; idev++) {
        model->threads[idev].join();
    }

    // Deallocate model instance
    delete model;
}
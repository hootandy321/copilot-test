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

    // Prepare operators and workspace
    size_t workspace_size = 0, temp_size = 0;
    // attn & mlp rmsnorm
    infiniopRMSNormDescriptor_t desc_norm;
    RUN_INFINI(infiniopCreateRMSNormDescriptor(
        rsrc.handle, &desc_norm, logits_in->desc(),
        logits_out->desc(), rsrc.w_attn_norm[0]->desc(),
        meta.epsilon));
    RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_norm, &workspace_size));
    workspace_size = std::max(workspace_size, temp_size);
    // Attention
    infiniopGemmDescriptor_t desc_attn_qkv, desc_attn_o;
    infiniopRearrangeDescriptor_t desc_qkv_bias;
    if (has_qkv_bias) {
        RUN_INFINI(infiniopCreateRearrangeDescriptor(
            rsrc.handle, &desc_qkv_bias, qkv_buf->desc(),
            TensorDesc::create(dt_logits, {ntok, (nh + nkvh * 2) * dh}, {0, 1})->desc()));
    }
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_attn_qkv, qkv_buf->desc(),
        logits_in->desc(), rsrc.w_attn_qkv[0]->desc()));
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_attn_o, logits_in->desc(),
        o_buf->desc(), rsrc.w_attn_out[0]->desc()));
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_attn_qkv, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_attn_o, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    infiniopRoPEDescriptor_t desc_rope_q, desc_rope_k;
    qkv_buf->dimSplit(1, {nh + nkvh * 2, dh}); // (ntok, nh + 2 * nkvh, dh)
    auto qkv_buf_q = qkv_buf->slice(1, 0, nh);
    auto qkv_buf_k = qkv_buf->slice(1, nh, nkvh);
    RUN_INFINI(infiniopCreateRoPEDescriptor(
        rsrc.handle, &desc_rope_q, qkv_buf_q->desc(), qkv_buf_q->desc(),
        pos_ids_buf->desc(), rsrc.sin_table->desc(),
        rsrc.cos_table->desc()));
    RUN_INFINI(infiniopGetRoPEWorkspaceSize(desc_rope_q, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    RUN_INFINI(infiniopCreateRoPEDescriptor(
        rsrc.handle, &desc_rope_k, qkv_buf_k->desc(), qkv_buf_k->desc(),
        pos_ids_buf->desc(), rsrc.sin_table->desc(),
        rsrc.cos_table->desc()));
    RUN_INFINI(infiniopGetRoPEWorkspaceSize(desc_rope_k, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    // attention inner
    auto desc_kv_rearranges = std::vector<infiniopRearrangeDescriptor_t>(nreq);
    auto desc_q_rearranges = std::vector<infiniopRearrangeDescriptor_t>(nreq);
    auto desc_qk_gemms = std::vector<infiniopGemmDescriptor_t>(nreq);
    auto desc_qk_softmaxs = std::vector<infiniopCausalSoftmaxDescriptor_t>(nreq);
    auto desc_attn_v_gemms = std::vector<infiniopGemmDescriptor_t>(nreq);
    auto desc_attn_v_rearranges = std::vector<infiniopRearrangeDescriptor_t>(nreq);
    size_t token_offset = 0;
    size_t max_qk_size = 0;
    size_t max_seq_len = 0;
    o_buf->dimSplit(1, {nh, dh});
    for (uint32_t req = 0; req < nreq; req++) {
        auto past_len = req_pos[req];
        auto seq_len = req_lens[req];
        auto total_len = past_len + seq_len;
        auto o = o_buf->slice({{0, token_offset, seq_len}});
        auto q = qkv_buf->slice({{0, token_offset, seq_len}, {1, 0, nh}});
        auto k = qkv_buf->slice({{0, token_offset, seq_len}, {1, nh, nkvh}});
        // auto v = qkv_buf->slice({{0, token_offset, seq_len}, {1, nh + nkvh, nkvh}});
        // kv cache tensors can share the same descriptor
        // [nkvh, dh, total_len]
        auto full_kv = kv_caches[req]->k[idev][0]->slice(0, 0, total_len)->permute({1, 2, 0});
        auto cache_kv = kv_caches[req]->k[idev][0]->slice(0, past_len, seq_len);

        RUN_INFINI(infiniopCreateRearrangeDescriptor(rsrc.handle, &desc_kv_rearranges[req],
                                                     cache_kv->desc(), k->desc()));

        // [nkvh, ngroup, seq_len, dh]
        q->dimSplit(1, {nkvh, ngroup})->permute({1, 2, 0, 3});
        auto q_t = TensorDesc::create(dt_logits, {nkvh, ngroup, seq_len, dh});
        // [seq_len, nkvh, ngroup, dh] -> [nkvh, ngroup, seq_len, dh]
        RUN_INFINI(infiniopCreateRearrangeDescriptor(rsrc.handle, &desc_q_rearranges[req],
                                                     q_t->desc(), q->desc()));
        // [nkvh, ngroup, seq_len, dh] -> [seq_len, nkvh, ngroup, dh]
        auto attn_v_t = q_t;
        auto attn_v = TensorDesc::createWithOrder(dt_logits, {nkvh, ngroup, seq_len, dh}, {1, 2, 0, 3});
        RUN_INFINI(infiniopCreateRearrangeDescriptor(rsrc.handle, &desc_attn_v_rearranges[req],
                                                     attn_v->desc(), attn_v_t->desc()));
        q_t = TensorDesc::create(dt_logits, {nkvh, ngroup * seq_len, dh});
        auto qk = TensorDesc::create(dt_logits, {nkvh, ngroup * seq_len, total_len});
        max_qk_size = std::max(max_qk_size, size_t(seq_len * total_len));
        max_seq_len = std::max(max_seq_len, size_t(seq_len));
        RUN_INFINI(infiniopCreateGemmDescriptor(
            rsrc.handle, &desc_qk_gemms[req], qk->desc(), q_t->desc(), full_kv->desc()));
        RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_qk_gemms[req], &temp_size));
        workspace_size = std::max(workspace_size, temp_size);

        // [nkvh, total_len, dh]
        auto full_v = kv_caches[req]->v[idev][0]->slice(0, 0, total_len)->permute({1, 0, 2});
        RUN_INFINI(infiniopCreateGemmDescriptor(
            rsrc.handle, &desc_attn_v_gemms[req], q_t->desc(), qk->desc(), full_v->desc()));
        RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_attn_v_gemms[req], &temp_size));
        workspace_size = std::max(workspace_size, temp_size);

        qk = TensorDesc::create(dt_logits, {nkvh * ngroup, seq_len, total_len});
        RUN_INFINI(infiniopCreateCausalSoftmaxDescriptor(
            rsrc.handle, &desc_qk_softmaxs[req], qk->desc(), qk->desc()));
        RUN_INFINI(infiniopGetCausalSoftmaxWorkspaceSize(desc_qk_softmaxs[req], &temp_size));
        workspace_size = std::max(workspace_size, temp_size);

        token_offset += seq_len;
    }
    auto qk_buf = Tensor::buffer(dt_logits, {nh, max_qk_size}, rsrc.memory_pool);
    auto rearrange_q_buf = Tensor::buffer(dt_logits, {nkvh, ngroup * max_seq_len, dh}, rsrc.memory_pool);
    auto attn_val_buf = Tensor::buffer(dt_logits, {nh, max_seq_len, dh}, rsrc.memory_pool);

    // MLP descriptors
    infiniopGemmDescriptor_t desc_ffn_gate_up, desc_ffn_down;
    infiniopSwiGLUDescriptor_t desc_swiglu;
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_ffn_gate_up, gate_up_buf->desc(),
        logits_out->desc(), rsrc.w_ffn_gate_up[0]->desc()));
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_ffn_gate_up, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    auto gate_buf = gate_up_buf->slice(1, 0, di);
    auto up_buf = gate_up_buf->slice(1, di, di);
    RUN_INFINI(infiniopCreateSwiGLUDescriptor(
        rsrc.handle, &desc_swiglu, gate_buf->desc(), up_buf->desc(), gate_buf->desc()));
    RUN_INFINI(infiniopGetSwiGLUWorkspaceSize(desc_swiglu, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_ffn_down, logits_in->desc(),
        gate_buf->desc(), rsrc.w_ffn_down[0]->desc()));
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_ffn_down, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);

    // Output and sample
    infiniopRMSNormDescriptor_t desc_norm_out;
    RUN_INFINI(infiniopCreateRMSNormDescriptor(
        rsrc.handle, &desc_norm_out, logits_out->slice(0, 0, 1)->desc(),
        logits_out->slice(0, 0, 1)->desc(),
        rsrc.w_out_norm->desc(), meta.epsilon));
    RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_norm_out, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    infiniopGemmDescriptor_t desc_out_embd;
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_out_embd, prob_buf->desc(),
        logits_out->slice(0, 0, nreq)->desc(),
        rsrc.w_out_embd->desc()));
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_out_embd, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    infiniopRandomSampleDescriptor_t desc_sample;
    RUN_INFINI(infiniopCreateRandomSampleDescriptor(
        rsrc.handle, &desc_sample,
        TensorDesc::create(INFINI_DTYPE_I64, {}, {})->desc(),
        TensorDesc::create(dt_logits, {dvoc}, {1})->desc()));
    RUN_INFINI(infiniopGetRandomSampleWorkspaceSize(desc_sample, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    // Allocate workspace
    std::shared_ptr<Storage> workspace_storage = Storage::createFromPool(workspace_size, rsrc.memory_pool);
    void *workspace = workspace_storage->memory();

    // Compute
    for (uint32_t layer = 0; layer < nlayer; layer++) {
        // 1. Attention
        // rms norm
        RUN_INFINI(infiniopRMSNorm(
            desc_norm, workspace, workspace_size,
            logits_out->data(), logits_in->data(),
            rsrc.w_attn_norm[layer]->data(), stream));
        // qkv_proj
        if (has_qkv_bias) {
            RUN_INFINI(infiniopRearrange(
                desc_qkv_bias,
                qkv_buf->data(), rsrc.b_attn_qkv[layer]->data(), stream));
        }
        RUN_INFINI(infiniopGemm(
            desc_attn_qkv, workspace, workspace_size,
            qkv_buf->data(), logits_out->data(),
            rsrc.w_attn_qkv[layer]->data(), 1.0, has_qkv_bias ? 1.0 : 0.0, stream));
        // rope
        RUN_INFINI(infiniopRoPE(
            desc_rope_q, workspace, workspace_size,
            qkv_buf->data(), qkv_buf->data(),
            pos_ids_buf->data(),
            rsrc.sin_table->data(),
            rsrc.cos_table->data(), stream));
        RUN_INFINI(infiniopRoPE(
            desc_rope_k, workspace, workspace_size,
            qkv_buf->data(nh * dh), qkv_buf->data(nh * dh),
            pos_ids_buf->data(),
            rsrc.sin_table->data(),
            rsrc.cos_table->data(),
            stream));

        size_t token_offset = 0;
        for (uint32_t req = 0; req < nreq; req++) {
            auto past_len = req_pos[req];
            auto seq_len = req_lens[req];
            auto o = o_buf->slice({{0, token_offset, seq_len}});
            auto q = qkv_buf->slice({{0, token_offset, seq_len}, {1, 0, nh}});
            auto k = qkv_buf->slice({{0, token_offset, seq_len}, {1, nh, nkvh}});
            auto v = qkv_buf->slice({{0, token_offset, seq_len}, {1, nh + nkvh, nkvh}});
            // self attention
            // concat
            RUN_INFINI(infiniopRearrange(
                desc_kv_rearranges[req],
                kv_caches[req]->k[idev][layer]->data(past_len * nkvh * dh),
                k->data(), stream));
            RUN_INFINI(infiniopRearrange(
                desc_kv_rearranges[req],
                kv_caches[req]->v[idev][layer]->data(past_len * nkvh * dh),
                v->data(), stream));
            // qk
            RUN_INFINI(infiniopRearrange(desc_q_rearranges[req], rearrange_q_buf->data(), q->data(), stream));
            RUN_INFINI(infiniopGemm(
                desc_qk_gemms[req], workspace, workspace_size,
                qk_buf->data(), rearrange_q_buf->data(), kv_caches[req]->k[idev][layer]->data(), 1. / sqrt(dh), 0.0, stream));
            // softmax
            RUN_INFINI(infiniopCausalSoftmax(
                desc_qk_softmaxs[req], workspace, workspace_size,
                qk_buf->data(), qk_buf->data(), stream));
            // attn val
            RUN_INFINI(infiniopGemm(
                desc_attn_v_gemms[req], workspace, workspace_size,
                attn_val_buf->data(), qk_buf->data(), kv_caches[req]->v[idev][layer]->data(), 1.0, 0.0, stream));
            // rearrange attn val
            RUN_INFINI(infiniopRearrange(
                desc_attn_v_rearranges[req],
                o->data(),
                attn_val_buf->data(), stream));

            token_offset += seq_len;
        }
        // o_proj
        RUN_INFINI(infiniopGemm(
            desc_attn_o, workspace, workspace_size,
            logits_in->data(), o_buf->data(),
            rsrc.w_attn_out[layer]->data(), 1.0, idev == 0 ? 1.0 : 0.0, stream)); // only rank 0 adds residual

        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }
        // 2. FFN
        // rms_norm
        RUN_INFINI(infiniopRMSNorm(
            desc_norm, workspace, workspace_size,
            logits_out->data(), logits_in->data(),
            rsrc.w_ffn_norm[layer]->data(), stream));
        RUN_INFINI(infiniopGemm(
            desc_ffn_gate_up, workspace, workspace_size,
            gate_up_buf->data(), logits_out->data(), rsrc.w_ffn_gate_up[layer]->data(),
            1.0, 0.0, stream));
        RUN_INFINI(infiniopSwiGLU(
            desc_swiglu, workspace, workspace_size,
            gate_buf->data(), up_buf->data(), gate_buf->data(), stream));
        RUN_INFINI(infiniopGemm(
            desc_ffn_down, workspace, workspace_size,
            logits_in->data(), gate_buf->data(),
            rsrc.w_ffn_down[layer]->data(), 1.0, idev == 0 ? 1.0 : 0.0, stream)); // only rank 0 adds residual

        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }
    }
    // Sample and Output
    if (idev == 0) {
        size_t token_offset = 0;
        for (uint32_t req = 0; req < nreq; req++) {
            auto seq_len = req_lens[req];
            token_offset += seq_len;
            RUN_INFINI(infiniopRMSNorm(
                desc_norm_out, workspace, workspace_size,
                logits_out->data(req * d),
                logits_in->data((token_offset - 1) * d),
                rsrc.w_out_norm->data(), stream));
        }
        RUN_INFINI(infiniopGemm(
            desc_out_embd, workspace, workspace_size,
            prob_buf->data(), logits_out->data(),
            rsrc.w_out_embd->data(), 1.0, 0.0, stream));
        std::random_device _rd;
        std::mt19937 gen(_rd());
        token_offset = 0;
        for (uint32_t req = 0; req < nreq; req++) {
            auto seq_len = req_lens[req];
            float random_val = std::uniform_real_distribution<float>(0, 1)(gen);
            // prob_buf->debug();
            RUN_INFINI(infiniopRandomSample(
                desc_sample, workspace, workspace_size,
                result_buf->data(req),
                prob_buf->data(req * dvoc),
                random_val,
                topp[req], topk[req], temperature[req],
                stream));
            // result_buf->debug();
            token_offset += seq_len;
        }
        RUN_INFINI(infinirtStreamSynchronize(stream));
        RUN_INFINI(infinirtMemcpy(result_cpu.data(), result_buf->data(),
                                  sizeof(int64_t) * nreq, INFINIRT_MEMCPY_D2H));
        for (uint32_t req = 0; req < nreq; req++) {
            output[req] = result_cpu[req];
        }
    }

    // Clean up
    infiniopDestroyRMSNormDescriptor(desc_norm);
    if (has_qkv_bias) {
        infiniopDestroyRearrangeDescriptor(desc_qkv_bias);
    }
    infiniopDestroyGemmDescriptor(desc_attn_qkv);
    infiniopDestroyGemmDescriptor(desc_attn_o);
    infiniopDestroyRoPEDescriptor(desc_rope_q);
    infiniopDestroyRoPEDescriptor(desc_rope_k);
    for (uint32_t req = 0; req < nreq; req++) {
        infiniopDestroyRearrangeDescriptor(desc_kv_rearranges[req]);
        infiniopDestroyRearrangeDescriptor(desc_q_rearranges[req]);
        infiniopDestroyGemmDescriptor(desc_qk_gemms[req]);
        infiniopDestroyCausalSoftmaxDescriptor(desc_qk_softmaxs[req]);
        infiniopDestroyGemmDescriptor(desc_attn_v_gemms[req]);
        infiniopDestroyRearrangeDescriptor(desc_attn_v_rearranges[req]);
    }
    infiniopDestroyGemmDescriptor(desc_ffn_gate_up);
    infiniopDestroySwiGLUDescriptor(desc_swiglu);
    infiniopDestroyGemmDescriptor(desc_ffn_down);
    infiniopDestroyRMSNormDescriptor(desc_norm_out);
    infiniopDestroyGemmDescriptor(desc_out_embd);
    infiniopDestroyRandomSampleDescriptor(desc_sample);
}

__C void
inferBatch(struct JiugeModel *model,
           const uint32_t *tokens, uint32_t ntok,
           const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
           struct KVCache **kv_caches,
           const float *temperature, const uint32_t *topk, const float *topp,
           uint32_t *output) {
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

    for (size_t idev = 0; idev < model->dev_ids.size(); idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].proceed = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }
    for (size_t i = model->dev_ids.size(); i > 0; i--) {
        auto idev = i - 1;
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].cv_done.wait(lock, [&] { return !(model->states[idev].proceed); });
        lock.unlock();
    }
}

void launchDevice(const JiugeMeta &meta, const JiugeWeights *weights, DeviceResource *rsrc, InferState &state, InferRequest &req,
                  infiniDevice_t device, int idev, int ndev, int dev_id, infinicclComm_t comm) {
    // Create Device Resource
    createDeviceResource(rsrc, &meta, weights, device, idev, ndev, dev_id, comm);
    {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.loaded = true;
        lock.unlock();
        state.cv_load.notify_one();
    }

    // Infer Loop
    while (true) {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.cv_start.wait(lock, [&] { return state.proceed || state.exit_flag; });
        // quit if exit_flag is set
        if (state.exit_flag) {
            break;
        }

        inferDeviceBatch(meta, *rsrc, idev, ndev, req.tokens, req.ntok, req.req_lens, req.nreq, req.req_pos, req.kv_caches, req.temperature, req.topk, req.topp, req.output);

        state.proceed = false;
        lock.unlock();
        state.cv_done.notify_one();
    }

    // Clean-Up
    releaseDeviceResource(*rsrc);
}

JiugeModel::JiugeModel(const JiugeMeta *_meta, const JiugeWeights *weights, infiniDevice_t device_, std::vector<int> device_ids) : meta(*_meta) {
    int ndev = int(device_ids.size());
    device = device_;
    dev_ids = device_ids;
    dev_resources = std::vector<DeviceResource>(ndev);
    states = std::vector<InferState>(ndev);
    threads.resize(ndev);
    RUN_INFINI(infinirtInit());
    auto comms = std::vector<infinicclComm_t>(ndev, nullptr);
    if (ndev > 1) {
        RUN_INFINI(infinicclCommInitAll(device, comms.data(), ndev, dev_ids.data()));
    }

    for (int i = 0; i < ndev; i++) {
        threads[i] = std::thread(launchDevice, std::cref(meta), weights, &dev_resources[i], std::ref(states[i]), std::ref(req), device, i, ndev, dev_ids[i], comms[i]);
    }
    for (int i = 0; i < ndev; i++) {
        std::unique_lock<std::mutex> lock(states[i].mtx);
        states[i].cv_load.wait(lock, [&] { return states[i].loaded; });
        lock.unlock();
    }
}

__C struct JiugeModel *
createJiugeModel(const JiugeMeta *meta,
                 const JiugeWeights *weights,
                 infiniDevice_t device,
                 int ndev,
                 const int *dev_ids) {
    std::vector<int> device_ids(ndev);
    std::copy(dev_ids, dev_ids + ndev, device_ids.begin());
    JiugeModel *model = new JiugeModel(meta, weights, device, device_ids);
    return model;
}

__C void destroyJiugeModel(struct JiugeModel *model) {
    auto ndev = model->dev_resources.size();

    for (size_t idev = 0; idev < ndev; idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].exit_flag = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }

    for (size_t idev = 0; idev < ndev; idev++) {
        model->threads[idev].join();
    }

    delete model;
}
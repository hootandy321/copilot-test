#include "qwen3_impl.hpp"
#include "qwen3_weight.hpp"

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "infinicore_infer.h"

#include <random>
#include <thread>
#include <vector>

void createDeviceResource(DeviceResource *rsrc, const Qwen3Meta *meta,
                          const Qwen3Weights *weights,
                          infiniDevice_t device, int idev,
                          int ndev, int dev_id,
                          infinicclComm_t comm) {
    RUN_INFINI(infinirtSetDevice(device, dev_id));
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle);
    infinirtStream_t stream;
    infinirtStreamCreate(&stream);

    std::vector<std::shared_ptr<Tensor>> w_attn_norm, w_attn_qkv, b_attn_qkv, w_attn_out,
        w_ffn_norm, w_ffn_gate_up, w_ffn_down;
    for (size_t layer = 0; layer < meta->nlayer; layer++) {
        w_attn_norm.push_back(
            getAttnNorm(meta, weights, layer));
        w_attn_qkv.push_back(
            getAttnQKV(meta, weights, layer, idev, ndev));
        if (weights->attn_qkv_b != nullptr) {
            b_attn_qkv.push_back(
                getAttnQKVBias(meta, weights, layer, idev, ndev));
        }
        w_attn_out.push_back(
            getAttnO(meta, weights, layer, idev, ndev));
        w_ffn_norm.push_back(
            getFFNNorm(meta, weights, layer));
        w_ffn_gate_up.push_back(
            getFFNGateUp(meta, weights, layer, idev, ndev));
        w_ffn_down.push_back(
            getFFNDown(meta, weights, layer, idev, ndev));
    }

    auto memory_pool = std::make_shared<MemoryPool>(128 * 1024 * 1024);

    *rsrc = DeviceResource{
        device,
        dev_id,
        handle,
        getInEmbd(meta, weights),
        getOutNorm(meta, weights),
        getOutEmbd(meta, weights),
        getSinTable(meta),
        getCosTable(meta),
        w_attn_norm,
        w_attn_qkv,
        b_attn_qkv,
        w_attn_out,
        w_ffn_norm,
        w_ffn_gate_up,
        w_ffn_down,
        stream,
        comm,
        memory_pool,
    };
    RUN_INFINI(infinirtDeviceSynchronize());
}

void releaseDeviceResource(DeviceResource &res) {
    infinirtDeviceSynchronize();
    // Release individual Tensors
    res.w_in_embd.reset();
    res.w_out_norm.reset();
    res.w_out_embd.reset();
    res.sin_table.reset();
    res.cos_table.reset();
    for (auto &t : res.w_attn_norm) {
        t.reset();
    }
    res.w_attn_norm.clear();
    for (auto &t : res.w_attn_qkv) {
        t.reset();
    }
    res.w_attn_qkv.clear();
    for (auto &t : res.b_attn_qkv) {
        t.reset();
    }
    res.b_attn_qkv.clear();
    for (auto &t : res.w_attn_out) {
        t.reset();
    }
    res.w_attn_out.clear();
    for (auto &t : res.w_ffn_norm) {
        t.reset();
    }
    res.w_ffn_norm.clear();
    for (auto &t : res.w_ffn_gate_up) {
        t.reset();
    }
    res.w_ffn_gate_up.clear();
    for (auto &t : res.w_ffn_down) {
        t.reset();
    }
    res.w_ffn_down.clear();

    res.memory_pool.reset();
    infiniopDestroyHandle(res.handle);
    infinirtStreamDestroy(res.stream);
}

void inferDeviceBatch(const Qwen3Meta &meta, DeviceResource &rsrc,
                      const uint32_t *tokens, uint32_t ntok,
                      const uint32_t *req_lens, uint32_t nreq,
                      const uint32_t *req_pos,
                      struct KVCache **kv_caches,
                      const float *temperature, const uint32_t *topk,
                      const float *topp, uint32_t *output) {
    auto dt_logits = meta.dt_logits;
    auto nlayer = meta.nlayer;
    auto d = meta.d;
    auto nh = meta.nh;
    auto nkvh = meta.nkvh;
    auto dh = meta.dh;
    auto di = meta.di;
    auto dvoc = meta.dvoc;
    auto epsilon = meta.epsilon;
    auto ngroup = nh / nkvh;
    auto idev = rsrc.device_id;

    // Create input tensors  
    auto pos_ids_buf = Tensor::buffer(INFINI_DTYPE_U32, {1, ntok}, rsrc.memory_pool);
    auto hidden_states = Tensor::buffer(dt_logits, {1, ntok, d}, rsrc.memory_pool);
    auto logits_in = Tensor::buffer(dt_logits, {1, ntok, d}, rsrc.memory_pool);
    auto logits_out = Tensor::buffer(dt_logits, {1, ntok, d}, rsrc.memory_pool);

    // Create position IDs
    uint32_t pos_offset = 0;
    for (uint32_t req = 0; req < nreq; req++) {
        auto req_start_pos = req_pos[req];
        auto req_len = req_lens[req];
        for (uint32_t i = 0; i < req_len; i++) {
            static_cast<uint32_t *>(pos_ids_buf->data())[pos_offset + i] = req_start_pos + i;
        }
        pos_offset += req_len;
    }

    // Input embedding (using direct memory copy like jiuge)
    for (uint32_t i = 0; i < ntok; i++) {
        RUN_INFINI(infinirtMemcpyAsync(hidden_states->data(i * d),
                                       rsrc.w_in_embd->data(tokens[i] * d),
                                       dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, rsrc.stream));
    }

    // Initialize workspace size calculation
    size_t workspace_size = 0, temp_size;

    // Create operation descriptors and calculate workspace size
    // RMS norm for layers
    infiniopRMSNormDescriptor_t desc_norm_in, desc_norm_ffn, desc_norm_out;
    RUN_INFINI(infiniopCreateRMSNormDescriptor(
        rsrc.handle, &desc_norm_in, logits_out->desc(),
        hidden_states->desc(), rsrc.w_attn_norm[0]->desc(), epsilon));
    RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_norm_in, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);

    // Attention QKV projection
    auto qkv_buf = Tensor::buffer(dt_logits, {1, ntok, (nh + 2 * nkvh) * dh}, rsrc.memory_pool);
    infiniopGemmDescriptor_t desc_qkv;
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_qkv, qkv_buf->desc(),
        logits_out->desc(), rsrc.w_attn_qkv[0]->desc()));
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_qkv, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);

    // Prepare Q, K, V buffers for Qwen3-specific processing
    qkv_buf->dimSplit(2, {nh * dh, nkvh * dh, nkvh * dh});
    auto qkv_buf_q = qkv_buf->slice(2, 0, nh * dh);
    auto qkv_buf_k = qkv_buf->slice(2, nh * dh, nkvh * dh);
    auto qkv_buf_v = qkv_buf->slice(2, nh * dh + nkvh * dh, nkvh * dh);

    // Qwen3 specific: RMS norm descriptors for Q and K after projection
    qkv_buf_q->dimSplit(2, {nh, dh});
    qkv_buf_k->dimSplit(2, {nkvh, dh});
    
    auto q_normed = Tensor::buffer(dt_logits, qkv_buf_q->shape(), rsrc.memory_pool);
    auto k_normed = Tensor::buffer(dt_logits, qkv_buf_k->shape(), rsrc.memory_pool);
    
    // Note: Qwen3 uses per-head RMS norm - this is a simplified implementation
    infiniopRMSNormDescriptor_t desc_norm_q, desc_norm_k;
    // For now, we'll apply RMS norm across the head dimension - this may need refinement
    RUN_INFINI(infiniopCreateRMSNormDescriptor(
        rsrc.handle, &desc_norm_q, q_normed->desc(),
        qkv_buf_q->desc(), nullptr, epsilon)); // Using nullptr for weight - per-head norm
    RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_norm_q, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    
    RUN_INFINI(infiniopCreateRMSNormDescriptor(
        rsrc.handle, &desc_norm_k, k_normed->desc(),
        qkv_buf_k->desc(), nullptr, epsilon));
    RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_norm_k, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);

    // RoPE descriptors
    infiniopRoPEDescriptor_t desc_rope_q, desc_rope_k;
    RUN_INFINI(infiniopCreateRoPEDescriptor(
        rsrc.handle, &desc_rope_q, q_normed->desc(), q_normed->desc(),
        pos_ids_buf->desc(), rsrc.sin_table->desc(),
        rsrc.cos_table->desc()));
    RUN_INFINI(infiniopGetRoPEWorkspaceSize(desc_rope_q, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    
    RUN_INFINI(infiniopCreateRoPEDescriptor(
        rsrc.handle, &desc_rope_k, k_normed->desc(), k_normed->desc(),
        pos_ids_buf->desc(), rsrc.sin_table->desc(),
        rsrc.cos_table->desc()));
    RUN_INFINI(infiniopGetRoPEWorkspaceSize(desc_rope_k, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);

    // Attention output and other operations
    auto o_buf = Tensor::buffer(dt_logits, {1, ntok, nh * dh}, rsrc.memory_pool);
    infiniopGemmDescriptor_t desc_attn_out;
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_attn_out, logits_in->desc(),
        o_buf->desc(), rsrc.w_attn_out[0]->desc()));
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_attn_out, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);

    // Residual connections
    infiniopAddDescriptor_t desc_add1, desc_add2;
    RUN_INFINI(infiniopCreateAddDescriptor(
        rsrc.handle, &desc_add1, hidden_states->desc(),
        hidden_states->desc(), logits_in->desc()));
    RUN_INFINI(infiniopGetAddWorkspaceSize(desc_add1, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);

    // FFN descriptors
    auto gate_up_buf = Tensor::buffer(dt_logits, {1, ntok, 2 * di}, rsrc.memory_pool);
    infiniopGemmDescriptor_t desc_ffn_gate_up, desc_ffn_down;
    
    RUN_INFINI(infiniopCreateRMSNormDescriptor(
        rsrc.handle, &desc_norm_ffn, logits_out->desc(),
        hidden_states->desc(), rsrc.w_ffn_norm[0]->desc(), epsilon));
    RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_norm_ffn, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);

    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_ffn_gate_up, gate_up_buf->desc(),
        logits_out->desc(), rsrc.w_ffn_gate_up[0]->desc()));
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_ffn_gate_up, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);

    // SwiGLU activation
    auto gate_buf = gate_up_buf->slice(2, 0, di);
    auto up_buf = gate_up_buf->slice(2, di, di);
    infiniopSwiGLUDescriptor_t desc_swiglu;
    RUN_INFINI(infiniopCreateSwiGLUDescriptor(
        rsrc.handle, &desc_swiglu, gate_buf->desc(), up_buf->desc(), gate_buf->desc()));
    RUN_INFINI(infiniopGetSwiGLUWorkspaceSize(desc_swiglu, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);

    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_ffn_down, logits_in->desc(),
        gate_buf->desc(), rsrc.w_ffn_down[0]->desc()));
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_ffn_down, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);

    RUN_INFINI(infiniopCreateAddDescriptor(
        rsrc.handle, &desc_add2, hidden_states->desc(),
        hidden_states->desc(), logits_in->desc()));
    RUN_INFINI(infiniopGetAddWorkspaceSize(desc_add2, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);

    // Final layer norm and output projection
    auto logits_buf = Tensor::buffer(dt_logits, {1, ntok, dvoc}, rsrc.memory_pool);
    infiniopGemmDescriptor_t desc_lm_head;
    RUN_INFINI(infiniopCreateRMSNormDescriptor(
        rsrc.handle, &desc_norm_out, logits_out->desc(),
        hidden_states->desc(), rsrc.w_out_norm->desc(), epsilon));
    RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_norm_out, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);

    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_lm_head, logits_buf->desc(),
        logits_out->desc(), rsrc.w_out_embd->desc()));
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_lm_head, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);

    // Allocate workspace
    auto workspace_storage = Tensor::buffer(INFINI_DTYPE_U8, {workspace_size}, rsrc.memory_pool);
    void *workspace = workspace_storage->data();

    // Execute the computation
    for (size_t layer = 0; layer < nlayer; layer++) {
        // 1. Input layer norm  
        RUN_INFINI(infiniopRMSNorm(
            desc_norm_in, workspace, workspace_size,
            logits_out->data(), hidden_states->data(),
            rsrc.w_attn_norm[layer]->data(), rsrc.stream));

        // 2. QKV projection
        RUN_INFINI(infiniopGemm(
            desc_qkv, workspace, workspace_size,
            qkv_buf->data(), logits_out->data(),
            rsrc.w_attn_qkv[layer]->data(), 1.0, 0.0, rsrc.stream));

        // 3. Qwen3 specific: Apply RMS norm to Q and K after projection
        RUN_INFINI(infiniopRMSNorm(
            desc_norm_q, workspace, workspace_size,
            q_normed->data(), qkv_buf_q->data(),
            nullptr, rsrc.stream)); // Per-head norm - no weight parameter
            
        RUN_INFINI(infiniopRMSNorm(
            desc_norm_k, workspace, workspace_size,
            k_normed->data(), qkv_buf_k->data(),
            nullptr, rsrc.stream));

        // 4. Apply RoPE to normalized Q and K
        RUN_INFINI(infiniopRoPE(
            desc_rope_q, workspace, workspace_size,
            q_normed->data(), q_normed->data(),
            pos_ids_buf->data(),
            rsrc.sin_table->data(),
            rsrc.cos_table->data(), rsrc.stream));
            
        RUN_INFINI(infiniopRoPE(
            desc_rope_k, workspace, workspace_size,
            k_normed->data(), k_normed->data(),
            pos_ids_buf->data(),
            rsrc.sin_table->data(),
            rsrc.cos_table->data(), rsrc.stream));

        // 5. Attention computation (simplified - would need full implementation)
        // TODO: Implement proper scaled dot-product attention with KV cache
        // For now, this is a placeholder that needs to be completed with:
        // - KV cache update
        // - Q @ K^T computation
        // - Causal masking and softmax  
        // - Attention @ V computation

        // 6. Attention output projection
        RUN_INFINI(infiniopGemm(
            desc_attn_out, workspace, workspace_size,
            logits_in->data(), o_buf->data(),
            rsrc.w_attn_out[layer]->data(), 1.0, 0.0, rsrc.stream));

        // 7. First residual connection
        RUN_INFINI(infiniopAdd(
            desc_add1, workspace, workspace_size,
            hidden_states->data(), hidden_states->data(), logits_in->data(), rsrc.stream));

        // 8. Post-attention layer norm (FFN norm)
        RUN_INFINI(infiniopRMSNorm(
            desc_norm_ffn, workspace, workspace_size,
            logits_out->data(), hidden_states->data(),
            rsrc.w_ffn_norm[layer]->data(), rsrc.stream));

        // 9. FFN Gate + Up projection
        RUN_INFINI(infiniopGemm(
            desc_ffn_gate_up, workspace, workspace_size,
            gate_up_buf->data(), logits_out->data(),
            rsrc.w_ffn_gate_up[layer]->data(), 1.0, 0.0, rsrc.stream));

        // 10. SwiGLU activation (SiLU for Qwen3)
        RUN_INFINI(infiniopSwiGLU(
            desc_swiglu, workspace, workspace_size,
            gate_buf->data(), gate_buf->data(), up_buf->data(), rsrc.stream));

        // 11. FFN Down projection
        RUN_INFINI(infiniopGemm(
            desc_ffn_down, workspace, workspace_size,
            logits_in->data(), gate_buf->data(),
            rsrc.w_ffn_down[layer]->data(), 1.0, 0.0, rsrc.stream));

        // 12. Second residual connection
        RUN_INFINI(infiniopAdd(
            desc_add2, workspace, workspace_size,
            hidden_states->data(), hidden_states->data(), logits_in->data(), rsrc.stream));
    }

    // Final layer norm
    RUN_INFINI(infiniopRMSNorm(
        desc_norm_out, workspace, workspace_size,
        logits_out->data(), hidden_states->data(),
        rsrc.w_out_norm->data(), rsrc.stream));

    // Output projection (language model head)
    RUN_INFINI(infiniopGemm(
        desc_lm_head, workspace, workspace_size,
        logits_buf->data(), logits_out->data(),
        rsrc.w_out_embd->data(), 1.0, 0.0, rsrc.stream));

    // TODO: Implement sampling logic for next token generation
    // This would include:
    // - Apply temperature scaling
    // - Top-k and top-p filtering  
    // - Multinomial sampling or greedy selection
    // - Return output tokens
}

Qwen3Model::Qwen3Model(const Qwen3Meta *meta, const Qwen3Weights *weights,
                       infiniDevice_t device, std::vector<int> device_ids)
    : meta(*meta), device(device), dev_ids(device_ids) {
    auto ndev = device_ids.size();
    dev_resources.resize(ndev);
    states.reserve(ndev);
    for (size_t i = 0; i < ndev; i++) {
        states.push_back(std::make_unique<InferState>());
    }
    threads.resize(ndev);

    auto comms = std::vector<infinicclComm_t>(ndev, nullptr);
    if (ndev > 1) {
        RUN_INFINI(infinicclCommInitAll(device, comms.data(), ndev, device_ids.data()));
    }

    for (size_t i = 0; i < ndev; i++) {
        createDeviceResource(&dev_resources[i], meta, weights, device, i, ndev,
                           device_ids[i], comms[i]);
    }

    // Start worker threads
    for (size_t i = 0; i < ndev; i++) {
        threads[i] = std::thread([this, i, &meta = this->meta]() {
            auto &state = *states[i];
            auto &rsrc = dev_resources[i];
            
            while (true) {
                std::unique_lock<std::mutex> lock(state.mtx);
                state.cv_load.wait(lock, [&state] { return state.loaded || state.exit_flag; });
                
                if (state.exit_flag) break;
                
                state.cv_start.wait(lock, [&state] { return state.proceed || state.exit_flag; });
                
                if (state.exit_flag) break;
                
                infinirtSetDevice(rsrc.device, rsrc.device_id);
                inferDeviceBatch(meta, rsrc, req.tokens, req.ntok, req.req_lens,
                               req.nreq, req.req_pos, req.kv_caches,
                               req.temperature, req.topk, req.topp, req.output);
                
                state.proceed = false;
                state.cv_done.notify_one();
            }
        });
    }
}

extern "C" {

struct Qwen3Model *createQwen3Model(const Qwen3Meta *meta,
                                     const Qwen3Weights *weights,
                                     infiniDevice_t device, int ndev,
                                     const int *dev_ids) {
    std::vector<int> device_ids(dev_ids, dev_ids + ndev);
    return new Qwen3Model(meta, weights, device, device_ids);
}

void destroyQwen3Model(struct Qwen3Model *model) {
    if (model) {
        // Signal threads to exit
        for (auto &state_ptr : model->states) {
            std::lock_guard<std::mutex> lock(state_ptr->mtx);
            state_ptr->exit_flag = true;
            state_ptr->cv_load.notify_one();
            state_ptr->cv_start.notify_one();
        }

        // Wait for threads to complete
        for (auto &thread : model->threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }

        // Release resources
        for (auto &rsrc : model->dev_resources) {
            releaseDeviceResource(rsrc);
        }

        delete model;
    }
}

void inferQwen3Batch(struct Qwen3Model *model,
                     const uint32_t *tokens, uint32_t ntok,
                     const uint32_t *req_lens, uint32_t nreq,
                     const uint32_t *req_pos, struct KVCache **kv_caches,
                     const float *temperature, const uint32_t *topk,
                     const float *topp, uint32_t *output) {
    if (!model) return;

    model->req = {tokens, ntok, req_lens, nreq, req_pos, kv_caches,
                  temperature, topk, topp, output};

    // Signal all devices
    for (auto &state_ptr : model->states) {
        std::lock_guard<std::mutex> lock(state_ptr->mtx);
        state_ptr->loaded = true;
        state_ptr->proceed = true;
        state_ptr->cv_load.notify_one();
        state_ptr->cv_start.notify_one();
    }

    // Wait for completion
    for (auto &state_ptr : model->states) {
        std::unique_lock<std::mutex> lock(state_ptr->mtx);
        state_ptr->cv_done.wait(lock, [&state_ptr] { return !state_ptr->proceed; });
        state_ptr->loaded = false;
    }
}

} // extern "C"
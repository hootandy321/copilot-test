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

    // Create RoPE tables for Qwen3
    auto rope_tables = getRoPETable(meta, device);

    auto memory_pool = std::make_shared<MemoryPool>(128 * 1024 * 1024);

    *rsrc = DeviceResource{
        device,
        dev_id,
        handle,
        getInEmbd(meta, weights),
        getOutNorm(meta, weights),
        getOutEmbd(meta, weights),
        rope_tables.first,   // sin_table
        rope_tables.second,  // cos_table
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
    auto tokens_buf = Tensor::buffer(INFINI_U32, {1, ntok}, rsrc.memory_pool);
    tokens_buf->setData(const_cast<uint32_t *>(tokens));

    auto pos_ids_buf = Tensor::buffer(INFINI_U32, {1, ntok}, rsrc.memory_pool);
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

    // Input embedding
    infiniopEmbeddingDescriptor_t desc_embd;
    RUN_INFINI(infiniopCreateEmbeddingDescriptor(
        rsrc.handle, &desc_embd, hidden_states->desc(),
        tokens_buf->desc(), rsrc.w_in_embd->desc()));
    size_t workspace_size = 0, temp_size;
    RUN_INFINI(infiniopGetEmbeddingWorkspaceSize(desc_embd, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);

    // Process each layer
    for (size_t layer = 0; layer < nlayer; layer++) {
        // Input layer norm
        infiniopRMSNormDescriptor_t desc_norm_in;
        RUN_INFINI(infiniopCreateRMSNormDescriptor(
            rsrc.handle, &desc_norm_in, logits_out->desc(),
            hidden_states->desc(), rsrc.w_attn_norm[layer]->desc(), epsilon));
        RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_norm_in, &temp_size));
        workspace_size = std::max(workspace_size, temp_size);

        // Attention QKV projection
        auto qkv_buf = Tensor::buffer(dt_logits, {1, ntok, (nh + 2 * nkvh) * dh}, rsrc.memory_pool);
        infiniopGemmDescriptor_t desc_qkv;
        RUN_INFINI(infiniopCreateGemmDescriptor(
            rsrc.handle, &desc_qkv, qkv_buf->desc(),
            logits_out->desc(), rsrc.w_attn_qkv[layer]->desc()));
        RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_qkv, &temp_size));
        workspace_size = std::max(workspace_size, temp_size);

        // Split QKV and apply RMS norm to Q and K (Qwen3 specific)
        qkv_buf->dimSplit(2, {nh * dh, nkvh * dh, nkvh * dh});
        auto qkv_buf_q = qkv_buf->slice(2, 0, nh * dh);
        auto qkv_buf_k = qkv_buf->slice(2, nh * dh, nkvh * dh);
        auto qkv_buf_v = qkv_buf->slice(2, nh * dh + nkvh * dh, nkvh * dh);

        // Qwen3 specific: Apply RMS norm to Q and K after projection
        qkv_buf_q->dimSplit(2, {nh, dh});
        qkv_buf_k->dimSplit(2, {nkvh, dh});
        
        // RMS norm for Q
        infiniopRMSNormDescriptor_t desc_norm_q;
        auto q_normed = Tensor::buffer(dt_logits, qkv_buf_q->shape(), rsrc.memory_pool);
        // Note: Qwen3 uses per-head RMS norm, would need to implement this properly
        
        // RMS norm for K  
        infiniopRMSNormDescriptor_t desc_norm_k;
        auto k_normed = Tensor::buffer(dt_logits, qkv_buf_k->shape(), rsrc.memory_pool);

        // Apply RoPE to Q and K
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

        // Attention computation (similar to jiuge but with normed Q/K)
        auto o_buf = Tensor::buffer(dt_logits, {1, ntok, nh * dh}, rsrc.memory_pool);

        // [Implementation details for attention would continue here...]
        // This is a simplified version focusing on the key Qwen3 differences

        // Output projection
        infiniopGemmDescriptor_t desc_attn_out;
        RUN_INFINI(infiniopCreateGemmDescriptor(
            rsrc.handle, &desc_attn_out, logits_in->desc(),
            o_buf->desc(), rsrc.w_attn_out[layer]->desc()));
        RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_attn_out, &temp_size));
        workspace_size = std::max(workspace_size, temp_size);

        // Residual connection
        infiniopAddDescriptor_t desc_add1;
        RUN_INFINI(infiniopCreateAddDescriptor(
            rsrc.handle, &desc_add1, hidden_states->desc(),
            hidden_states->desc(), logits_in->desc()));

        // Post-attention layer norm
        infiniopRMSNormDescriptor_t desc_norm_ffn;
        RUN_INFINI(infiniopCreateRMSNormDescriptor(
            rsrc.handle, &desc_norm_ffn, logits_out->desc(),
            hidden_states->desc(), rsrc.w_ffn_norm[layer]->desc(), epsilon));
        RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_norm_ffn, &temp_size));
        workspace_size = std::max(workspace_size, temp_size);

        // MLP: Gate + Up projection
        auto gate_up_buf = Tensor::buffer(dt_logits, {1, ntok, 2 * di}, rsrc.memory_pool);
        infiniopGemmDescriptor_t desc_ffn_gate_up;
        RUN_INFINI(infiniopCreateGemmDescriptor(
            rsrc.handle, &desc_ffn_gate_up, gate_up_buf->desc(),
            logits_out->desc(), rsrc.w_ffn_gate_up[layer]->desc()));
        RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_ffn_gate_up, &temp_size));
        workspace_size = std::max(workspace_size, temp_size);

        // SwiGLU activation (SiLU for Qwen3)
        auto gate_buf = gate_up_buf->slice(2, 0, di);
        auto up_buf = gate_up_buf->slice(2, di, di);
        infiniopSwiGLUDescriptor_t desc_swiglu;
        RUN_INFINI(infiniopCreateSwiGLUDescriptor(
            rsrc.handle, &desc_swiglu, gate_buf->desc(), up_buf->desc(), gate_buf->desc()));
        RUN_INFINI(infiniopGetSwiGLUWorkspaceSize(desc_swiglu, &temp_size));
        workspace_size = std::max(workspace_size, temp_size);

        // Down projection
        infiniopGemmDescriptor_t desc_ffn_down;
        RUN_INFINI(infiniopCreateGemmDescriptor(
            rsrc.handle, &desc_ffn_down, logits_in->desc(),
            gate_buf->desc(), rsrc.w_ffn_down[layer]->desc()));
        RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_ffn_down, &temp_size));
        workspace_size = std::max(workspace_size, temp_size);

        // Residual connection
        infiniopAddDescriptor_t desc_add2;
        RUN_INFINI(infiniopCreateAddDescriptor(
            rsrc.handle, &desc_add2, hidden_states->desc(),
            hidden_states->desc(), logits_in->desc()));
    }

    // Final layer norm and output projection
    infiniopRMSNormDescriptor_t desc_norm_out;
    RUN_INFINI(infiniopCreateRMSNormDescriptor(
        rsrc.handle, &desc_norm_out, logits_out->desc(),
        hidden_states->desc(), rsrc.w_out_norm->desc(), epsilon));
    RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_norm_out, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);

    auto logits_buf = Tensor::buffer(dt_logits, {1, ntok, dvoc}, rsrc.memory_pool);
    infiniopGemmDescriptor_t desc_lm_head;
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_lm_head, logits_buf->desc(),
        logits_out->desc(), rsrc.w_out_embd->desc()));
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_lm_head, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);

    // Sampling
    auto workspace = Tensor::buffer(INFINI_U8, {workspace_size}, rsrc.memory_pool);
    // [Implementation of sampling logic would continue here...]
}

Qwen3Model::Qwen3Model(const Qwen3Meta *meta, const Qwen3Weights *weights,
                       infiniDevice_t device, std::vector<int> device_ids)
    : meta(*meta), device(device), dev_ids(device_ids) {
    auto ndev = device_ids.size();
    dev_resources.resize(ndev);
    states.resize(ndev);
    threads.resize(ndev);

    infinicclComm_t comm = nullptr;
    if (ndev > 1) {
        infinicclCommInitRank(&comm, ndev, 0, device_ids.data());
    }

    for (size_t i = 0; i < ndev; i++) {
        createDeviceResource(&dev_resources[i], meta, weights, device, i, ndev,
                           device_ids[i], comm);
    }

    // Start worker threads
    for (size_t i = 0; i < ndev; i++) {
        threads[i] = std::thread([this, i]() {
            auto &state = states[i];
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
        for (auto &state : model->states) {
            std::lock_guard<std::mutex> lock(state.mtx);
            state.exit_flag = true;
            state.cv_load.notify_one();
            state.cv_start.notify_one();
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

void inferBatch(struct Qwen3Model *model,
                const uint32_t *tokens, uint32_t ntok,
                const uint32_t *req_lens, uint32_t nreq,
                const uint32_t *req_pos, struct KVCache **kv_caches,
                const float *temperature, const uint32_t *topk,
                const float *topp, uint32_t *output) {
    if (!model) return;

    model->req = {tokens, ntok, req_lens, nreq, req_pos, kv_caches,
                  temperature, topk, topp, output};

    // Signal all devices
    for (auto &state : model->states) {
        std::lock_guard<std::mutex> lock(state.mtx);
        state.loaded = true;
        state.proceed = true;
        state.cv_load.notify_one();
        state.cv_start.notify_one();
    }

    // Wait for completion
    for (auto &state : model->states) {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.cv_done.wait(lock, [&state] { return !state.proceed; });
        state.loaded = false;
    }
}

} // extern "C"
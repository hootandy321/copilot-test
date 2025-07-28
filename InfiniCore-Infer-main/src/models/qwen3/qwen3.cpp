#include "qwen3_impl.hpp"
#include "qwen3_weight.hpp"

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "infinicore_infer.h"

#include <random>
#include <thread>
#include <vector>

void createQwen3DeviceResource(Qwen3DeviceResource *rsrc, const Qwen3Meta *meta,
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
        w_ffn_norm, w_ffn_gate_up, w_ffn_down, w_q_norm, w_k_norm;
    
    for (size_t layer = 0; layer < meta->nlayer; layer++) {
        w_attn_norm.push_back(
            getQwen3AttnNorm(meta, weights, layer));
        w_attn_qkv.push_back(
            getQwen3AttnQKV(meta, weights, layer, idev, ndev));
        if (weights->attn_qkv_b != nullptr) {
            b_attn_qkv.push_back(
                getQwen3AttnQKVBias(meta, weights, layer, idev, ndev));
        }
        w_attn_out.push_back(
            getQwen3AttnO(meta, weights, layer, idev, ndev));
        w_ffn_norm.push_back(
            getQwen3FFNNorm(meta, weights, layer));
        w_ffn_gate_up.push_back(
            getQwen3FFNGateUp(meta, weights, layer, idev, ndev));
        w_ffn_down.push_back(
            getQwen3FFNDown(meta, weights, layer, idev, ndev));
        
        // Qwen3-specific: Q/K normalization weights
        if (weights->q_norm != nullptr) {
            w_q_norm.push_back(
                getQwen3QNorm(meta, weights, layer));
        }
        if (weights->k_norm != nullptr) {
            w_k_norm.push_back(
                getQwen3KNorm(meta, weights, layer));
        }
    }

    auto memory_pool = std::make_shared<MemoryPool>(128 * 1024 * 1024);

    // Qwen3-specific: Create Q/K normalization descriptors
    std::vector<infiniopRMSNormDescriptor_t> desc_q_norm, desc_k_norm;
    size_t qk_norm_workspace_size = 0;
    bool has_qk_norm = w_q_norm.size() > 0 && w_k_norm.size() > 0;
    
    if (has_qk_norm) {
        desc_q_norm.resize(meta->nlayer);
        desc_k_norm.resize(meta->nlayer);
        
        auto nh = meta->nh / ndev;
        auto nkvh = meta->nkvh / ndev;
        auto dh = meta->dh;
        auto dt_logits = meta->dt_logits;
        
        for (size_t layer = 0; layer < meta->nlayer; layer++) {
            size_t temp_size = 0;
            
            // Q normalization: create descriptor for per-head normalization
            // Input/output shape: (ntok * nh, dh) - will be reshaped at runtime
            // Weight shape: (dh) - per head dimension normalization
            auto q_reshaped = TensorDesc::create(dt_logits, {1, dh}); // Placeholder for (ntok * nh, dh)
            RUN_INFINI(infiniopCreateRMSNormDescriptor(
                handle, &desc_q_norm[layer], q_reshaped->desc(), q_reshaped->desc(),
                w_q_norm[layer]->desc(), meta->epsilon));
            RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_q_norm[layer], &temp_size));
            qk_norm_workspace_size = std::max(qk_norm_workspace_size, temp_size);
            
            // K normalization: create descriptor for per-head normalization  
            // Input/output shape: (ntok * nkvh, dh) - will be reshaped at runtime
            // Weight shape: (dh) - per head dimension normalization
            auto k_reshaped = TensorDesc::create(dt_logits, {1, dh}); // Placeholder for (ntok * nkvh, dh)
            RUN_INFINI(infiniopCreateRMSNormDescriptor(
                handle, &desc_k_norm[layer], k_reshaped->desc(), k_reshaped->desc(),
                w_k_norm[layer]->desc(), meta->epsilon));
            RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_k_norm[layer], &temp_size));
            qk_norm_workspace_size = std::max(qk_norm_workspace_size, temp_size);
        }
    }

    *rsrc = Qwen3DeviceResource{
        device,
        dev_id,
        handle,
        getQwen3InEmbd(meta, weights),
        getQwen3OutNorm(meta, weights),
        getQwen3OutEmbd(meta, weights),
        getQwen3SinTable(meta),
        getQwen3CosTable(meta),
        w_attn_norm,
        w_attn_qkv,
        b_attn_qkv,
        w_attn_out,
        w_ffn_norm,
        w_ffn_gate_up,
        w_ffn_down,
        w_q_norm,
        w_k_norm,
        desc_q_norm,
        desc_k_norm,
        qk_norm_workspace_size,
        stream,
        comm,
        memory_pool,
    };
    RUN_INFINI(infinirtDeviceSynchronize());
}

void releaseQwen3DeviceResource(Qwen3DeviceResource &res) {
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
    
    // Qwen3-specific cleanup
    for (auto &t : res.w_q_norm) {
        t.reset();
    }
    res.w_q_norm.clear();
    for (auto &t : res.w_k_norm) {
        t.reset();
    }
    res.w_k_norm.clear();
    
    // Cleanup Q/K normalization descriptors
    for (auto &desc : res.desc_q_norm) {
        infiniopDestroyRMSNormDescriptor(desc);
    }
    res.desc_q_norm.clear();
    for (auto &desc : res.desc_k_norm) {
        infiniopDestroyRMSNormDescriptor(desc);
    }
    res.desc_k_norm.clear();
    
    infiniopDestroyHandle(res.handle);
    res.handle = nullptr;
    infinirtStreamDestroy(res.stream);
    res.stream = nullptr;
    infinicclCommDestroy(res.comm);
    res.comm = nullptr;
}

void inferQwen3DeviceBatch(const Qwen3Meta &meta, Qwen3DeviceResource &rsrc,
                           uint32_t idev, uint32_t ndev,
                           const uint32_t *tokens, uint32_t ntok,
                           const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                           struct KVCache **kv_caches,
                           const float *temperature, const uint32_t *topk, const float *topp,
                           uint32_t *output) {
    auto nlayer = meta.nlayer;
    auto nkvh = meta.nkvh / ndev;
    auto nh = meta.nh / ndev;
    auto dh = meta.dh;
    auto d = meta.d;
    auto dt_logits = meta.dt_logits;
    auto di = meta.di / ndev;
    auto dvoc = meta.dvoc;
    auto stream = rsrc.stream;
    bool has_qkv_bias = rsrc.b_attn_qkv.size() > 0;
    bool has_qk_norm = rsrc.w_q_norm.size() > 0 && rsrc.w_k_norm.size() > 0;

    // Allocate buffers
    auto logits_in = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto logits_out = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto qkv_buf = Tensor::buffer(dt_logits, {ntok, (nh + nkvh * 2) * dh}, rsrc.memory_pool);
    auto gate_up_buf = Tensor::buffer(dt_logits, {ntok, 2 * di}, rsrc.memory_pool);
    auto o_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, rsrc.memory_pool);
    auto prob_buf = Tensor::buffer(dt_logits, {nreq, dvoc}, rsrc.memory_pool);
    auto result_buf = Tensor::buffer(INFINI_DTYPE_I64, {nreq}, rsrc.memory_pool);
    auto result_cpu = std::vector<int64_t>(nreq);

    // Qwen3-specific: Q/K normalization buffers
    std::shared_ptr<Tensor> q_norm_buf, k_norm_buf;
    if (has_qk_norm) {
        q_norm_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, rsrc.memory_pool);
        k_norm_buf = Tensor::buffer(dt_logits, {ntok, nkvh * dh}, rsrc.memory_pool);
    }

    // Prepare inputs
    auto batch_pos_ids = std::vector<uint32_t>(ntok);
    size_t req_start = 0;
    for (uint32_t req = 0; req < nreq; req++) {
        for (uint32_t i = 0; i < req_lens[req]; i++) {
            batch_pos_ids[req_start + i] = req_pos[req] + i;
        }
        req_start += req_lens[req];
    }

    std::shared_ptr<Tensor> pos_ids_buf;
    if (rsrc.device == INFINI_DEVICE_CPU) {
        pos_ids_buf = Tensor::weight(batch_pos_ids.data(), INFINI_DTYPE_U32, {ntok});
    } else {
        pos_ids_buf = Tensor::buffer(INFINI_DTYPE_U32, {ntok}, rsrc.memory_pool);
        RUN_INFINI(infinirtMemcpyAsync(pos_ids_buf->data(), batch_pos_ids.data(), sizeof(uint32_t) * ntok,
                                       INFINIRT_MEMCPY_H2D, stream));
    }
    
    for (uint32_t i = 0; i < ntok; i++) {
        RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d),
                                       rsrc.w_in_embd->data(tokens[i] * d),
                                       dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
    }

    // Prepare operators and workspace
    size_t workspace_size = 0, temp_size = 0;
    
    // RMSNorm descriptors
    infiniopRMSNormDescriptor_t desc_norm;
    RUN_INFINI(infiniopCreateRMSNormDescriptor(
        rsrc.handle, &desc_norm, logits_in->desc(),
        logits_out->desc(), rsrc.w_attn_norm[0]->desc(),
        meta.epsilon));
    RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_norm, &workspace_size));
    workspace_size = std::max(workspace_size, temp_size);

    // Qwen3-specific: Q/K normalization workspace size
    if (has_qk_norm) {
        workspace_size = std::max(workspace_size, rsrc.qk_norm_workspace_size);
    }

    // Attention operators
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
    
    // RoPE descriptors (following jiuge pattern)
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
    
    // Attention inner descriptors (following jiuge pattern)
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
        // kv cache tensors can share the same descriptor
        // [nkvh, dh, total_len]
        auto full_kv = kv_caches[req]->k[idev][0]->slice(0, 0, total_len)->permute({1, 2, 0});
        auto cache_kv = kv_caches[req]->k[idev][0]->slice(0, past_len, seq_len);

        RUN_INFINI(infiniopCreateRearrangeDescriptor(rsrc.handle, &desc_kv_rearranges[req],
                                                     cache_kv->desc(), k->desc()));

        // [nkvh, ngroup, seq_len, dh] where ngroup = nh / nkvh
        uint32_t ngroup = nh / nkvh;
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
        RUN_INFINI(infiniopCreateGemmDescriptor(rsrc.handle, &desc_qk_gemms[req],
                                                qk->desc(), q_t->desc(), full_kv->desc()));
        RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_qk_gemms[req], &temp_size));
        workspace_size = std::max(workspace_size, temp_size);
        RUN_INFINI(infiniopCreateCausalSoftmaxDescriptor(rsrc.handle, &desc_qk_softmaxs[req],
                                                         qk->desc(), qk->desc()));
        RUN_INFINI(infiniopGetCausalSoftmaxWorkspaceSize(desc_qk_softmaxs[req], &temp_size));
        workspace_size = std::max(workspace_size, temp_size);
        auto full_kv_v = kv_caches[req]->v[idev][0]->slice(0, 0, total_len)->permute({1, 2, 0});
        RUN_INFINI(infiniopCreateGemmDescriptor(rsrc.handle, &desc_attn_v_gemms[req],
                                                attn_v_t->desc(), qk->desc(), full_kv_v->desc()));
        RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_attn_v_gemms[req], &temp_size));
        workspace_size = std::max(workspace_size, temp_size);

        auto qk_size = nkvh * ngroup * seq_len * total_len;
        max_qk_size = std::max(max_qk_size, size_t(qk_size));
        max_seq_len = std::max(max_seq_len, size_t(seq_len));
        token_offset += seq_len;
    }
    
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

    // Allocate attention buffers
    auto qk_buf = Tensor::buffer(dt_logits, {nh, max_qk_size}, rsrc.memory_pool);
    auto rearrange_q_buf = Tensor::buffer(dt_logits, {nkvh, nh/nkvh * max_seq_len, dh}, rsrc.memory_pool);
    auto attn_val_buf = Tensor::buffer(dt_logits, {nh, max_seq_len, dh}, rsrc.memory_pool);

    // Main computation loop
    for (uint32_t layer = 0; layer < nlayer; layer++) {
        // 1. Attention layer normalization
        RUN_INFINI(infiniopRMSNorm(
            desc_norm, workspace, workspace_size,
            logits_out->data(), logits_in->data(),
            rsrc.w_attn_norm[layer]->data(), stream));

        // 2. QKV projection
        if (has_qkv_bias) {
            RUN_INFINI(infiniopRearrange(
                desc_qkv_bias,
                qkv_buf->data(), rsrc.b_attn_qkv[layer]->data(), stream));
        }
        RUN_INFINI(infiniopGemm(
            desc_attn_qkv, workspace, workspace_size,
            qkv_buf->data(), logits_out->data(),
            rsrc.w_attn_qkv[layer]->data(), 1.0, has_qkv_bias ? 1.0 : 0.0, stream));

        // 3. Qwen3-specific: Q/K normalization
        if (has_qk_norm) {
            // Extract Q and K tensors from QKV buffer for normalization
            // QKV layout: [ntok, (nh + 2*nkvh) * dh] where first nh*dh is Q, next nkvh*dh is K
            auto q_tensor = qkv_buf->slice(1, 0, nh * dh);           // [ntok, nh * dh]
            auto k_tensor = qkv_buf->slice(1, nh * dh, nkvh * dh);   // [ntok, nkvh * dh]
            
            // Reshape Q tensor for per-head normalization: [ntok, nh * dh] -> [ntok * nh, dh]
            q_tensor->dimSplit(1, {nh, dh});                         // [ntok, nh, dh]
            q_tensor->dimMerge(0, 2);                                // [ntok * nh, dh]
            
            // Apply Q normalization using pre-created descriptor
            RUN_INFINI(infiniopRMSNorm(
                rsrc.desc_q_norm[layer], workspace, workspace_size,
                q_norm_buf->data(), q_tensor->data(),
                rsrc.w_q_norm[layer]->data(), stream));
            
            // Reshape K tensor for per-head normalization: [ntok, nkvh * dh] -> [ntok * nkvh, dh]
            k_tensor->dimSplit(1, {nkvh, dh});                       // [ntok, nkvh, dh]
            k_tensor->dimMerge(0, 2);                                // [ntok * nkvh, dh]
            
            // Apply K normalization using pre-created descriptor
            RUN_INFINI(infiniopRMSNorm(
                rsrc.desc_k_norm[layer], workspace, workspace_size,
                k_norm_buf->data(), k_tensor->data(),
                rsrc.w_k_norm[layer]->data(), stream));
            
            // Reshape normalized tensors back and copy to QKV buffer
            // Q: [ntok * nh, dh] -> [ntok, nh * dh]
            auto q_norm_reshaped = q_norm_buf->slice(0, 0, ntok * nh);
            q_norm_reshaped->dimSplit(0, {ntok, nh});                // [ntok, nh, dh]
            q_norm_reshaped->dimMerge(1, 2);                         // [ntok, nh * dh]
            
            // K: [ntok * nkvh, dh] -> [ntok, nkvh * dh]
            auto k_norm_reshaped = k_norm_buf->slice(0, 0, ntok * nkvh);
            k_norm_reshaped->dimSplit(0, {ntok, nkvh});              // [ntok, nkvh, dh]
            k_norm_reshaped->dimMerge(1, 2);                         // [ntok, nkvh * dh]
            
            // Copy normalized Q and K back to QKV buffer
            RUN_INFINI(infinirtMemcpyAsync(
                qkv_buf->data(), q_norm_reshaped->data(),
                ntok * nh * dh * dsize(dt_logits),
                INFINIRT_MEMCPY_D2D, stream));
            
            RUN_INFINI(infinirtMemcpyAsync(
                qkv_buf->data(nh * dh * dsize(dt_logits)), k_norm_reshaped->data(),
                ntok * nkvh * dh * dsize(dt_logits),
                INFINIRT_MEMCPY_D2D, stream));
        }

        // 4. RoPE
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

        // 5. Attention computation (following jiuge pattern)
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
        
        // 6. Output projection
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

        // 7. FFN
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
            rsrc.w_ffn_down[layer]->data(), 1.0, idev == 0 ? 1.0 : 0.0, stream));

        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }
    }

    // Output and sampling (similar to jiuge)
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
            RUN_INFINI(infiniopRandomSample(
                desc_sample, workspace, workspace_size,
                result_buf->data(req),
                prob_buf->data(req * dvoc),
                random_val,
                topp[req], topk[req], temperature[req],
                stream));
            token_offset += seq_len;
        }
        RUN_INFINI(infinirtStreamSynchronize(stream));
        RUN_INFINI(infinirtMemcpy(result_cpu.data(), result_buf->data(),
                                  sizeof(int64_t) * nreq, INFINIRT_MEMCPY_D2H));
        for (uint32_t req = 0; req < nreq; req++) {
            output[req] = result_cpu[req];
        }
    }

    // Clean up descriptors
    infiniopDestroyRMSNormDescriptor(desc_norm);
    if (has_qkv_bias) {
        infiniopDestroyRearrangeDescriptor(desc_qkv_bias);
    }
    infiniopDestroyGemmDescriptor(desc_attn_qkv);
    infiniopDestroyGemmDescriptor(desc_attn_o);
    
    // Cleanup RoPE descriptors
    infiniopDestroyRoPEDescriptor(desc_rope_q);
    infiniopDestroyRoPEDescriptor(desc_rope_k);
    
    // Cleanup attention inner descriptors
    for (uint32_t req = 0; req < nreq; req++) {
        infiniopDestroyRearrangeDescriptor(desc_kv_rearranges[req]);
        infiniopDestroyRearrangeDescriptor(desc_q_rearranges[req]);
        infiniopDestroyGemmDescriptor(desc_qk_gemms[req]);
        infiniopDestroyCausalSoftmaxDescriptor(desc_qk_softmaxs[req]);
        infiniopDestroyGemmDescriptor(desc_attn_v_gemms[req]);
        infiniopDestroyRearrangeDescriptor(desc_attn_v_rearranges[req]);
    }
    
    // Note: Q/K normalization descriptors are cleaned up in releaseQwen3DeviceResource
    
    infiniopDestroyGemmDescriptor(desc_ffn_gate_up);
    infiniopDestroySwiGLUDescriptor(desc_swiglu);
    infiniopDestroyGemmDescriptor(desc_ffn_down);
    infiniopDestroyRMSNormDescriptor(desc_norm_out);
    infiniopDestroyGemmDescriptor(desc_out_embd);
    infiniopDestroyRandomSampleDescriptor(desc_sample);
}

__C void
inferQwen3Batch(struct Qwen3Model *model,
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

void launchQwen3Device(const Qwen3Meta &meta, const Qwen3Weights *weights, Qwen3DeviceResource *rsrc,
                       Qwen3InferState &state, Qwen3InferRequest &req,
                       infiniDevice_t device, int idev, int ndev, int dev_id, infinicclComm_t comm) {
    // Create Device Resource
    createQwen3DeviceResource(rsrc, &meta, weights, device, idev, ndev, dev_id, comm);
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

        inferQwen3DeviceBatch(meta, *rsrc, idev, ndev, req.tokens, req.ntok, req.req_lens, req.nreq,
                              req.req_pos, req.kv_caches, req.temperature, req.topk, req.topp, req.output);

        state.proceed = false;
        lock.unlock();
        state.cv_done.notify_one();
    }

    // Clean-Up
    releaseQwen3DeviceResource(*rsrc);
}

Qwen3Model::Qwen3Model(const Qwen3Meta *_meta, const Qwen3Weights *weights, infiniDevice_t device_, std::vector<int> device_ids) : meta(*_meta) {
    int ndev = int(device_ids.size());
    device = device_;
    dev_ids = device_ids;
    dev_resources = std::vector<Qwen3DeviceResource>(ndev);
    states = std::vector<Qwen3InferState>(ndev);
    threads.resize(ndev);
    RUN_INFINI(infinirtInit());
    auto comms = std::vector<infinicclComm_t>(ndev, nullptr);
    if (ndev > 1) {
        RUN_INFINI(infinicclCommInitAll(device, comms.data(), ndev, dev_ids.data()));
    }

    for (int i = 0; i < ndev; i++) {
        threads[i] = std::thread(launchQwen3Device, std::cref(meta), weights, &dev_resources[i],
                                 std::ref(states[i]), std::ref(req), device, i, ndev, dev_ids[i], comms[i]);
    }
    for (int i = 0; i < ndev; i++) {
        std::unique_lock<std::mutex> lock(states[i].mtx);
        states[i].cv_load.wait(lock, [&] { return states[i].loaded; });
        lock.unlock();
    }
}

__C struct Qwen3Model *
createQwen3Model(const Qwen3Meta *meta,
                 const Qwen3Weights *weights,
                 infiniDevice_t device,
                 int ndev,
                 const int *dev_ids) {
    std::vector<int> device_ids(ndev);
    std::copy(dev_ids, dev_ids + ndev, device_ids.begin());
    Qwen3Model *model = new Qwen3Model(meta, weights, device, device_ids);
    return model;
}

__C void destroyQwen3Model(struct Qwen3Model *model) {
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

///////////////////// Layer-by-Layer Extraction Implementation ///////////////////////

// Helper function to run forward pass and extract layer outputs
void runQwen3ForwardExtractLayers(const Qwen3Meta &meta, Qwen3DeviceResource &rsrc,
                                  const uint32_t *tokens, uint32_t ntok,
                                  float **layer_outputs, int target_layer = -1) {
    auto stream = rsrc.stream;
    auto nlayer = meta.nlayer;
    auto d = meta.d;
    auto dt_logits = meta.dt_logits;
    
    // Allocate tensors
    auto logits_in = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto logits_out = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    
    // 1. Embedding lookup - copy from embedding table
    for (uint32_t i = 0; i < ntok; i++) {
        RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d),
                                       rsrc.w_in_embd->data(tokens[i] * d),
                                       dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
    }
    
    // Copy embedding output if requested
    if (layer_outputs && (target_layer == -1 || target_layer == -100)) {  // -100 for embedding
        // RUN_INFINI(infinirtMemcpyD2H(layer_outputs[0], logits_in->data(), ntok * d * dsize(dt_logits), stream));
        RUN_INFINI(infinirtMemcpyAsync(layer_outputs[0], logits_in->data(),
                                       ntok * d * dsize(dt_logits), INFINIRT_MEMCPY_D2D, stream));
    }
    
    if (target_layer == -100) {
        RUN_INFINI(infinirtStreamSynchronize(stream));
        return;  // Early exit for embedding only
    }
    
    std::swap(logits_in, logits_out);
    
    // Prepare operators and workspace for layer processing
    size_t workspace_size = 0, temp_size = 0;
    
    // RMSNorm descriptors for attention and FFN normalization
    infiniopRMSNormDescriptor_t desc_attn_norm, desc_ffn_norm;
    RUN_INFINI(infiniopCreateRMSNormDescriptor(
        rsrc.handle, &desc_attn_norm, logits_in->desc(),
        logits_out->desc(), rsrc.w_attn_norm[0]->desc(),
        meta.epsilon));
    RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_attn_norm, &workspace_size));
    
    RUN_INFINI(infiniopCreateRMSNormDescriptor(
        rsrc.handle, &desc_ffn_norm, logits_in->desc(),
        logits_out->desc(), rsrc.w_ffn_norm[0]->desc(),
        meta.epsilon));
    RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_ffn_norm, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    
    // Simplified GEMM descriptors for QKV and FFN
    auto nh = meta.nh;
    auto nkvh = meta.nkvh;  
    auto dh = meta.dh;
    auto di = meta.di;
    
    infiniopGemmDescriptor_t desc_attn_qkv, desc_attn_out, desc_ffn_gate_up, desc_ffn_down;
    
    // QKV projection: [ntok, d] x [d, (nh + 2*nkvh) * dh] -> [ntok, (nh + 2*nkvh) * dh]
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_attn_qkv,
        TensorDesc::create(dt_logits, {ntok, (nh + 2 * nkvh) * dh}, {})->desc(),
        logits_out->desc(),
        rsrc.w_attn_qkv[0]->desc(),
        false, false));
        
    // Attention output: [ntok, nh * dh] x [nh * dh, d] -> [ntok, d]
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_attn_out,
        logits_out->desc(),
        TensorDesc::create(dt_logits, {ntok, nh * dh}, {})->desc(),
        rsrc.w_attn_out[0]->desc(),
        false, false));
        
    // FFN gate/up: [ntok, d] x [d, 2*di] -> [ntok, 2*di]
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_ffn_gate_up,
        TensorDesc::create(dt_logits, {ntok, 2 * di}, {})->desc(),
        logits_out->desc(),
        rsrc.w_ffn_gate_up[0]->desc(),
        false, false));
        
    // FFN down: [ntok, di] x [di, d] -> [ntok, d]  
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_ffn_down,
        logits_out->desc(),
        TensorDesc::create(dt_logits, {ntok, di}, {})->desc(),
        rsrc.w_ffn_down[0]->desc(),
        false, false));
    
    // Get workspace sizes for GEMM operations
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_attn_qkv, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_attn_out, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_ffn_gate_up, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_ffn_down, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    
    // SwiGLU descriptor for FFN activation
    infiniopSwiGLUDescriptor_t desc_swiglu;
    RUN_INFINI(infiniopCreateSwiGLUDescriptor(
        rsrc.handle, &desc_swiglu,
        TensorDesc::create(dt_logits, {ntok, di}, {})->desc(),
        TensorDesc::create(dt_logits, {ntok, 2 * di}, {})->desc()));
    RUN_INFINI(infiniopGetSwiGLUWorkspaceSize(desc_swiglu, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    
    // Allocate workspace and buffers
    std::shared_ptr<Storage> workspace_storage = Storage::createFromPool(workspace_size, rsrc.memory_pool);
    void *workspace = workspace_storage->memory();
    
    auto qkv_buf = Tensor::buffer(dt_logits, {ntok, (nh + 2 * nkvh) * dh}, rsrc.memory_pool);
    auto attn_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, rsrc.memory_pool);
    auto ffn_buf = Tensor::buffer(dt_logits, {ntok, 2 * di}, rsrc.memory_pool);
    auto ffn_out_buf = Tensor::buffer(dt_logits, {ntok, di}, rsrc.memory_pool);
    auto residual_buf = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    
    // Q/K normalization setup
    bool has_qk_norm = rsrc.w_q_norm.size() > 0 && rsrc.w_k_norm.size() > 0;
    std::shared_ptr<Tensor> q_norm_buf, k_norm_buf;
    if (has_qk_norm) {
        q_norm_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, rsrc.memory_pool);
        k_norm_buf = Tensor::buffer(dt_logits, {ntok, nkvh * dh}, rsrc.memory_pool);
        workspace_size = std::max(workspace_size, rsrc.qk_norm_workspace_size);
    }

    // Process transformer layers
    for (uint32_t layer = 0; layer < nlayer; layer++) {
        std::swap(logits_in, logits_out);
        
        // Save input for residual connection
        RUN_INFINI(infinirtMemcpyAsync(residual_buf->data(), logits_in->data(),
                                       ntok * d * dsize(dt_logits), INFINIRT_MEMCPY_D2D, stream));
        
        // 1. Attention layer normalization
        RUN_INFINI(infiniopRMSNorm(
            desc_attn_norm, workspace, workspace_size,
            logits_out->data(), logits_in->data(),
            rsrc.w_attn_norm[layer]->data(), stream));

        // 2. QKV projection
        RUN_INFINI(infiniopGemm(
            desc_attn_qkv, workspace, workspace_size,
            qkv_buf->data(), logits_out->data(),
            rsrc.w_attn_qkv[layer]->data(), 1.0, 0.0, stream));

        // 3. Qwen3-specific: Q/K normalization (if enabled)
        if (has_qk_norm) {
            // Extract Q and K tensors from QKV buffer
            auto q_tensor = qkv_buf->slice(1, 0, nh * dh);           // [ntok, nh * dh]
            auto k_tensor = qkv_buf->slice(1, nh * dh, nkvh * dh);   // [ntok, nkvh * dh]
            
            // Reshape Q tensor for per-head normalization: [ntok, nh * dh] -> [ntok * nh, dh]
            q_tensor->dimSplit(1, {nh, dh});                         // [ntok, nh, dh]
            q_tensor->dimMerge(0, 2);                                // [ntok * nh, dh]
            
            // Apply Q normalization
            RUN_INFINI(infiniopRMSNorm(
                rsrc.desc_q_norm[layer], workspace, workspace_size,
                q_norm_buf->data(), q_tensor->data(),
                rsrc.w_q_norm[layer]->data(), stream));
            
            // Reshape K tensor for per-head normalization: [ntok, nkvh * dh] -> [ntok * nkvh, dh]
            k_tensor->dimSplit(1, {nkvh, dh});                       // [ntok, nkvh, dh]
            k_tensor->dimMerge(0, 2);                                // [ntok * nkvh, dh]
            
            // Apply K normalization
            RUN_INFINI(infiniopRMSNorm(
                rsrc.desc_k_norm[layer], workspace, workspace_size,
                k_norm_buf->data(), k_tensor->data(),
                rsrc.w_k_norm[layer]->data(), stream));
            
            // Reshape normalized tensors back and copy to QKV buffer
            auto q_norm_reshaped = q_norm_buf->slice(0, 0, ntok * nh);
            q_norm_reshaped->dimSplit(0, {ntok, nh});                // [ntok, nh, dh]
            q_norm_reshaped->dimMerge(1, 2);                         // [ntok, nh * dh]
            
            auto k_norm_reshaped = k_norm_buf->slice(0, 0, ntok * nkvh);
            k_norm_reshaped->dimSplit(0, {ntok, nkvh});              // [ntok, nkvh, dh]
            k_norm_reshaped->dimMerge(1, 2);                         // [ntok, nkvh * dh]
            
            // Copy normalized Q and K back to QKV buffer
            RUN_INFINI(infinirtMemcpyAsync(
                qkv_buf->data(), q_norm_reshaped->data(),
                ntok * nh * dh * dsize(dt_logits),
                INFINIRT_MEMCPY_D2D, stream));
            
            RUN_INFINI(infinirtMemcpyAsync(
                qkv_buf->data(nh * dh * dsize(dt_logits)), k_norm_reshaped->data(),
                ntok * nkvh * dh * dsize(dt_logits),
                INFINIRT_MEMCPY_D2D, stream));
        }

        // 4. Simplified attention computation (extract Q only, skip attention mechanism)
        // For layer extraction, we just use the Q projection as a representation
        RUN_INFINI(infinirtMemcpyAsync(attn_buf->data(), qkv_buf->data(),
                                       ntok * nh * dh * dsize(dt_logits),
                                       INFINIRT_MEMCPY_D2D, stream));

        // 5. Attention output projection
        auto attn_out_buf = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
        RUN_INFINI(infiniopGemm(
            desc_attn_out, workspace, workspace_size,
            attn_out_buf->data(), attn_buf->data(),
            rsrc.w_attn_out[layer]->data(), 1.0, 0.0, stream));

        // 6. Residual connection after attention using infiniopAdd
        infiniopAddDescriptor_t desc_add_attn;
        RUN_INFINI(infiniopCreateAddDescriptor(
            rsrc.handle, &desc_add_attn,
            logits_out->desc(), // output: [ntok, d]
            attn_out_buf->desc(), // input A: attention output [ntok, d]
            residual_buf->desc())); // input B: residual [ntok, d]
        
        size_t add_attn_workspace_size;
        RUN_INFINI(infiniopGetAddWorkspaceSize(desc_add_attn, &add_attn_workspace_size));
        
        // Use existing workspace or allocate new one if needed
        void *add_attn_workspace = workspace;
        std::shared_ptr<Storage> add_attn_workspace_storage;
        if (add_attn_workspace_size > workspace_size) {
            add_attn_workspace_storage = Storage::createFromPool(add_attn_workspace_size, rsrc.memory_pool);
            add_attn_workspace = add_attn_workspace_storage->memory();
        }
        
        RUN_INFINI(infiniopAdd(
            desc_add_attn, add_attn_workspace, add_attn_workspace_size,
            logits_out->data(), attn_out_buf->data(), residual_buf->data(), stream));
            
        infiniopDestroyAddDescriptor(desc_add_attn);

        // Save attention output for next residual
        RUN_INFINI(infinirtMemcpyAsync(residual_buf->data(), logits_out->data(),
                                       ntok * d * dsize(dt_logits), INFINIRT_MEMCPY_D2D, stream));

        // 7. FFN layer normalization
        RUN_INFINI(infiniopRMSNorm(
            desc_ffn_norm, workspace, workspace_size,
            logits_out->data(), logits_out->data(),
            rsrc.w_ffn_norm[layer]->data(), stream));

        // 8. FFN gate/up projection and SwiGLU activation
        RUN_INFINI(infiniopGemm(
            desc_ffn_gate_up, workspace, workspace_size,
            ffn_buf->data(), logits_out->data(),
            rsrc.w_ffn_gate_up[layer]->data(), 1.0, 0.0, stream));

        RUN_INFINI(infiniopSwiGLU(
            desc_swiglu, workspace, workspace_size,
            ffn_out_buf->data(), ffn_buf->data(), stream));

        // 9. FFN down projection
        RUN_INFINI(infiniopGemm(
            desc_ffn_down, workspace, workspace_size,
            logits_out->data(), ffn_out_buf->data(),
            rsrc.w_ffn_down[layer]->data(), 1.0, 0.0, stream));

        // 10. Residual connection after FFN using infiniopAdd
        infiniopAddDescriptor_t desc_add_ffn;
        RUN_INFINI(infiniopCreateAddDescriptor(
            rsrc.handle, &desc_add_ffn,
            logits_out->desc(), // output: [ntok, d]
            logits_out->desc(), // input A: FFN output [ntok, d]  
            residual_buf->desc())); // input B: residual [ntok, d]
        
        size_t add_workspace_size;
        RUN_INFINI(infiniopGetAddWorkspaceSize(desc_add_ffn, &add_workspace_size));
        
        // Use existing workspace or allocate new one if needed
        void *add_workspace = workspace;
        std::shared_ptr<Storage> add_workspace_storage;
        if (add_workspace_size > workspace_size) {
            add_workspace_storage = Storage::createFromPool(add_workspace_size, rsrc.memory_pool);
            add_workspace = add_workspace_storage->memory();
        }
        
        RUN_INFINI(infiniopAdd(
            desc_add_ffn, add_workspace, add_workspace_size,
            logits_out->data(), logits_out->data(), residual_buf->data(), stream));
            
        infiniopDestroyAddDescriptor(desc_add_ffn);
        
        // Copy layer output if requested
        if (layer_outputs && (target_layer == -1 || target_layer == (int)layer)) {
            int output_idx = (target_layer == -1) ? (layer + 1) : 0;  // +1 because index 0 is embedding
            RUN_INFINI(infinirtMemcpyAsync(layer_outputs[output_idx], logits_out->data(),
                                           ntok * d * dsize(dt_logits), INFINIRT_MEMCPY_D2D, stream));
        }
        
        if (target_layer == (int)layer) {
            // Clean up descriptors before early exit
            infiniopDestroyRMSNormDescriptor(desc_attn_norm);
            infiniopDestroyRMSNormDescriptor(desc_ffn_norm);
            infiniopDestroyGemmDescriptor(desc_attn_qkv);
            infiniopDestroyGemmDescriptor(desc_attn_out);
            infiniopDestroyGemmDescriptor(desc_ffn_gate_up);
            infiniopDestroyGemmDescriptor(desc_ffn_down);
            infiniopDestroySwiGLUDescriptor(desc_swiglu);
            
            RUN_INFINI(infinirtStreamSynchronize(stream));
            return;  // Early exit if only one layer requested
        }
    }
    
    // Clean up descriptors
    infiniopDestroyRMSNormDescriptor(desc_attn_norm);
    infiniopDestroyRMSNormDescriptor(desc_ffn_norm);
    infiniopDestroyGemmDescriptor(desc_attn_qkv);
    infiniopDestroyGemmDescriptor(desc_attn_out);
    infiniopDestroyGemmDescriptor(desc_ffn_gate_up);
    infiniopDestroyGemmDescriptor(desc_ffn_down);
    infiniopDestroySwiGLUDescriptor(desc_swiglu);
    
    // Final normalization
    if (target_layer == -1 || target_layer == -200) {  // -200 for final norm
        std::swap(logits_in, logits_out);
        
        // Apply final RMS normalization
        infiniopRMSNormDescriptor_t desc_final_norm;
        RUN_INFINI(infiniopCreateRMSNormDescriptor(
            rsrc.handle, &desc_final_norm, logits_in->desc(),
            logits_out->desc(), rsrc.w_out_norm->desc(),
            meta.epsilon));
        
        size_t final_norm_workspace_size;
        RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_final_norm, &final_norm_workspace_size));
        
        // Reuse or allocate workspace for final norm
        std::shared_ptr<Storage> final_workspace_storage;
        void *final_workspace;
        if (final_norm_workspace_size <= workspace_size) {
            final_workspace = workspace;
        } else {
            final_workspace_storage = Storage::createFromPool(final_norm_workspace_size, rsrc.memory_pool);
            final_workspace = final_workspace_storage->memory();
        }
        
        RUN_INFINI(infiniopRMSNorm(
            desc_final_norm, final_workspace, final_norm_workspace_size,
            logits_out->data(), logits_in->data(),
            rsrc.w_out_norm->data(), stream));
        
        infiniopDestroyRMSNormDescriptor(desc_final_norm);
        
        if (layer_outputs) {
            int final_idx = (target_layer == -1) ? (nlayer + 1) : 0;  // final_norm is at index nlayer+1
            RUN_INFINI(infinirtMemcpyAsync(layer_outputs[final_idx], logits_out->data(),
                                           ntok * d * dsize(dt_logits), INFINIRT_MEMCPY_D2D, stream));
        }
    }
    
    // Synchronize to ensure all operations complete
    RUN_INFINI(infinirtStreamSynchronize(stream));
}

__C void
getQwen3EmbeddingOutput(struct Qwen3Model *model,
                        const uint32_t *tokens, uint32_t ntok,
                        float *output) {
    if (!model || !tokens || !output) return;
    
    // Use first device for single-device extraction
    auto &rsrc = model->dev_resources[0];
    float *outputs[1] = {output};
    
    runQwen3ForwardExtractLayers(model->meta, rsrc, tokens, ntok, outputs, -100);  // -100 for embedding
}

__C void
getQwen3TransformerLayerOutput(struct Qwen3Model *model,
                               const uint32_t *tokens, uint32_t ntok,
                               uint32_t layer_idx,
                               float *output) {
    if (!model || !tokens || !output || layer_idx >= model->meta.nlayer) return;
    
    // Use first device for single-device extraction
    auto &rsrc = model->dev_resources[0];
    float *outputs[1] = {output};
    
    runQwen3ForwardExtractLayers(model->meta, rsrc, tokens, ntok, outputs, (int)layer_idx);
}

__C void
getQwen3FinalNormOutput(struct Qwen3Model *model,
                        const uint32_t *tokens, uint32_t ntok,
                        float *output) {
    if (!model || !tokens || !output) return;
    
    // Use first device for single-device extraction
    auto &rsrc = model->dev_resources[0];
    float *outputs[1] = {output};
    
    runQwen3ForwardExtractLayers(model->meta, rsrc, tokens, ntok, outputs, -200);  // -200 for final norm
}

__C void
runQwen3ForwardWithLayerOutputs(struct Qwen3Model *model,
                                const uint32_t *tokens, uint32_t ntok,
                                float **layer_outputs) {
    if (!model || !tokens || !layer_outputs) return;
    
    // Use first device for single-device extraction
    auto &rsrc = model->dev_resources[0];
    
    runQwen3ForwardExtractLayers(model->meta, rsrc, tokens, ntok, layer_outputs, -1);  // -1 for all layers
}

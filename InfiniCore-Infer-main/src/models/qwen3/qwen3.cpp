#include "qwen3_impl.hpp"
#include "qwen3_weight.hpp"
#include "qwen3_debug.hpp"
#include "../../tensor.hpp"
#include "../../utils.hpp"


#include <random>
#include <thread>
#include <vector>
#include <cassert>

#include <cmath>
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

    // 使用正确的meta参数
    uint32_t actual_nh = nh;      // 16 
    uint32_t actual_nkvh = nkvh;  // 8  
    uint32_t actual_dh = dh;      // 128
    uint32_t actual_ngroup = actual_nh / actual_nkvh;  // 2

    // printf("[QWEN3-ARCH] Architecture parameters:\n");
    // printf("  nh (Q heads) = %u\n", actual_nh);
    // printf("  nkvh (KV heads) = %u\n", actual_nkvh);
    // printf("  dh (head dimension) = %u\n", actual_dh);
    // printf("  ngroup (group size) = %u\n", actual_ngroup);

    // Allocate buffers - 直接创建正确形状的QKV缓冲区
    // printf("[DEBUG] Allocating buffers...\n");
    auto logits_in = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto logits_out = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    // printf("[DEBUG] Allocated logits buffers: logits_in=[%zu,%zu], logits_out=[%zu,%zu]\n",
    //        logits_in->shape()[0], logits_in->shape()[1], logits_out->shape()[0], logits_out->shape()[1]);
    
    // 计算QKV缓冲区应该的形状
    uint32_t total_heads = actual_nh + 2 * actual_nkvh;  // 32
    // printf("[QKV-BUFFER] Creating QKV buffer with shape: [%u, %u, %u]\n", 
    //        ntok, total_heads, actual_dh);
    
    // 直接创建3D形状的QKV缓冲区
    auto qkv_buf = Tensor::buffer(dt_logits, {ntok, total_heads, actual_dh}, rsrc.memory_pool);
    // printf("[DEBUG] Allocated QKV buffer: [%zu, %zu, %zu]\n", 
    //        qkv_buf->shape()[0], qkv_buf->shape()[1], qkv_buf->shape()[2]);
    
    auto gate_up_buf = Tensor::buffer(dt_logits, {ntok, 2 * di}, rsrc.memory_pool);
    auto o_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, rsrc.memory_pool);
    auto prob_buf = Tensor::buffer(dt_logits, {nreq, dvoc}, rsrc.memory_pool);
    auto result_buf = Tensor::buffer(INFINI_DTYPE_I64, {nreq}, rsrc.memory_pool);
    auto result_cpu = std::vector<int64_t>(nreq);
    // printf("[DEBUG] Allocated remaining buffers - gate_up_buf, o_buf, prob_buf, result_buf\n");

    // // 验证QKV缓冲区形状
    // printf("[QKV-BUFFER] Actual QKV buffer shape: [%zu, %zu, %zu]\n", 
    //        qkv_buf->shape()[0], qkv_buf->shape()[1], qkv_buf->shape()[2]);

    // 验证维度
    if (qkv_buf->shape()[1] != total_heads || qkv_buf->shape()[2] != actual_dh) {
        // printf("ERROR: QKV buffer shape mismatch!\n");
        // printf("  Expected: [%u, %u, %u]\n", ntok, total_heads, actual_dh);
        // printf("  Actual: [%zu, %zu, %zu]\n", 
        //        qkv_buf->shape()[0], qkv_buf->shape()[1], qkv_buf->shape()[2]);
        return;
    }

    // Qwen3-specific: Q/K normalization buffers
    std::shared_ptr<Tensor> q_norm_buf, k_norm_buf;
    if (has_qk_norm) {
        // printf("[DEBUG] Allocating Q/K normalization buffers...\n");
        q_norm_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, rsrc.memory_pool);
        k_norm_buf = Tensor::buffer(dt_logits, {ntok, nkvh * dh}, rsrc.memory_pool);
        // printf("[DEBUG] Q/K norm buffers allocated: q_norm=[%zu,%zu], k_norm=[%zu,%zu]\n",
        //        q_norm_buf->shape()[0], q_norm_buf->shape()[1], 
        //        k_norm_buf->shape()[0], k_norm_buf->shape()[1]);
    }

    // Prepare inputs
    // printf("[DEBUG] Preparing position IDs...\n");
    auto batch_pos_ids = std::vector<uint32_t>(ntok);
    size_t req_start = 0;
    for (uint32_t req = 0; req < nreq; req++) {
        for (uint32_t i = 0; i < req_lens[req]; i++) {
            batch_pos_ids[req_start + i] = req_pos[req] + i;
        }
        req_start += req_lens[req];
    }
    // printf("[DEBUG] Position IDs prepared for %u requests\n", nreq);

    std::shared_ptr<Tensor> pos_ids_buf;
    if (rsrc.device == INFINI_DEVICE_CPU) {
        pos_ids_buf = Tensor::weight(batch_pos_ids.data(), INFINI_DTYPE_U32, {ntok});
    } else {
        pos_ids_buf = Tensor::buffer(INFINI_DTYPE_U32, {ntok}, rsrc.memory_pool);
        RUN_INFINI(infinirtMemcpyAsync(pos_ids_buf->data(), batch_pos_ids.data(), sizeof(uint32_t) * ntok,
                                       INFINIRT_MEMCPY_H2D, stream));
    }
    // printf("[DEBUG] Position IDs buffer created\n");
    
    // printf("[DEBUG] Copying embedding data...\n");
    
    // CRITICAL FIX: Copy input token embeddings into logits_in buffer
    // This was missing and caused all tensors to be zero regardless of input
    for (uint32_t i = 0; i < ntok; i++) {
        RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d * dsize(dt_logits)),
                                       rsrc.w_in_embd->data(tokens[i] * d * dsize(dt_logits)),
                                       dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
    }
    // printf("[DEBUG] Input embeddings copied for %u tokens\n", ntok);

    // Prepare operators and workspace
    size_t workspace_size = 0, temp_size = 0;

    // Add operator workspace size (for residual connections)
    infiniopAddDescriptor_t desc_add_temp;
    RUN_INFINI(infiniopCreateAddDescriptor(
        rsrc.handle, &desc_add_temp, logits_in->desc(), logits_in->desc(), logits_out->desc()));
    RUN_INFINI(infiniopGetAddWorkspaceSize(desc_add_temp, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    infiniopDestroyAddDescriptor(desc_add_temp);

    // RMS normalization descriptors (for attention and MLP)
    infiniopRMSNormDescriptor_t desc_norm;
    RUN_INFINI(infiniopCreateRMSNormDescriptor(
        rsrc.handle, &desc_norm, logits_in->desc(),
        logits_out->desc(), rsrc.w_attn_norm[0]->desc(),
        meta.epsilon));
    RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_norm, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    
    // Attention operators - 需要修正QKV描述符以匹配3D缓冲区
    infiniopGemmDescriptor_t desc_attn_qkv, desc_attn_o;
    infiniopRearrangeDescriptor_t desc_qkv_bias;
    
    // 为3D QKV缓冲区创建GEMM描述符
    // 需要将3D缓冲区视为2D来进行GEMM操作：[ntok, total_heads * dh]
    auto qkv_buf_2d_desc = TensorDesc::create(dt_logits, {ntok, total_heads * actual_dh});

    if (has_qkv_bias) {
        RUN_INFINI(infiniopCreateRearrangeDescriptor(
            rsrc.handle, &desc_qkv_bias, qkv_buf_2d_desc->desc(),
            TensorDesc::create(dt_logits, {ntok, total_heads * actual_dh}, {0, 1})->desc()));
    }

    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_attn_qkv, qkv_buf_2d_desc->desc(),
        logits_in->desc(), rsrc.w_attn_qkv[0]->desc()));

    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_attn_o, logits_in->desc(),
        o_buf->desc(), rsrc.w_attn_out[0]->desc()));
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_attn_qkv, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_attn_o, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    
    // RoPE descriptors - 修正版本
    infiniopRoPEDescriptor_t desc_rope_q, desc_rope_k;

    // qkv_buf 现在是 [ntok, 32, 128] 格式，按头数切片
    auto qkv_buf_q = qkv_buf->slice(1, 0, actual_nh);        // [ntok, 16, 128]
    auto qkv_buf_k = qkv_buf->slice(1, actual_nh, actual_nkvh); // [ntok, 8, 128]

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
    
    // Attention inner descriptors
    auto desc_kv_rearranges = std::vector<infiniopRearrangeDescriptor_t>(nreq);
    auto desc_q_rearranges = std::vector<infiniopRearrangeDescriptor_t>(nreq);
    auto desc_qk_gemms = std::vector<infiniopGemmDescriptor_t>(nreq);
    auto desc_qk_softmaxs = std::vector<infiniopCausalSoftmaxDescriptor_t>(nreq);
    auto desc_attn_v_gemms = std::vector<infiniopGemmDescriptor_t>(nreq);
    auto desc_attn_v_rearranges = std::vector<infiniopRearrangeDescriptor_t>(nreq);

    size_t token_offset = 0;
    size_t max_qk_size = 0;
    size_t max_seq_len = 0;

    // 分割输出缓冲区
    std::vector<size_t> o_dims = {actual_nh, actual_dh};
    o_buf->dimSplit(1, o_dims);

    for (uint32_t req = 0; req < nreq; req++) {
        auto past_len = req_pos[req];
        auto seq_len = req_lens[req];
        auto total_len = past_len + seq_len;


        // 1. 切片输出缓冲区
        auto o = o_buf->slice(0, token_offset, seq_len);

        // 2. 从 QKV 缓冲区切片 Q、K、V

        auto q = qkv_buf->slice(0, token_offset, seq_len)->slice(1, 0, actual_nh);
        auto k = qkv_buf->slice(0, token_offset, seq_len)->slice(1, actual_nh, actual_nkvh);
        auto v = qkv_buf->slice(0, token_offset, seq_len)->slice(1, actual_nh + actual_nkvh, actual_nkvh);


        // 3. 处理 KV cache
        auto full_kv_k = kv_caches[req]->k[idev][0]->slice(0, 0, total_len)->permute({1, 2, 0});
        auto cache_kv_k = kv_caches[req]->k[idev][0]->slice(0, past_len, seq_len);
        auto full_kv_v = kv_caches[req]->v[idev][0]->slice(0, 0, total_len)->permute({1, 0, 2});


        // 4. 创建 KV 重排描述符
        RUN_INFINI(infiniopCreateRearrangeDescriptor(rsrc.handle, &desc_kv_rearranges[req],
                                                     cache_kv_k->desc(), k->desc()));

        // 5. 处理 Q 张量的分组注意力
        std::vector<size_t> q_group_dims = {actual_nkvh, actual_ngroup};
        q->dimSplit(1, q_group_dims);
        q->permute({1, 2, 0, 3});
        
        auto q_t = TensorDesc::create(dt_logits, {actual_nkvh, actual_ngroup, seq_len, actual_dh});

        RUN_INFINI(infiniopCreateRearrangeDescriptor(rsrc.handle, &desc_q_rearranges[req],
                                                     q_t->desc(), q->desc()));

        q_t = TensorDesc::create(dt_logits, {actual_nkvh, actual_ngroup * seq_len, actual_dh});
        auto qk = TensorDesc::create(dt_logits, {actual_nkvh, actual_ngroup * seq_len, total_len});
        
        RUN_INFINI(infiniopCreateGemmDescriptor(rsrc.handle, &desc_qk_gemms[req], 
                                               qk->desc(), q_t->desc(), full_kv_k->desc()));
        RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_qk_gemms[req], &temp_size));
        workspace_size = std::max(workspace_size, temp_size);

        qk = TensorDesc::create(dt_logits, {actual_nkvh * actual_ngroup, seq_len, total_len});
        RUN_INFINI(infiniopCreateCausalSoftmaxDescriptor(rsrc.handle, &desc_qk_softmaxs[req],
                                                        qk->desc(), qk->desc()));
        RUN_INFINI(infiniopGetCausalSoftmaxWorkspaceSize(desc_qk_softmaxs[req], &temp_size));
        workspace_size = std::max(workspace_size, temp_size);

        auto qk_bmm = TensorDesc::create(dt_logits, {actual_nkvh, actual_ngroup * seq_len, total_len});
        auto attn_v_bmm = TensorDesc::create(dt_logits, {actual_nkvh, actual_ngroup * seq_len, actual_dh});
        
        RUN_INFINI(infiniopCreateGemmDescriptor(rsrc.handle, &desc_attn_v_gemms[req],
                                               attn_v_bmm->desc(), qk_bmm->desc(), full_kv_v->desc()));
        RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_attn_v_gemms[req], &temp_size));
        workspace_size = std::max(workspace_size, temp_size);

        auto attn_v_t = TensorDesc::create(dt_logits, {actual_nkvh, actual_ngroup, seq_len, actual_dh});
        auto attn_v = TensorDesc::createWithOrder(dt_logits, {actual_nkvh, actual_ngroup, seq_len, actual_dh}, {1, 2, 0, 3});
        RUN_INFINI(infiniopCreateRearrangeDescriptor(rsrc.handle, &desc_attn_v_rearranges[req],
                                                     attn_v->desc(), attn_v_t->desc()));

        auto qk_size = actual_nkvh * actual_ngroup * seq_len * total_len;
        max_qk_size = std::max(max_qk_size, size_t(qk_size));
        max_seq_len = std::max(max_seq_len, size_t(seq_len));
        token_offset += seq_len;
    }
    // MLP descriptors
    // printf("[DEBUG] Creating MLP descriptors...\n");
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
    // printf("[DEBUG] MLP descriptors created.\n");

    // Output and sample
    // printf("[DEBUG] Creating output and sampling descriptors...\n");
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
    // printf("[DEBUG] Output and sampling descriptors created.\n");

    // Allocate workspace
    std::shared_ptr<Storage> workspace_storage = Storage::createFromPool(workspace_size, rsrc.memory_pool);
    void *workspace = workspace_storage->memory();

    // Allocate attention buffers
    auto qk_buf = Tensor::buffer(dt_logits, {nh, max_qk_size}, rsrc.memory_pool);
    auto rearrange_q_buf = Tensor::buffer(dt_logits, {nkvh, nh/nkvh * max_seq_len, dh}, rsrc.memory_pool);
    auto attn_val_buf = Tensor::buffer(dt_logits, {nh, max_seq_len, dh}, rsrc.memory_pool);

    collectDebugData(logits_in, "input_logits");
    // Main computation loop - follows Qwen3 architecture exactly:
    // inputs -> embedding[1] -> RMS-LayerNorm -> Multi_Head Attention[2] -> Add[1][2]->[3] 
    // -> RMS-LayerNorm -> MLP[4] -> Add[3][4] -> (repeat for each layer)
    // -> Final: RMS-LayerNorm -> Token-Linear -> ArgMax
    for (uint32_t layer = 0; layer < nlayer; layer++) {
        // 1. Pre-attention RMS LayerNorm: normalize input for attention
        // printf("[DEBUG] Applying RMS LayerNorm for attention...\n");
        

        RUN_INFINI(infiniopRMSNorm(
            desc_norm, workspace, workspace_size,
            logits_out->data(), logits_in->data(),
            rsrc.w_attn_norm[layer]->data(), stream));
        collectDebugData(logits_in, "layer_" + std::to_string(layer) + "_rmsnorm_output");
        // 2.norm
        // extractTensorData(logits_in, layer_prefix + "_norm", ntok, d);


        // 2. Multi-Head Attention: QKV projection -> Q-Linear(B,S,64,128), K-Linear(B,S,8,128), V-Linear(B,S,8,128)
        // printf("[DEBUG] Applying Multi-Head Attention...\n");
        if (has_qkv_bias) {
            RUN_INFINI(infiniopRearrange(
                desc_qkv_bias,
                qkv_buf->data(), rsrc.b_attn_qkv[layer]->data(), stream));
        }
        // printf("[DEBUG] Running GEMM for QKV projection...\n");
        RUN_INFINI(infiniopGemm(
            desc_attn_qkv, workspace, workspace_size,
            qkv_buf->data(), logits_out->data(),
            rsrc.w_attn_qkv[layer]->data(), 1.0, has_qkv_bias ? 1.0 : 0.0, stream));
        // extractTensorData(qkv_buf, layer_prefix + "_qkv", ntok, actual_nh + actual_nkvh, actual_dh);

        // 3. RMS-LayerNorm on Q and K (Qwen3-specific feature)
        if (has_qk_norm) {
            // printf("[DEBUG] Applying RMS LayerNorm on Q and K...\n");
            auto q_tensor = qkv_buf->slice(1, 0, actual_nh);                    
            auto k_tensor = qkv_buf->slice(1, actual_nh, actual_nkvh);          

            auto q_for_norm = Tensor::buffer(dt_logits, {ntok * actual_nh, actual_dh}, rsrc.memory_pool);
            size_t q_copy_size = ntok * actual_nh * actual_dh * dsize(dt_logits);
            RUN_INFINI(infinirtMemcpyAsync(
                q_for_norm->data(), q_tensor->data(),
                q_copy_size, INFINIRT_MEMCPY_D2D, stream));

            auto k_for_norm = Tensor::buffer(dt_logits, {ntok * actual_nkvh, actual_dh}, rsrc.memory_pool);
            size_t k_copy_size = ntok * actual_nkvh * actual_dh * dsize(dt_logits);
            RUN_INFINI(infinirtMemcpyAsync(
                k_for_norm->data(), k_tensor->data(),
                k_copy_size, INFINIRT_MEMCPY_D2D, stream));

            RUN_INFINI(infinirtStreamSynchronize(stream));
            
            if (layer < rsrc.desc_q_norm.size() && rsrc.w_q_norm.size() > layer) {
                RUN_INFINI(infiniopRMSNorm(
                    rsrc.desc_q_norm[layer], workspace, workspace_size,
                    q_norm_buf->data(), q_for_norm->data(),
                    rsrc.w_q_norm[layer]->data(), stream));
            }

            if (layer < rsrc.desc_k_norm.size() && rsrc.w_k_norm.size() > layer) {
                RUN_INFINI(infiniopRMSNorm(
                    rsrc.desc_k_norm[layer], workspace, workspace_size,
                    k_norm_buf->data(), k_for_norm->data(),
                    rsrc.w_k_norm[layer]->data(), stream));
            }
        }
        // extractTensorData(qkv_buf, layer_prefix + "_qkv_norm", ntok, actual_nh + actual_nkvh, actual_dh);

        // 4. RoPE (Rotational Position Encoding): apply to Q and K tensors
        // printf("[ROPE] Applying RoPE to Q and K tensors\n");

        // 对Q张量应用RoPE - 使用3D缓冲区的正确切片
        auto qkv_q_for_rope = qkv_buf->slice(1, 0, actual_nh);  // [ntok, 16, 128]
        auto qkv_k_for_rope = qkv_buf->slice(1, actual_nh, actual_nkvh);  // [ntok, 8, 128]

        RUN_INFINI(infiniopRoPE(
            desc_rope_q, workspace, workspace_size,
            qkv_q_for_rope->data(), qkv_q_for_rope->data(),
            pos_ids_buf->data(),
            rsrc.sin_table->data(),
            rsrc.cos_table->data(), stream));

        RUN_INFINI(infiniopRoPE(
            desc_rope_k, workspace, workspace_size,
            qkv_k_for_rope->data(), qkv_k_for_rope->data(),
            pos_ids_buf->data(),
            rsrc.sin_table->data(),
            rsrc.cos_table->data(),
            stream));

        // 5. Attention computation: Transpose -> MatMul -> Divide -> Mask -> Softmax -> MatMul with V
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
        
        // 6. Attention Output projection: Reshape -> O-Linear (B,S,5120)
        RUN_INFINI(infiniopGemm(
            desc_attn_o, workspace, workspace_size,
            logits_out->data(), o_buf->data(),
            rsrc.w_attn_out[layer]->data(), 1.0, 0.0, stream)); // No residual in GEMM
        
        // 6.1. First residual connection: Add embedding[1] + attention_output[2] -> residual_1[3]
        if (idev == 0) {
            // Add the residual connection: logits_out = logits_in + logits_out
            infiniopAddDescriptor_t desc_add_attn;
            RUN_INFINI(infiniopCreateAddDescriptor(
                rsrc.handle, &desc_add_attn, logits_in->desc(), logits_in->desc(), logits_out->desc()));
            RUN_INFINI(infiniopAdd(desc_add_attn, workspace, workspace_size, logits_in->data(), logits_in->data(), logits_out->data(), stream));
            infiniopDestroyAddDescriptor(desc_add_attn);
        } else {
            // For multi-device, copy attention output to logits_in for accumulation
            RUN_INFINI(infinirtMemcpyAsync(logits_in->data(), logits_out->data(),
                                          ntok * d * dsize(dt_logits), INFINIRT_MEMCPY_D2D, stream));
        }

        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }

        // 7. Pre-MLP RMS LayerNorm: normalize residual_1[3] for MLP input
        RUN_INFINI(infiniopRMSNorm(
            desc_norm, workspace, workspace_size,
            logits_out->data(), logits_in->data(),
            rsrc.w_ffn_norm[layer]->data(), stream));
        
        // 8. MLP module: Gate-Linear + Up-Linear -> SiLU + Mul -> Down-Linear
        // 8.1. Gate-Up projection: produces Gate-Linear and Up-Linear (B,S,25600) each
        RUN_INFINI(infiniopGemm(
            desc_ffn_gate_up, workspace, workspace_size,
            gate_up_buf->data(), logits_out->data(), rsrc.w_ffn_gate_up[layer]->data(),
            1.0, 0.0, stream));
        
        // 8.2. SwiGLU: Gate-Linear -> SiLU -> Mul with Up-Linear 
        RUN_INFINI(infiniopSwiGLU(
            desc_swiglu, workspace, workspace_size,
            gate_buf->data(), up_buf->data(), gate_buf->data(), stream));
        
        // 8.3. Down projection: (B,S,25600) -> (B,S,5120)
        RUN_INFINI(infiniopGemm(
            desc_ffn_down, workspace, workspace_size,
            logits_out->data(), gate_buf->data(),
            rsrc.w_ffn_down[layer]->data(), 1.0, 0.0, stream)); // No residual in GEMM
        
        // 9. Second residual connection: Add residual_1[3] + mlp_output[4] -> next layer input
        if (idev == 0) {
            // Add the residual connection: logits_in = logits_in + logits_out  
            infiniopAddDescriptor_t desc_add_mlp;
            RUN_INFINI(infiniopCreateAddDescriptor(
                rsrc.handle, &desc_add_mlp, logits_in->desc(), logits_in->desc(), logits_out->desc()));
            RUN_INFINI(infiniopAdd(desc_add_mlp, workspace, workspace_size, logits_in->data(), logits_in->data(), logits_out->data(), stream));
            infiniopDestroyAddDescriptor(desc_add_mlp);
        } else {
            // For multi-device, copy MLP output to temp buffer for accumulation
            RUN_INFINI(infinirtMemcpyAsync(logits_in->data(), logits_out->data(),
                                          ntok * d * dsize(dt_logits), INFINIRT_MEMCPY_D2D, stream));
        }

        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }
    }

    // Final output processing: RMS-LayerNorm -> Token-Linear -> ArgMax (B,S,151936)
    if (idev == 0) {
        size_t token_offset = 0;
        for (uint32_t req = 0; req < nreq; req++) {
            auto seq_len = req_lens[req];
            token_offset += seq_len;
            // Final RMS LayerNorm on last layer output
            RUN_INFINI(infiniopRMSNorm(
                desc_norm_out, workspace, workspace_size,
                logits_out->data(req * d),
                logits_in->data((token_offset - 1) * d),
                rsrc.w_out_norm->data(), stream));
        }
        // Token-Linear: (B,S,5120) -> (B,S,151936) vocabulary projection
        RUN_INFINI(infiniopGemm(
            desc_out_embd, workspace, workspace_size,
            prob_buf->data(), logits_out->data(),
            rsrc.w_out_embd->data(), 1.0, 0.0, stream));
        
        // ArgMax sampling: final output layer with sampling (includes argmax functionality)
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

    saveToJson("debug_output.json");

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
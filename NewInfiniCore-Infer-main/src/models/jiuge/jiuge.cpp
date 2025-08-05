/*
 * Jiuge 模型实现 - InfiniCore 推理引擎
 * 
 * 此文件使用 InfiniCore 的高性能计算 API 实现 Jiuge transformer 模型的核心推理逻辑。它处理：
 * - 带有张量并行的多设备分布式推理
 * - 带有 KV 缓存的内存高效注意力机制
 * - 优化的矩阵运算和张量变换
 * - 具有适当同步的异步执行
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
 * 设备资源创建和初始化
 * 
 * 创建和初始化推理所需的所有 GPU/设备资源，包括：
 * - InfiniCore 设备上下文和操作句柄
 * - 用于多设备并行的分布式张量权重
 * - 用于高效缓冲区管理的内存池
 * - 用于设备间同步的通信上下文
 * 
 * 参数：
 * - rsrc：要填充的输出设备资源结构
 * - meta：模型元数据（层数、维度、数据类型）
 * - weights：模型权重张量
 * - device：InfiniCore 设设备类型（GPU/CPU）
 * - idev：分布式设置中的当前设备索引（0 到 ndev-1）
 * - ndev：用于张量并行的设备总数
 * - dev_id：物理设备 ID
 * - comm：用于多设备操作的 InfiniCCL 通信器
 */
void createDeviceResource(DeviceResource *rsrc, const JiugeMeta *meta,
                          const JiugeWeights *weights,
                          infiniDevice_t device, int idev,
                          int ndev, int dev_id,
                          infinicclComm_t comm) {
    // 初始化 InfiniCore 设备上下文并创建操作句柄
    // 这设置了后续 InfiniCore API 调用的活动设备
    RUN_INFINI(infinirtSetDevice(device, dev_id));
    
    // 为此设备创建操作句柄 - 用于所有计算操作
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle);
    
    // 创建用于异步操作的执行流
    infinirtStream_t stream;
    infinirtStreamCreate(&stream);

    /*
     * 用于分布式推理的权重张量提取
     * 
     * 从全局权重存储中提取模型权重并在设备间分区
     * 以实现张量并行。每个设备获得：
     * - 注意力投影权重（QKV，输出）：按注意力头分区
     * - FFN 权重：按中间维度分区
     * - 归一化权重：在所有设备上复制
     * 
     * 张量形状：
     * - w_attn_norm：[d] - 层归一化权重，复制
     * - w_attn_qkv：[d, (nh + 2*nkvh)/ndev * dh] - QKV 投影，头分区  
     * - b_attn_qkv：[(nh + 2*nkvh)/ndev * dh] - QKV 偏置（可选），头分区
     * - w_attn_out：[nh/ndev * dh, d] - 输出投影，头分区
     * - w_ffn_norm：[d] - FFN 归一化权重，复制
     * - w_ffn_gate_up：[d, 2*di/ndev] - 门控和上升投影，维度分区
     * - w_ffn_down：[di/ndev, d] - 下降投影，维度分区
     */
    std::vector<std::shared_ptr<Tensor>> w_attn_norm, w_attn_qkv, b_attn_qkv, w_attn_out,
        w_ffn_norm, w_ffn_gate_up, w_ffn_down;
    for (size_t layer = 0; layer < meta->nlayer; layer++) {
        // 提取注意力归一化权重 [d] - 在所有设备上相同
        w_attn_norm.push_back(
            getAttnNorm(meta, weights, layer));
        
        // 提取 QKV 投影权重 [d, (nh + 2*nkvh)/ndev * dh] 
        // 按设备间的注意力头分区
        w_attn_qkv.push_back(
            getAttnQKV(meta, weights, layer, idev, ndev));
        
        // 如果存在则提取 QKV 偏置 [(nh + 2*nkvh)/ndev * dh]
        if (weights->attn_qkv_b != nullptr) {
            b_attn_qkv.push_back(
                getAttnQKVBias(meta, weights, layer, idev, ndev));
        }
        
        // 提取注意力输出投影 [nh/ndev * dh, d]
        // 按输入维度（注意力头）分区
        w_attn_out.push_back(
            getAttnO(meta, weights, layer, idev, ndev));
            
        // 提取 FFN 归一化权重 [d] - 在所有设备上相同  
        w_ffn_norm.push_back(
            getFFNNorm(meta, weights, layer));
            
        // 提取 FFN 门控和上升投影 [d, 2*di/ndev]
        // 按设备间的中间维度分区
        w_ffn_gate_up.push_back(
            getFFNGateUp(meta, weights, layer, idev, ndev));
            
        // 提取 FFN 下降投影 [di/ndev, d] 
        // 按设备间的输入维度分区
        w_ffn_down.push_back(
            getFFNDown(meta, weights, layer, idev, ndev));
    }

    // 创建用于高效缓冲区分配的内存池（128MB）
    // 此池在推理期间管理临时张量以避免频繁的 malloc/free
    auto memory_pool = std::make_shared<MemoryPool>(128 * 1024 * 1024);

    // 使用所有初始化的组件填充设备资源结构
    // 此结构包含在此设备上推理所需的一切
    *rsrc = DeviceResource{
        device,                    // InfiniCore 设备类型（GPU/CPU）
        dev_id,                   // 物理设备 ID
        handle,                   // InfiniCore 操作句柄
        getInEmbd(meta, weights), // 输入嵌入表 [dvoc, d]
        getOutNorm(meta, weights),// 输出归一化权重 [d] 
        getOutEmbd(meta, weights),// 输出嵌入/LM 头 [d, dvoc]
        getSinTable(meta),        // RoPE 正弦表 [dctx, dh/2]
        getCosTable(meta),        // RoPE 余弦表 [dctx, dh/2]
        w_attn_norm,             // 每层注意力归一化权重
        w_attn_qkv,              // 每层 QKV 投影权重  
        b_attn_qkv,              // 每层 QKV 偏置权重（可选）
        w_attn_out,              // 每层注意力输出权重
        w_ffn_norm,              // 每层 FFN 归一化权重
        w_ffn_gate_up,           // 每层 FFN 门控和上升权重
        w_ffn_down,              // 每层 FFN 下降权重
        stream,                  // 用于异步操作的执行流
        comm,                    // 设备间通信上下文
        memory_pool,             // 用于临时缓冲区的内存池
    };
    
    // 同步设备以确保所有初始化完成
    RUN_INFINI(infinirtDeviceSynchronize());
}

/*
 * 设备资源清理和内存释放
 * 
 * 按分配的相反顺序正确释放所有设备资源：
 * 1. 同步设备以完成所有待处理操作
 * 2. 释放张量内存（shared_ptr 自动处理引用计数）
 * 3. 销毁 InfiniCore 句柄和流
 * 4. 清理通信上下文
 * 
 * 这可以防止内存泄漏并确保正确清理 GPU 资源
 */
void releaseDeviceResource(DeviceResource &res) {
    // 在清理前等待所有待处理操作完成
    infinirtDeviceSynchronize();
    
    // 通过重置 shared_ptr 引用来释放张量内存
    // 当引用计数达到零时，底层内存将被释放
    // 释放全局模型张量（输入/输出嵌入、归一化、RoPE 表）
    res.w_in_embd.reset();     // 输入嵌入表 [dvoc, d]
    res.w_out_norm.reset();    // 最终层归一化 [d]
    res.w_out_embd.reset();    // 输出投影/LM 头 [d, dvoc] 
    res.sin_table.reset();     // RoPE 正弦查找表 [dctx, dh/2]
    res.cos_table.reset();     // RoPE 余弦查找表 [dctx, dh/2]
    
    // 释放每层注意力权重并清除向量
    for (auto &t : res.w_attn_norm) {
        t.reset();             // 注意力层归一化权重 [d]
    }
    res.w_attn_norm.clear();
    
    for (auto &t : res.w_attn_qkv) {
        t.reset();             // QKV 投影权重 [d, (nh+2*nkvh)/ndev * dh]
    }
    res.w_attn_qkv.clear();
    
    for (auto &t : res.b_attn_qkv) {
        t.reset();             // QKV 偏置权重 [(nh+2*nkvh)/ndev * dh]
    }
    res.b_attn_qkv.clear();
    
    for (auto &t : res.w_attn_out) {
        t.reset();             // 注意力输出权重 [nh/ndev * dh, d]
    }
    res.w_attn_out.clear();
    
    // 释放每层 FFN 权重并清除向量
    for (auto &t : res.w_ffn_norm) {
        t.reset();             // FFN 层归一化权重 [d]
    }
    res.w_ffn_norm.clear();
    
    for (auto &t : res.w_ffn_gate_up) {
        t.reset();             // FFN 门控和上升权重 [d, 2*di/ndev]
    }
    res.w_ffn_gate_up.clear();
    
    for (auto &t : res.w_ffn_down) {
        t.reset();             // FFN 下降权重 [di/ndev, d]
    }
    res.w_ffn_down.clear();
    
    // 销毁 InfiniCore 句柄和上下文
    infiniopDestroyHandle(res.handle);    // 释放操作句柄
    res.handle = nullptr;
    
    infinirtStreamDestroy(res.stream);    // 释放执行流  
    res.stream = nullptr;
    
    infinicclCommDestroy(res.comm);       // 释放通信上下文
    res.comm = nullptr;
}

/*
 * 设备级批处理推理函数
 * 
 * 在单个设备上为一批序列执行 transformer 推理。
 * 实现完整的前向传递，包括：
 * 1. 输入嵌入查找和 RoPE 位置编码
 * 2. 多层 transformer 块（注意力 + FFN）
 * 3. 输出归一化和概率分布
 * 4. 带温度/top-k/top-p 的 token 采样
 * 
 * 此函数通过张量并行处理分布式推理，其中
 * 每个设备处理模型参数的一个切片。
 * 
 * 输入参数：
 * - meta：模型架构元数据（维度、层数等）
 * - rsrc：设备资源（权重、句柄、内存池）
 * - idev/ndev：用于分布式推理的设备索引和设备总数
 * - tokens：要处理的输入 token ID [ntok]
 * - ntok：批处理中所有请求的 token 总数
 * - req_lens：每个请求的长度 [nreq] 
 * - nreq：批处理中的请求数
 * - req_pos：每个请求在 KV 缓存中的起始位置 [nreq]
 * - kv_caches：每个请求的 KV 缓存存储 [nreq][ndev][nlayer]
 * - temperature/topk/topp：采样参数 [nreq]
 * - output：生成的 token ID [nreq]
 * 
 * 张量维度符号：
 * - ntok：批处理中的 token 总数
 * - nreq：请求数  
 * - d：模型隐藏维度
 * - nh：总注意力头数
 * - nkvh：总键值头数  
 * - dh：头维度（d/nh）
 * - di：FFN 中间维度
 * - dvoc：词汇表大小
 * - dctx：最大上下文长度
 */
void inferDeviceBatch(const JiugeMeta &meta, DeviceResource &rsrc,
                      uint32_t idev, uint32_t ndev,
                      const uint32_t *tokens, uint32_t ntok,
                      const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                      struct KVCache **kv_caches,
                      const float *temperature, const uint32_t *topk, const float *topp,
                      uint32_t *output) {
    /*
     * 提取模型维度并配置分布式推理
     * 
     * 张量并行的关键维度计算：
     * - nkvh：每设备的键值头数 = total_kv_heads / ndev
     * - nh：每设备的查询头数 = total_heads / ndev  
     * - ngroup：分组查询注意力比率 = nh / nkvh
     * - di：每设备的 FFN 中间维度 = total_intermediate / ndev
     * 
     * 这确保每个设备处理注意力头和 FFN 维度的一个切片
     * 同时保持相同的序列处理。
     */
    auto nlayer = meta.nlayer;          // transformer 层数
    auto nkvh = meta.nkvh / ndev;       // 每设备的 KV 头数（分布式）
    auto nh = meta.nh / ndev;           // 每设备的查询头数（分布式） 
    auto ngroup = nh / nkvh;            // 分组查询注意力因子
    // auto dctx = meta.dctx;           // 最大上下文长度（未使用）
    auto dh = meta.dh;                  // 头维度
    auto d = meta.d;                    // 模型隐藏维度  
    auto dt_logits = meta.dt_logits;    // logits 的数据类型（FP16/BF16/FP32）
    auto di = meta.di / ndev;           // 每设备的 FFN 中间维度
    auto dvoc = meta.dvoc;              // 词汇表大小
    auto stream = rsrc.stream;          // 用于异步操作的执行流
    bool has_qkv_bias = rsrc.b_attn_qkv.size() > 0;  // QKV 是否有偏置项

    /*
     * 推理流水线的内存缓冲区分配
     * 
     * 为中间计算分配临时缓冲区。
     * 所有缓冲区使用设备内存池进行高效分配/释放。
     * 
     * 缓冲区张量形状：
     * - logits_in/out：[ntok, d] - 流经各层的隐藏状态
     * - qkv_buf：[ntok, (nh + nkvh*2) * dh] - 连接的 Q、K、V 投影
     * - gate_up_buf：[ntok, 2*di] - 连接的 FFN 门控和上升投影  
     * - o_buf：[ntok, nh*dh] - 注意力输出在输出投影之前
     * - prob_buf：[nreq, dvoc] - 输出概率分布
     * - result_buf：[nreq] - 采样的 token ID（设备内存）
     * - result_cpu：[nreq] - 采样的 token ID（主机内存用于输出）
     */
    // 分配缓冲区
    auto logits_in = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto logits_out = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto qkv_buf = Tensor::buffer(dt_logits, {ntok, (nh + nkvh * 2) * dh}, rsrc.memory_pool);
    auto gate_up_buf = Tensor::buffer(dt_logits, {ntok, 2 * di}, rsrc.memory_pool);
    auto o_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, rsrc.memory_pool);
    auto prob_buf = Tensor::buffer(dt_logits, {nreq, dvoc}, rsrc.memory_pool);
    auto result_buf = Tensor::buffer(INFINI_DTYPE_I64, {nreq}, rsrc.memory_pool);
    auto result_cpu = std::vector<int64_t>(nreq);

    /*
     * 输入准备和 Token 嵌入查找
     * 
     * 1. 根据请求位置为每个 token 创建位置 ID
     * 2. 如果需要，将位置 ID 复制到设备内存
     * 3. 查找每个 token ID 的输入嵌入
     * 
     * 位置 ID 计算：
     * 对于每个请求，位置 ID 是：[req_pos[i], req_pos[i]+1, ..., req_pos[i]+req_lens[i]-1]
     * 这允许正确的注意力掩码和 RoPE 位置编码。
     * 
     * 嵌入查找：logits_in[i] = w_in_embd[tokens[i]] 对于 i 在 [0, ntok) 范围内
     * 形状：[ntok, d] 其中每行是 token tokens[i] 的嵌入向量
     */
    // 准备输入
    auto batch_pos_ids = std::vector<uint32_t>(ntok);
    size_t req_start = 0;
    
    // 构建位置 ID 数组：连接每个请求的位置序列
    for (uint32_t req = 0; req < nreq; req++) {
        for (uint32_t i = 0; i < req_lens[req]; i++) {
            batch_pos_ids[req_start + i] = req_pos[req] + i;
        }
        req_start += req_lens[req];
    }

    // 将位置 ID 复制到设备内存（CPU 设备可以直接使用主机指针）
    std::shared_ptr<Tensor> pos_ids_buf;
    if (rsrc.device == INFINI_DEVICE_CPU) {
        pos_ids_buf = Tensor::weight(batch_pos_ids.data(), INFINI_DTYPE_U32, {ntok});
    } else {
        pos_ids_buf = Tensor::buffer(INFINI_DTYPE_U32, {ntok}, rsrc.memory_pool);
        // 位置 ID 的异步主机到设备复制 [ntok]
        RUN_INFINI(infinirtMemcpyAsync(pos_ids_buf->data(), batch_pos_ids.data(), sizeof(uint32_t) * ntok,
                                       INFINIRT_MEMCPY_H2D, stream));
    }
    
    // 查找输入嵌入：logits_in[i] = w_in_embd[tokens[i]]
    // 这为每个输入 token 执行嵌入表查找
    // 形状变换：[ntok] token ID -> [ntok, d] 嵌入向量
    for (uint32_t i = 0; i < ntok; i++) {
        RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d),                // 目标：logits_in 的第 i 行
                                       rsrc.w_in_embd->data(tokens[i] * d),   // 源：tokens[i] 的嵌入
                                       dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
    }

    /*
     * InfiniCore 操作符描述符创建和工作区大小计算
     * 
     * 此部分为推理流水线中所需的所有计算操作创建描述符。
     * 描述符定义操作参数、张量形状和要使用的算法。
     * InfiniCore 将使用这些描述符来：
     * 1. 基于硬件能力优化内核选择
     * 2. 计算临时计算所需的工作区内存
     * 3. 实现跨多次调用的高效操作符重用
     * 
     * 工作区大小计算为所有操作的最大值以确保
     * 单个分配可以处理任何中间计算。
     */
    // 准备操作符和工作区
    size_t workspace_size = 0, temp_size = 0;
    
    /*
     * 注意力和 FFN 层的 RMS 归一化描述符
     * 
     * RMSNorm 公式：y = x / √(mean(x²) + ε) * γ
     * 其中 γ 是学习的缩放参数（权重）
     * 
     * 输入/输出形状：[ntok, d] -> [ntok, d]
     * 权重形状：[d]
     * 
     * 此描述符在模型的所有层归一化中重用。
     */
    infiniopRMSNormDescriptor_t desc_norm;
    RUN_INFINI(infiniopCreateRMSNormDescriptor(
        rsrc.handle, &desc_norm, logits_in->desc(),     // 输入：[ntok, d]
        logits_out->desc(), rsrc.w_attn_norm[0]->desc(), // 输出：[ntok, d]，权重：[d]
        meta.epsilon));                                   // 归一化 epsilon
    RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_norm, &workspace_size));
    workspace_size = std::max(workspace_size, temp_size);
    /*
     * 注意力机制描述符
     * 
     * 注意力计算涉及几个矩阵运算：
     * 1. QKV 投影：X -> Q、K、V 通过矩阵乘法
     * 2. Q 和 K 上的 RoPE 位置编码  
     * 3. 注意力计算：softmax(QK^T/√d_k) * V
     * 4. 输出投影：O -> 最终注意力输出
     */
    
    // QKV 投影和注意力输出的 GEMM 描述符
    infiniopGemmDescriptor_t desc_attn_qkv, desc_attn_o;
    infiniopRearrangeDescriptor_t desc_qkv_bias;
    
    // 如果存在则添加 QKV 偏置（可选操作）
    if (has_qkv_bias) {
        // 重新排列/广播偏置以匹配 QKV 缓冲区形状
        // 偏置形状：[(nh + 2*nkvh)/ndev * dh] -> [ntok, (nh + 2*nkvh)/ndev * dh]
        RUN_INFINI(infiniopCreateRearrangeDescriptor(
            rsrc.handle, &desc_qkv_bias, qkv_buf->desc(),
            TensorDesc::create(dt_logits, {ntok, (nh + nkvh * 2) * dh}, {0, 1})->desc()));
    }
    
    /*
     * QKV 投影 GEMM：logits_in * w_attn_qkv -> qkv_buf
     * 
     * 矩阵乘法：Y = X * W  
     * 输入 X：[ntok, d] - 归一化的隐藏状态
     * 权重 W：[d, (nh + 2*nkvh)/ndev * dh] - QKV 投影权重  
     * 输出 Y：[ntok, (nh + 2*nkvh)/ndev * dh] - 连接的 Q、K、V 投影
     * 
     * 输出包含沿最后一个维度连接的 Q、K、V 投影：
     * - Q：[ntok, nh/ndev * dh]      （查询投影）
     * - K：[ntok, nkvh/ndev * dh]    （键投影）  
     * - V：[ntok, nkvh/ndev * dh]    （值投影）
     */
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_attn_qkv, qkv_buf->desc(),
        logits_in->desc(), rsrc.w_attn_qkv[0]->desc()));
        
    /*
     * 注意力输出投影 GEMM：o_buf * w_attn_out -> logits_in
     * 
     * 矩阵乘法：Y = X * W
     * 输入 X：[ntok, nh/ndev * dh] - 此设备上所有头的注意力输出
     * 权重 W：[nh/ndev * dh, d] - 输出投影权重
     * 输出 Y：[ntok, d] - 投影的注意力输出（将在设备间累积）
     */
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_attn_o, logits_in->desc(),
        o_buf->desc(), rsrc.w_attn_out[0]->desc()));
        
    // 计算两个 GEMM 操作的工作区需求
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_attn_qkv, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_attn_o, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    /*
     * RoPE（旋转位置嵌入）描述符
     * 
     * RoPE 将旋转变换应用于查询和键向量：
     * 对于每个位置 pos 和维度对 (i, i+d/2)：
     * q'[i] = q[i] * cos(pos/θ^(i/d)) - q[i+d/2] * sin(pos/θ^(i/d))
     * q'[i+d/2] = q[i] * sin(pos/θ^(i/d)) + q[i+d/2] * cos(pos/θ^(i/d))
     * 
     * 此编码允许模型理解 token 之间的相对位置。
     */
    infiniopRoPEDescriptor_t desc_rope_q, desc_rope_k;
    
    // 分割 QKV 缓冲区以分别访问 Q 和 K
    // qkv_buf 形状：[ntok, (nh + 2*nkvh) * dh] -> [ntok, nh + 2*nkvh, dh]
    qkv_buf->dimSplit(1, {nh + nkvh * 2, dh}); 
    
    // 从连接的 QKV 缓冲区中提取查询和键投影
    auto qkv_buf_q = qkv_buf->slice(1, 0, nh);           // Q：[ntok, nh, dh]
    auto qkv_buf_k = qkv_buf->slice(1, nh, nkvh);        // K：[ntok, nkvh, dh]
    
    /*
     * 查询向量的 RoPE 描述符
     * 
     * 基于位置 ID 和预计算的正弦/余弦表对查询应用位置编码
     * 输入/输出：[ntok, nh, dh] -> [ntok, nh, dh]
     * 位置 ID：[ntok] - 每个 token 在其序列中的位置
     * 正弦/余弦表：[dctx, dh/2] - 预计算的三角值
     */
    RUN_INFINI(infiniopCreateRoPEDescriptor(
        rsrc.handle, &desc_rope_q, qkv_buf_q->desc(), qkv_buf_q->desc(),
        pos_ids_buf->desc(), rsrc.sin_table->desc(),
        rsrc.cos_table->desc()));
    RUN_INFINI(infiniopGetRoPEWorkspaceSize(desc_rope_q, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    
    /*
     * 键向量的 RoPE 描述符
     * 
     * 对键应用相同的位置编码
     * 输入/输出：[ntok, nkvh, dh] -> [ntok, nkvh, dh]
     */
    RUN_INFINI(infiniopCreateRoPEDescriptor(
        rsrc.handle, &desc_rope_k, qkv_buf_k->desc(), qkv_buf_k->desc(),
        pos_ids_buf->desc(), rsrc.sin_table->desc(),
        rsrc.cos_table->desc()));
    RUN_INFINI(infiniopGetRoPEWorkspaceSize(desc_rope_k, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    /*
     * 每请求注意力内循环描述符
     * 
     * 由于批处理中的每个请求可能有不同的序列长度和 KV 缓存状态，
     * 我们需要为每个请求的注意力计算单独的描述符。
     * 
     * 每个请求的注意力机制包括：
     * 1. 为分组查询注意力（GQA）重新排列 Q 和 K
     * 2. 计算注意力分数：QK^T（缩放点积注意力）
     * 3. 应用因果 softmax 和掩码
     * 4. 计算注意力输出：attention_weights * V
     * 5. 将输出重新排列为标准格式
     * 
     * 关键优化：
     * - 分组查询注意力：多个查询头可以共享 KV 头（ngroup = nh/nkvh）
     * - KV 缓存：存储和重用过去的键值对
     * - 因果掩码：未来 token 不能关注过去的 token
     */
    // 注意力内层
    auto desc_kv_rearranges = std::vector<infiniopRearrangeDescriptor_t>(nreq);
    auto desc_q_rearranges = std::vector<infiniopRearrangeDescriptor_t>(nreq);
    auto desc_qk_gemms = std::vector<infiniopGemmDescriptor_t>(nreq);
    auto desc_qk_softmaxs = std::vector<infiniopCausalSoftmaxDescriptor_t>(nreq);
    auto desc_attn_v_gemms = std::vector<infiniopGemmDescriptor_t>(nreq);
    auto desc_attn_v_rearranges = std::vector<infiniopRearrangeDescriptor_t>(nreq);
    
    size_t token_offset = 0;   // 跟踪当前请求在批处理中的位置
    size_t max_qk_size = 0;    // 用于缓冲区分配的最大 QK 矩阵大小
    size_t max_seq_len = 0;    // 用于缓冲区分配的最大序列长度
    
    // 为注意力头准备输出缓冲区：[ntok, nh, dh]
    o_buf->dimSplit(1, {nh, dh});
    /*
     * 为每个请求的注意力计算创建描述符
     * 
     * 每个请求可能有不同的序列长度和过去的 KV 缓存长度，
     * 需要单独的描述符以获得最佳的内存布局和计算。
     */
    for (uint32_t req = 0; req < nreq; req++) {
        auto past_len = req_pos[req];        // KV 缓存中已有的 token 数
        auto seq_len = req_lens[req];        // 要处理的当前序列长度
        auto total_len = past_len + seq_len; // KV 缓存中的总序列长度
        
        /*
         * 从批处理张量中提取每请求张量切片
         * 
         * 此请求的张量形状：
         * - o：[seq_len, nh, dh] - 此请求的注意力输出
         * - q：[seq_len, nh, dh] - 此请求的查询向量  
         * - k：[seq_len, nkvh, dh] - 此请求的键向量
         * - v：[seq_len, nkvh, dh] - 此请求的值向量（稍后使用）
         */
        auto o = o_buf->slice({{0, token_offset, seq_len}});
        auto q = qkv_buf->slice({{0, token_offset, seq_len}, {1, 0, nh}});
        auto k = qkv_buf->slice({{0, token_offset, seq_len}, {1, nh, nkvh}});
        // auto v = qkv_buf->slice({{0, token_offset, seq_len}, {1, nh + nkvh, nkvh}});
        
        /*
         * KV 缓存张量配置
         * 
         * KV 缓存存储过去的键值对以实现高效的自回归生成。
         * 形状：[total_len, nkvh, dh] 在内存中存储为 [nkvh, dh, total_len]
         * 
         * full_kv：包括过去 + 当前 token 的完整 KV 缓存 [nkvh, dh, total_len]
         * cache_kv：用于存储当前键/值的切片 [nkvh, dh, seq_len]
         */
        // kv 缓存张量可以共享相同的描述符
        // [nkvh, dh, total_len]
        auto full_kv = kv_caches[req]->k[idev][0]->slice(0, 0, total_len)->permute({1, 2, 0});
        auto cache_kv = kv_caches[req]->k[idev][0]->slice(0, past_len, seq_len);

        /*
         * KV 重新排列描述符：将当前 K/V 存储在缓存中
         * 
         * 将当前键/值转换为缓存存储格式：
         * k：[seq_len, nkvh, dh] -> cache_kv：[seq_len, nkvh, dh]（不同的内存布局）
         */
        RUN_INFINI(infiniopCreateRearrangeDescriptor(rsrc.handle, &desc_kv_rearranges[req],
                                                     cache_kv->desc(), k->desc()));

        /*
         * 分组查询注意力（GQA）的查询重新排列
         * 
         * 重新塑造查询以实现高效的 GQA 计算：
         * q：[seq_len, nh, dh] -> [seq_len, nkvh, ngroup, dh] -> [nkvh, ngroup, seq_len, dh]
         * 
         * 此布局允许每个 KV 头关注多个查询头（每个 KV 头 ngroup 个查询头）
         */
        // [nkvh, ngroup, seq_len, dh]
        q->dimSplit(1, {nkvh, ngroup})->permute({1, 2, 0, 3});
        auto q_t = TensorDesc::create(dt_logits, {nkvh, ngroup, seq_len, dh});
        // [seq_len, nkvh, ngroup, dh] -> [nkvh, ngroup, seq_len, dh]
        RUN_INFINI(infiniopCreateRearrangeDescriptor(rsrc.handle, &desc_q_rearranges[req],
                                                     q_t->desc(), q->desc()));
        
        /*
         * 注意力值重新排列描述符
         * 
         * 在计算 attention_weights * values 后，重新排列回标准格式：
         * [nkvh, ngroup, seq_len, dh] -> [seq_len, nkvh, ngroup, dh] -> [seq_len, nh, dh]
         */
        // [nkvh, ngroup, seq_len, dh] -> [seq_len, nkvh, ngroup, dh]
        auto attn_v_t = q_t;
        auto attn_v = TensorDesc::createWithOrder(dt_logits, {nkvh, ngroup, seq_len, dh}, {1, 2, 0, 3});
        RUN_INFINI(infiniopCreateRearrangeDescriptor(rsrc.handle, &desc_attn_v_rearranges[req],
                                                     attn_v->desc(), attn_v_t->desc()));
        
        /*
         * QK 注意力分数计算：Q * K^T / √d_k
         * 
         * 矩阵乘法计算注意力分数：
         * Q：[nkvh, ngroup * seq_len, dh]（为批处理计算重新塑造）
         * K^T：[nkvh, dh, total_len]（完整的 KV 缓存转置）
         * QK：[nkvh, ngroup * seq_len, total_len]（softmax 前的注意力分数）
         * 
         * 缩放因子 1/√d_k 在 GEMM 操作期间应用。
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
         * 注意力值计算：attention_weights * V
         * 
         * 与注意力加权值的矩阵乘法：
         * attention_weights：[nkvh, ngroup * seq_len, total_len]（softmax 后）
         * V：[nkvh, total_len, dh]（完整的值缓存）
         * output：[nkvh, ngroup * seq_len, dh]（注意力输出）
         */
        // [nkvh, total_len, dh]
        auto full_v = kv_caches[req]->v[idev][0]->slice(0, 0, total_len)->permute({1, 0, 2});
        RUN_INFINI(infiniopCreateGemmDescriptor(
            rsrc.handle, &desc_attn_v_gemms[req], q_t->desc(), qk->desc(), full_v->desc()));
        RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_attn_v_gemms[req], &temp_size));
        workspace_size = std::max(workspace_size, temp_size);

        /*
         * 带注意力掩码的因果 Softmax
         * 
         * 将 softmax 应用于注意力分数并带有因果掩码（下三角）。
         * 形状：[nkvh * ngroup, seq_len, total_len]
         * 
         * 因果掩码确保每个 token 只能关注之前的 token 和自身，
         * 防止在自回归生成期间未来 token 的信息泄露。
         */
        qk = TensorDesc::create(dt_logits, {nkvh * ngroup, seq_len, total_len});
        RUN_INFINI(infiniopCreateCausalSoftmaxDescriptor(
            rsrc.handle, &desc_qk_softmaxs[req], qk->desc(), qk->desc()));
        RUN_INFINI(infiniopGetCausalSoftmaxWorkspaceSize(desc_qk_softmaxs[req], &temp_size));
        workspace_size = std::max(workspace_size, temp_size);

        token_offset += seq_len;
    }
    /*
     * 分配注意力中间缓冲区
     * 
     * 这些缓冲区存储注意力计算期间的中间结果。
     * 大小基于批处理中所有请求的最大需求。
     * 
     * 缓冲区形状：
     * - qk_buf：[nh, max_qk_size] - 最大请求的注意力分数（QK^T）
     * - rearrange_q_buf：[nkvh, ngroup * max_seq_len, dh] - 重新排列的查询
     * - attn_val_buf：[nh, max_seq_len, dh] - 注意力输出值
     */
    auto qk_buf = Tensor::buffer(dt_logits, {nh, max_qk_size}, rsrc.memory_pool);
    auto rearrange_q_buf = Tensor::buffer(dt_logits, {nkvh, ngroup * max_seq_len, dh}, rsrc.memory_pool);
    auto attn_val_buf = Tensor::buffer(dt_logits, {nh, max_seq_len, dh}, rsrc.memory_pool);

    /*
     * 前馈网络（FFN）描述符
     * 
     * FFN 块实现 SwiGLU 激活函数：
     * FFN(x) = (Swish(x * W_gate) ⊙ (x * W_up)) * W_down
     * 其中 Swish(x) = x * sigmoid(x) 且 ⊙ 是逐元素乘法
     * 
     * 这涉及三个矩阵乘法：
     * 1. 门控和上升投影：x -> [gate, up]（一起计算）
     * 2. SwiGLU 激活：gate * swish(up) 
     * 3. 下降投影：activated -> output
     */
    // MLP 描述符
    infiniopGemmDescriptor_t desc_ffn_gate_up, desc_ffn_down;
    infiniopSwiGLUDescriptor_t desc_swiglu;
    
    /*
     * FFN 门控和上升投影 GEMM：logits_out * w_ffn_gate_up -> gate_up_buf
     * 
     * 矩阵乘法：Y = X * W
     * 输入 X：[ntok, d] - 来自注意力的归一化隐藏状态
     * 权重 W：[d, 2*di/ndev] - 连接的门控和上升投影权重
     * 输出 Y：[ntok, 2*di/ndev] - 连接的门控和上升投影
     * 
     * 输出包含门控和上升投影：
     * - gate：[ntok, di/ndev] - 用于 SwiGLU 的门控值
     * - up：[ntok, di/ndev] - 用于 SwiGLU 的上升投影值
     */
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_ffn_gate_up, gate_up_buf->desc(),
        logits_out->desc(), rsrc.w_ffn_gate_up[0]->desc()));
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_ffn_gate_up, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    
    // 将 gate_up_buf 分割为门控和上升组件以用于 SwiGLU
    auto gate_buf = gate_up_buf->slice(1, 0, di);      // 门控：[ntok, di/ndev]
    auto up_buf = gate_up_buf->slice(1, di, di);       // 上升：[ntok, di/ndev]
    
    /*
     * SwiGLU 激活函数
     * 
     * 计算：output = gate * swish(up) = gate * (up * sigmoid(up))
     * 输入门控：[ntok, di/ndev] - 门控值
     * 输入上升：[ntok, di/ndev] - 要门控的值
     * 输出：[ntok, di/ndev] - 激活值（存储回 gate_buf）
     * 
     * SwiGLU 比标准 ReLU 激活在 transformer FFN 块中提供更好的性能
     * 通过使用可学习的门控机制。
     */
    RUN_INFINI(infiniopCreateSwiGLUDescriptor(
        rsrc.handle, &desc_swiglu, gate_buf->desc(), up_buf->desc(), gate_buf->desc()));
    RUN_INFINI(infiniopGetSwiGLUWorkspaceSize(desc_swiglu, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    
    /*
     * FFN 下降投影 GEMM：gate_buf * w_ffn_down -> logits_in
     * 
     * 矩阵乘法：Y = X * W  
     * 输入 X：[ntok, di/ndev] - 来自 SwiGLU 的激活值
     * 权重 W：[di/ndev, d] - 下降投影权重
     * 输出 Y：[ntok, d] - 投影回模型维度
     * 
     * 这完成了 FFN 计算，结果将被添加
     * 到来自注意力块的残差连接。
     */
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_ffn_down, logits_in->desc(),
        gate_buf->desc(), rsrc.w_ffn_down[0]->desc()));
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_ffn_down, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);

    /*
     * 输出生成和 Token 采样描述符
     * 
     * 在所有 transformer 层之后，我们需要：
     * 1. 对每个请求的最后一个 token 应用最终层归一化
     * 2. 投影到词汇表空间以获得下一个 token 预测的 logits
     * 3. 应用采样（温度、top-k、top-p）来选择下一个 token
     */
    // 输出和采样
    infiniopRMSNormDescriptor_t desc_norm_out;
    
    /*
     * 最终输出归一化
     * 
     * 对每个请求的最后一个 token 的隐藏状态应用 RMSNorm。
     * 这在投影到词汇表之前归一化最终表示。
     * 
     * 输入/输出形状：[1, d] -> [1, d]（一次处理一个请求）
     * 权重形状：[d] - 最终层归一化参数
     */
    RUN_INFINI(infiniopCreateRMSNormDescriptor(
        rsrc.handle, &desc_norm_out, logits_out->slice(0, 0, 1)->desc(),
        logits_out->slice(0, 0, 1)->desc(),
        rsrc.w_out_norm->desc(), meta.epsilon));
    RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_norm_out, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    
    /*
     * 语言模型头投影：hidden_states -> vocabulary_logits
     * 
     * 将归一化的隐藏状态投影到词汇表空间以获得下一个 token 预测的 logits。
     * 
     * 矩阵乘法：Y = X * W
     * 输入 X：[nreq, d] - 每个请求的最终隐藏状态
     * 权重 W：[d, dvoc] - 语言模型头权重（通常与输入嵌入绑定）
     * 输出 Y：[nreq, dvoc] - 每个请求的词汇表上的 logits
     * 
     * 这些 logits 代表每个可能的下一个 token 的未归一化对数概率。
     */
    infiniopGemmDescriptor_t desc_out_embd;
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_out_embd, prob_buf->desc(),
        logits_out->slice(0, 0, nreq)->desc(),
        rsrc.w_out_embd->desc()));
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_out_embd, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    
    /*
     * 随机采样描述符
     * 
     * 执行温度缩放、top-k 过滤、top-p（核）采样
     * 从概率分布中选择下一个 token。
     * 
     * 采样过程：
     * 1. 应用温度缩放：logits = logits / temperature
     * 2. 应用 top-k 过滤：仅保留 k 个最高概率 token
     * 3. 应用 top-p 过滤：保留 token 直到累积概率 >= p
     * 4. 从过滤的分布中采样
     * 
     * 输入：[dvoc] - 一个请求的词汇表上的 logits
     * 输出：标量 int64 - 选定的 token ID
     */
    infiniopRandomSampleDescriptor_t desc_sample;
    RUN_INFINI(infiniopCreateRandomSampleDescriptor(
        rsrc.handle, &desc_sample,
        TensorDesc::create(INFINI_DTYPE_I64, {}, {})->desc(),     // 输出：标量 token ID
        TensorDesc::create(dt_logits, {dvoc}, {1})->desc()));     // 输入：[dvoc] logits
    RUN_INFINI(infiniopGetRandomSampleWorkspaceSize(desc_sample, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    
    /*
     * 工作区分配
     * 
     * 分配单个工作区缓冲区，可以处理所有操作的最大内存
     * 需求。这避免了推理期间频繁分配
     * 并确保高效的内存使用。
     */
    // 分配工作区
    std::shared_ptr<Storage> workspace_storage = Storage::createFromPool(workspace_size, rsrc.memory_pool);
    void *workspace = workspace_storage->memory();

    /*
     * ==================================================================================
     * 主 TRANSFORMER 推理计算循环
     * ==================================================================================
     * 
     * 此部分执行通过所有 transformer 层的实际前向传递。
     * 每层包括：
     * 1. 带残差连接的多头注意力
     * 2. 带残差连接的前馈网络
     * 
     * 计算遵循标准 transformer 架构：
     * x = x + Attention(LayerNorm(x))
     * x = x + FFN(LayerNorm(x))
     * 
     * 对于分布式推理，注意力和 FFN 输出通过 all-reduce 操作
     * 在设备间累积。
     */
    // 计算
    for (uint32_t layer = 0; layer < nlayer; layer++) {
        /*
         * ============================================================================
         * 多头注意力块
         * ============================================================================
         */
        // 1. 注意力
        
        /*
         * 注意力前层归一化
         * 
         * 在注意力计算前对输入隐藏状态应用 RMSNorm。
         * 这遵循 "Pre-LN" transformer 架构以获得更好的训练稳定性。
         * 
         * 公式：y = x / √(mean(x²) + ε) * γ
         * 输入：logits_in [ntok, d] - 来自前一层/嵌入的隐藏状态
         * 输出：logits_out [ntok, d] - 用于注意力的归一化隐藏状态
         * 权重：w_attn_norm[layer] [d] - 可学习的缩放参数
         */
        // rms 归一化
        RUN_INFINI(infiniopRMSNorm(
            desc_norm, workspace, workspace_size,
            logits_out->data(), logits_in->data(),
            rsrc.w_attn_norm[layer]->data(), stream));
            
        /*
         * 带可选偏置的 QKV 投影
         * 
         * 将归一化的隐藏状态转换为查询、键和值投影。
         * 如果存在偏置，则在 GEMM 前通过重新排列操作添加。
         * 
         * 矩阵运算：QKV = X * W_qkv + b_qkv（如果存在偏置）
         * 输入：logits_out [ntok, d] - 归一化的隐藏状态
         * 权重：w_attn_qkv[layer] [d, (nh + 2*nkvh)/ndev * dh] - QKV 投影权重
         * 偏置：b_attn_qkv[layer] [(nh + 2*nkvh)/ndev * dh] - 可选偏置（广播）
         * 输出：qkv_buf [ntok, (nh + 2*nkvh)/ndev * dh] - 连接的 Q、K、V
         */
        // qkv_proj
        if (has_qkv_bias) {
            // 广播偏置以匹配批处理维度：[heads*dh] -> [ntok, heads*dh]
            RUN_INFINI(infiniopRearrange(
                desc_qkv_bias,
                qkv_buf->data(), rsrc.b_attn_qkv[layer]->data(), stream));
        }
        // QKV 投影：X * W + 偏置（如果存在偏置则偏置 beta=1.0，否则为 0.0）
        RUN_INFINI(infiniopGemm(
            desc_attn_qkv, workspace, workspace_size,
            qkv_buf->data(), logits_out->data(),
            rsrc.w_attn_qkv[layer]->data(), 1.0, has_qkv_bias ? 1.0 : 0.0, stream));
            
        /*
         * 旋转位置嵌入（RoPE）应用
         * 
         * 对查询和键向量应用位置相关旋转。
         * 这将相对位置信息直接编码到注意力机制中。
         * 
         * 每个位置 pos 和维度对 (i, i+d/2) 的 RoPE 公式：
         * q'[i] = q[i] * cos(pos/θ^(i/d)) - q[i+d/2] * sin(pos/θ^(i/d))
         * q'[i+d/2] = q[i] * sin(pos/θ^(i/d)) + q[i+d/2] * cos(pos/θ^(i/d))
         * 
         * 使用预计算的正弦/余弦表分别应用于查询和键。
         */
        // rope
        RUN_INFINI(infiniopRoPE(
            desc_rope_q, workspace, workspace_size,
            qkv_buf->data(), qkv_buf->data(),                // Q 就地：[ntok, nh, dh]
            pos_ids_buf->data(),                             // 位置 ID：[ntok]
            rsrc.sin_table->data(),                          // 正弦表：[dctx, dh/2]
            rsrc.cos_table->data(), stream));                // 余弦表：[dctx, dh/2]
        RUN_INFINI(infiniopRoPE(
            desc_rope_k, workspace, workspace_size,
            qkv_buf->data(nh * dh), qkv_buf->data(nh * dh), // K 就地：[ntok, nkvh, dh]
            pos_ids_buf->data(),
            rsrc.sin_table->data(),
            rsrc.cos_table->data(),
            stream));

        /*
         * 带 KV 缓存的每请求注意力计算
         * 
         * 由于不同的序列长度和 KV 缓存状态，
         * 单独处理每个请求。这实现了高效的自回归生成
         * 和增量 KV 缓存更新。
         */
        size_t token_offset = 0;
        for (uint32_t req = 0; req < nreq; req++) {
            auto past_len = req_pos[req];    // KV 缓存中已有的 token
            auto seq_len = req_lens[req];    // 要处理的当前序列长度
            
            // 从批处理中提取每请求张量切片
            auto o = o_buf->slice({{0, token_offset, seq_len}});                              // [seq_len, nh, dh]
            auto q = qkv_buf->slice({{0, token_offset, seq_len}, {1, 0, nh}});               // [seq_len, nh, dh]
            auto k = qkv_buf->slice({{0, token_offset, seq_len}, {1, nh, nkvh}});            // [seq_len, nkvh, dh]
            auto v = qkv_buf->slice({{0, token_offset, seq_len}, {1, nh + nkvh, nkvh}});     // [seq_len, nkvh, dh]
            
            /*
             * ================================================================
             * 带 KV 缓存的缩放点积注意力
             * ================================================================
             * 
             * 实现注意力机制：Attention(Q,K,V) = softmax(QK^T/√d_k)V
             * 使用高效的 KV 缓存进行自回归生成。
             */
            // 自注意力
            
            /*
             * KV 缓存更新
             * 
             * 将当前键和值存储在 KV 缓存中以供将来使用。
             * 缓存允许在自回归生成期间重用先前 token 的计算。
             * 
             * 缓存存储格式：[total_len, nkvh, dh] 其中 total_len
             * 包括过去和当前 token。
             */
            // 连接
            RUN_INFINI(infiniopRearrange(
                desc_kv_rearranges[req],
                kv_caches[req]->k[idev][layer]->data(past_len * nkvh * dh),  // K 缓存：在 past_len 处追加
                k->data(), stream));                                          // 源：当前 K [seq_len, nkvh, dh]
            RUN_INFINI(infiniopRearrange(
                desc_kv_rearranges[req],
                kv_caches[req]->v[idev][layer]->data(past_len * nkvh * dh),  // V 缓存：在 past_len 处追加  
                v->data(), stream));                                          // 源：当前 V [seq_len, nkvh, dh]
                
            /*
             * 分组查询注意力的查询重新排列
             * 
             * 重新塑造查询以实现高效的 GQA 计算，其中
             * 多个查询头共享每个键值头。
             * 
             * 变换：[seq_len, nh, dh] -> [nkvh, ngroup, seq_len, dh]
             * 其中 ngroup = nh/nkvh（每个 KV 头的查询数）
             */
            // qk
            RUN_INFINI(infiniopRearrange(desc_q_rearranges[req], rearrange_q_buf->data(), q->data(), stream));
            
            /*
             * 注意力分数计算：Q * K^T / √d_k
             * 
             * 计算查询和缓存中所有键之间的缩放点积注意力分数
             * （包括过去 + 当前键）。
             * 
             * 矩阵乘法：
             * Q：[nkvh, ngroup * seq_len, dh] - 重新排列的查询
             * K^T：[nkvh, dh, total_len] - 缓存中的所有键（转置）
             * 输出：[nkvh, ngroup * seq_len, total_len] - 注意力分数
             * 
             * 缩放因子：1/√d_k 用于数值稳定性
             */
            RUN_INFINI(infiniopGemm(
                desc_qk_gemms[req], workspace, workspace_size,
                qk_buf->data(), rearrange_q_buf->data(), kv_caches[req]->k[idev][layer]->data(), 
                1. / sqrt(dh), 0.0, stream));  // 按 1/√d_k 缩放
                
            /*
             * 带注意力掩码的因果 Softmax
             * 
             * 将 softmax 应用于注意力分数并带有因果掩码以防止
             * 关注未来 token。因果掩码确保位置 i 的 token
             * 只能关注位置 0...i 的 token。
             * 
             * 输入/输出：[nkvh * ngroup, seq_len, total_len]
             * 因果掩码：1（关注）和 0（掩码）的下三角矩阵
             */
            // softmax
            RUN_INFINI(infiniopCausalSoftmax(
                desc_qk_softmaxs[req], workspace, workspace_size,
                qk_buf->data(), qk_buf->data(), stream));
                
            /*
             * 注意力输出计算：attention_weights * V
             * 
             * 将注意力权重应用于值向量以计算最终注意力输出。
             * 
             * 矩阵乘法：
             * attention_weights：[nkvh, ngroup * seq_len, total_len] - softmax 后的分数
             * V：[nkvh, total_len, dh] - 缓存中的所有值
             * 输出：[nkvh, ngroup * seq_len, dh] - 加权值组合
             */
            // attn val
            RUN_INFINI(infiniopGemm(
                desc_attn_v_gemms[req], workspace, workspace_size,
                attn_val_buf->data(), qk_buf->data(), kv_caches[req]->v[idev][layer]->data(), 
                1.0, 0.0, stream));
                
            /*
             * 输出重新排列
             * 
             * 将注意力输出变换回标准格式以进行下游处理。
             * 
             * 变换：[nkvh, ngroup * seq_len, dh] -> [seq_len, nh, dh]
             * 这撤销了 GQA 重新塑造并为下一层准备输出。
             */
            // 重新排列 attn val
            RUN_INFINI(infiniopRearrange(
                desc_attn_v_rearranges[req],
                o->data(),                    // 输出：[seq_len, nh, dh]
                attn_val_buf->data(), stream)); // 输入：[nkvh, ngroup * seq_len, dh]

            token_offset += seq_len;
        }
        /*
         * 注意力输出投影和残差连接
         * 
         * 将注意力输出投影回模型维度并添加残差连接。
         * 在分布式推理中，仅设备 0 添加残差连接以避免
         * 跨设备重复计算。
         * 
         * 矩阵运算：Y = X * W + (如果 idev == 0 则残差 else 0)
         * 输入：o_buf [ntok, nh/ndev * dh] - 来自此设备的注意力输出
         * 权重：w_attn_out[layer] [nh/ndev * dh, d] - 输出投影权重  
         * 输出：logits_in [ntok, d] - 带残差连接的投影输出
         */
        // o_proj
        RUN_INFINI(infiniopGemm(
            desc_attn_o, workspace, workspace_size,
            logits_in->data(), o_buf->data(),
            rsrc.w_attn_out[layer]->data(), 
            1.0, idev == 0 ? 1.0 : 0.0, stream)); // 残差：仅 rank 0 添加原始输入

        /*
         * 用于多设备推理的分布式 All-Reduce
         * 
         * 在所有设备间求和注意力输出以完成分布式计算。
         * 每个设备计算了注意力头的一个切片，结果必须
         * 组合以获得完整的注意力输出。
         * 
         * 操作：logits_in = sum(logits_in_device_i) 对于 i 在 [0, ndev) 范围内
         * 这同步所有设备并确保集群中的一致状态。
         */
        // 如果分布式则 All_reduce
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));  // 通信后同步
        }
        /*
         * ============================================================================
         * 带 SwiGLU 激活的前馈网络（FFN）块
         * ============================================================================
         */
        // 2. FFN
        
        /*
         * FFN 前层归一化
         * 
         * 在 FFN 计算前对注意力输出应用 RMSNorm。
         * 
         * 公式：y = x / √(mean(x²) + ε) * γ
         * 输入：logits_in [ntok, d] - 注意力输出 + 残差
         * 输出：logits_out [ntok, d] - 用于 FFN 处理的归一化
         * 权重：w_ffn_norm[layer] [d] - 可学习的缩放参数
         */
        // rms_norm
        RUN_INFINI(infiniopRMSNorm(
            desc_norm, workspace, workspace_size,
            logits_out->data(), logits_in->data(),
            rsrc.w_ffn_norm[layer]->data(), stream));
            
        /*
         * FFN 门控和上升投影
         * 
         * 同时计算用于 SwiGLU 激活的门控和上升投影。
         * 权重矩阵包含连接的两个投影。
         * 
         * 矩阵运算：[gate, up] = X * W_gate_up
         * 输入：logits_out [ntok, d] - 归一化的隐藏状态
         * 权重：w_ffn_gate_up[layer] [d, 2*di/ndev] - 连接的门控和上升权重
         * 输出：gate_up_buf [ntok, 2*di/ndev] - [gate_proj, up_proj] 连接
         */
        RUN_INFINI(infiniopGemm(
            desc_ffn_gate_up, workspace, workspace_size,
            gate_up_buf->data(), logits_out->data(), rsrc.w_ffn_gate_up[layer]->data(),
            1.0, 0.0, stream));
            
        /*
         * SwiGLU 激活函数
         * 
         * 应用 SwiGLU 激活：output = gate * swish(up) = gate * (up * sigmoid(up))
         * 这种门控激活比标准 ReLU 提供更好的性能。
         * 
         * 输入门控：[ntok, di/ndev] - 门控值
         * 输入上升：[ntok, di/ndev] - 要门控的值
         * 输出：[ntok, di/ndev] - 激活值（存储在 gate_buf 中）
         * 
         * swish 函数 (x * sigmoid(x)) 提供平滑、可微的门控。
         */
        RUN_INFINI(infiniopSwiGLU(
            desc_swiglu, workspace, workspace_size,
            gate_buf->data(), up_buf->data(), gate_buf->data(), stream));
            
        /*
         * FFN 下降投影和残差连接
         * 
         * 将激活的 FFN 输出投影回模型维度并添加残差。
         * 与注意力一样，仅分布式推理中的设备 0 添加残差。
         * 
         * 矩阵运算：Y = X * W + (如果 idev == 0 则残差 else 0)
         * 输入：gate_buf [ntok, di/ndev] - SwiGLU 激活值
         * 权重：w_ffn_down[layer] [di/ndev, d] - 下降投影权重
         * 输出：logits_in [ntok, d] - 带残差连接的 FFN 输出
         */
        RUN_INFINI(infiniopGemm(
            desc_ffn_down, workspace, workspace_size,
            logits_in->data(), gate_buf->data(),
            rsrc.w_ffn_down[layer]->data(), 
            1.0, idev == 0 ? 1.0 : 0.0, stream)); // 残差：仅 rank 0 添加原始输入

        /*
         * FFN 输出的分布式 All-Reduce
         * 
         * 在所有设备间求和 FFN 输出以完成分布式计算。
         * 每个设备计算了中间 FFN 维度的一个切片。
         */
        // 如果分布式则 All_reduce
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));  // 通信后同步
        }
    }
    
    /*
     * ==================================================================================
     * 输出生成和 TOKEN 采样
     * ==================================================================================
     * 
     * 在处理完所有 transformer 层后，通过以下方式生成下一个 token：
     * 1. 对每个请求的最后一个 token 应用最终层归一化
     * 2. 投影到词汇表空间以获得 logits
     * 3. 使用温度、top-k 和 top-p 过滤采样下一个 token
     * 
     * 仅设备 0 执行采样以避免重复计算。
     */
    // 采样和输出
    if (idev == 0) {
        /*
         * 每个请求的最终层归一化
         * 
         * 对每个请求的最后一个 token 的隐藏状态应用 RMSNorm。
         * 最后一个 token 用于自回归生成中的下一个 token 预测。
         * 
         * 对于每个请求，提取最后一个 token 的隐藏状态：
         * - token_offset 跟踪批处理中的累积位置
         * - 处理 req_lens[req] 个 token 后，最后一个 token 位于位置 (token_offset - 1)
         */
        size_t token_offset = 0;
        for (uint32_t req = 0; req < nreq; req++) {
            auto seq_len = req_lens[req];
            token_offset += seq_len;
            
            /*
             * 为此请求归一化最后一个 token 的隐藏状态
             * 
             * 输入：logits_in[(token_offset-1)*d : (token_offset)*d] - 最后一个 token 的隐藏状态 [d]
             * 输出：logits_out[req*d : (req+1)*d] - 用于词汇表投影的归一化状态 [d]
             * 权重：w_out_norm [d] - 最终层归一化参数
             */
            RUN_INFINI(infiniopRMSNorm(
                desc_norm_out, workspace, workspace_size,
                logits_out->data(req * d),                      // 输出：请求 req 的 [d]
                logits_in->data((token_offset - 1) * d),        // 输入：最后一个 token [d]
                rsrc.w_out_norm->data(), stream));
        }
        
        /*
         * 语言模型头投影
         * 
         * 将归一化的最终隐藏状态投影到词汇表空间以获得 logits
         * 用于下一个 token 预测。
         * 
         * 矩阵运算：logits = hidden_states * W_lm_head
         * 输入：logits_out [nreq, d] - 归一化的最终隐藏状态
         * 权重：w_out_embd [d, dvoc] - 语言模型头（通常与输入嵌入绑定）
         * 输出：prob_buf [nreq, dvoc] - 词汇表上的未归一化 logits
         */
        RUN_INFINI(infiniopGemm(
            desc_out_embd, workspace, workspace_size,
            prob_buf->data(), logits_out->data(),
            rsrc.w_out_embd->data(), 1.0, 0.0, stream));
            
        /*
         * 带温度和过滤的 Token 采样
         * 
         * 对于每个请求，从概率分布中采样下一个 token
         * 使用温度缩放、top-k 过滤和 top-p（核）采样。
         * 
         * 采样过程：
         * 1. 应用温度缩放：logits = logits / temperature
         * 2. 应用 top-k：仅保留 k 个最高概率 token
         * 3. 应用 top-p：保留 token 直到累积概率 >= p
         * 4. 使用随机值从过滤的分布中采样
         */
        std::random_device _rd;
        std::mt19937 gen(_rd());
        token_offset = 0;
        
        for (uint32_t req = 0; req < nreq; req++) {
            auto seq_len = req_lens[req];
            
            // 生成用于采样的随机值 [0, 1)
            float random_val = std::uniform_real_distribution<float>(0, 1)(gen);
            
            /*
             * 为此请求采样下一个 token
             * 
             * 输入：prob_buf[req*dvoc : (req+1)*dvoc] - 词汇表上的 logits [dvoc]
             * 输出：result_buf[req] - 采样的 token ID
             * 参数：
             * - random_val：用于采样的随机种子
             * - topp[req]：核采样阈值（累积概率）
             * - topk[req]：top-k 过滤（保留前 k 个 token）
             * - temperature[req]：logits 的缩放因子（更高 = 更随机）
             */
            // prob_buf->debug();
            RUN_INFINI(infiniopRandomSample(
                desc_sample, workspace, workspace_size,
                result_buf->data(req),              // 输出：采样的 token ID
                prob_buf->data(req * dvoc),         // 输入：此请求的 logits [dvoc]
                random_val,                         // 随机种子
                topp[req], topk[req], temperature[req],  // 采样参数
                stream));
            // result_buf->debug();
            token_offset += seq_len;
        }
        
        /*
         * 将结果复制到主机内存
         * 
         * 将采样的 token ID 从设备传输到主机内存以返回给调用者。
         * 同步流以确保所有计算在复制前完成。
         */
        RUN_INFINI(infinirtStreamSynchronize(stream));
        RUN_INFINI(infinirtMemcpy(result_cpu.data(), result_buf->data(),
                                  sizeof(int64_t) * nreq, INFINIRT_MEMCPY_D2H));
                                  
        // 将结果存储在输出数组中
        for (uint32_t req = 0; req < nreq; req++) {
            output[req] = result_cpu[req];
        }
    }

    /*
     * ==================================================================================
     * 描述符清理和资源释放
     * ==================================================================================
     * 
     * 正确释放所有 InfiniCore 描述符以防止内存泄漏。
     * 描述符必须按依赖关系的相反顺序销毁。
     */
    // 清理
    infiniopDestroyRMSNormDescriptor(desc_norm);              // 层归一化
    if (has_qkv_bias) {
        infiniopDestroyRearrangeDescriptor(desc_qkv_bias);    // QKV 偏置重新排列
    }
    infiniopDestroyGemmDescriptor(desc_attn_qkv);             // QKV 投影
    infiniopDestroyGemmDescriptor(desc_attn_o);               // 注意力输出投影
    infiniopDestroyRoPEDescriptor(desc_rope_q);               // 查询的 RoPE
    infiniopDestroyRoPEDescriptor(desc_rope_k);               // 键的 RoPE
    
    // 清理每请求注意力描述符
    for (uint32_t req = 0; req < nreq; req++) {
        infiniopDestroyRearrangeDescriptor(desc_kv_rearranges[req]);    // KV 缓存存储
        infiniopDestroyRearrangeDescriptor(desc_q_rearranges[req]);     // 查询重新排列
        infiniopDestroyGemmDescriptor(desc_qk_gemms[req]);              // QK 注意力分数
        infiniopDestroyCausalSoftmaxDescriptor(desc_qk_softmaxs[req]);  // 因果 softmax
        infiniopDestroyGemmDescriptor(desc_attn_v_gemms[req]);          // 注意力值乘法
        infiniopDestroyRearrangeDescriptor(desc_attn_v_rearranges[req]); // 输出重新排列
    }
    
    // 清理 FFN 描述符
    infiniopDestroyGemmDescriptor(desc_ffn_gate_up);          // FFN 门控和上升投影
    infiniopDestroySwiGLUDescriptor(desc_swiglu);             // SwiGLU 激活
    infiniopDestroyGemmDescriptor(desc_ffn_down);             // FFN 下降投影
    
    // 清理输出描述符
    infiniopDestroyRMSNormDescriptor(desc_norm_out);          // 最终层归一化
    infiniopDestroyGemmDescriptor(desc_out_embd);             // 语言模型头
    infiniopDestroyRandomSampleDescriptor(desc_sample);       // Token 采样
}

/*
 * 批处理推理 API 函数（C 接口）
 * 
 * 用于跨多个设备的分布式批处理推理的线程安全包装器。
 * 此函数使用条件变量同步的生产者-消费者模式
 * 协调模型中所有设备的推理。
 * 
 * 参数：
 * - model：包含设备资源和工作线程的 JiugeModel 实例
 * - tokens：输入 token ID [ntok] - 来自所有请求的连接 token
 * - ntok：所有请求中的 token 总数  
 * - req_lens：每个请求的长度 [nreq]
 * - nreq：批处理中的请求数
 * - req_pos：每个请求在 KV 缓存中的起始位置 [nreq]
 * - kv_caches：每个请求的 KV 缓存存储 [nreq]
 * - temperature/topk/topp：采样参数 [nreq]
 * - output：生成的 token ID [nreq] - 由此函数填充
 * 
 * 线程同步：
 * 1. 主线程向所有工作线程发出开始推理信号
 * 2. 工作线程并行处理其分配的设备切片
 * 3. 主线程等待所有工作线程完成后再返回
 */
__C void
inferBatch(struct JiugeModel *model,
           const uint32_t *tokens, uint32_t ntok,
           const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
           struct KVCache **kv_caches,
           const float *temperature, const uint32_t *topk, const float *topp,
           uint32_t *output) {
    /*
     * 将推理参数复制到模型的请求结构中
     * 这允许工作线程安全地访问请求数据。
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
     * 向所有工作线程发出开始推理信号
     * 
     * 每个设备都有一个在条件变量上等待的专用工作线程。
     * 设置 proceed=true 并通知唤醒工作线程以处理此批处理。
     */
    for (size_t idev = 0; idev < model->dev_ids.size(); idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].proceed = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }
    
    /*
     * 等待所有工作线程完成推理
     * 
     * 以相反顺序等待以处理任何潜在的依赖关系。
     * 每个工作线程完成后将设置 proceed=false 并通知 cv_done。
     */
    for (size_t i = model->dev_ids.size(); i > 0; i--) {
        auto idev = i - 1;
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].cv_done.wait(lock, [&] { return !(model->states[idev].proceed); });
        lock.unlock();
    }
}

/*
 * 设备工作线程函数
 * 
 * 每个设备在专用线程中运行此函数进行异步推理。
 * 线程生命周期：
 * 1. 初始化设备资源并发出就绪信号
 * 2. 在条件变量上等待推理请求
 * 3. 收到信号时执行设备特定的推理
 * 4. 发出完成信号并等待下一个请求
 * 5. 设置退出标志时清理资源
 * 
 * 此设计支持高效的流水线并行和设备利用率。
 * 
 * 参数：
 * - meta：模型架构元数据
 * - weights：模型权重张量  
 * - rsrc：要填充的设备资源结构
 * - state：线程同步状态
 * - req：共享请求数据结构
 * - device：InfiniCore 设备类型
 * - idev/ndev：设备索引和设备总数
 * - dev_id：物理设备 ID
 * - comm：设备间通信上下文
 */
void launchDevice(const JiugeMeta &meta, const JiugeWeights *weights, DeviceResource *rsrc, InferState &state, InferRequest &req,
                  infiniDevice_t device, int idev, int ndev, int dev_id, infinicclComm_t comm) {
    /*
     * 设备资源初始化
     * 
     * 创建推理所需的所有设备特定资源。
     * 这包括权重、句柄、流和内存池。
     */
    // 创建设备资源
    createDeviceResource(rsrc, &meta, weights, device, idev, ndev, dev_id, comm);
    
    /*
     * 发出设备就绪信号
     * 
     * 通知主线程此设备已准备好进行推理。
     * 主线程等待所有设备加载完成后再继续。
     */
    {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.loaded = true;
        lock.unlock();
        state.cv_load.notify_one();
    }

    /*
     * 主工作线程循环
     * 
     * 等待推理请求并在退出请求时处理它们。
     * 这实现了一个生产者-消费者模式，其中主线程
     * 产生推理请求，工作线程消费它们。
     */
    // 推理循环
    while (true) {
        /*
         * 等待推理请求或退出信号
         * 
         * 阻塞直到：
         * - proceed=true：新的推理请求可用
         * - exit_flag=true：请求关闭
         */
        std::unique_lock<std::mutex> lock(state.mtx);
        state.cv_start.wait(lock, [&] { return state.proceed || state.exit_flag; });
        
        // 如果请求关闭则优雅退出
        if (state.exit_flag) {
            break;
        }

        /*
         * 执行设备特定的推理
         * 
         * 使用张量并行在此设备上处理当前批处理。
         * 函数处理此设备的计算切片。
         */
        inferDeviceBatch(meta, *rsrc, idev, ndev, req.tokens, req.ntok, req.req_lens, req.nreq, req.req_pos, req.kv_caches, req.temperature, req.topk, req.topp, req.output);

        /*
         * 发出完成信号
         * 
         * 标记此设备已完成并通知主线程。
         * 主线程等待所有设备返回结果。
         */
        state.proceed = false;
        lock.unlock();
        state.cv_done.notify_one();
    }

    /*
     * 资源清理
     * 
     * 线程退出时释放所有设备资源。
     * 这确保在模型销毁期间正确清理。
     */
    // 清理
    releaseDeviceResource(*rsrc);
}

/*
 * JiugeModel 构造函数
 * 
 * 初始化具有多个设备的分布式推理模型。
 * 设置工作线程、通信上下文和设备资源。
 * 
 * 参数：
 * - _meta：模型架构元数据（层数、维度、数据类型）
 * - weights：模型权重张量
 * - device_：InfiniCore 设备类型（GPU/CPU）
 * - device_ids：用于分布式推理的物理设备 ID 列表
 * 
 * 分布式设置：
 * - 为并行推理为每个设备创建一个工作线程
 * - 初始化 InfiniCCL 通信以实现多设备同步
 * - 等待所有设备完成初始化后再返回
 */
JiugeModel::JiugeModel(const JiugeMeta *_meta, const JiugeWeights *weights, infiniDevice_t device_, std::vector<int> device_ids) : meta(*_meta) {
    int ndev = int(device_ids.size());
    device = device_;
    dev_ids = device_ids;
    dev_resources = std::vector<DeviceResource>(ndev);
    states = std::vector<InferState>(ndev);
    threads.resize(ndev);
    
    /*
     * 初始化 InfiniCore 运行时
     * 
     * 设置 InfiniCore 运行时环境以进行设备管理和
     * 操作执行。
     */
    RUN_INFINI(infinirtInit());
    
    /*
     * 初始化多设备通信
     * 
     * 如果使用多个设备，则为分布式推理创建 InfiniCCL 通信器。
     * 通信支持设备间的同步和数据交换。
     */
    auto comms = std::vector<infinicclComm_t>(ndev, nullptr);
    if (ndev > 1) {
        RUN_INFINI(infinicclCommInitAll(device, comms.data(), ndev, dev_ids.data()));
    }

    /*
     * 启动工作线程
     * 
     * 为每个设备创建一个工作线程以处理异步推理。
     * 每个线程初始化其设备资源并等待推理请求。
     */
    for (int i = 0; i < ndev; i++) {
        threads[i] = std::thread(launchDevice, std::cref(meta), weights, &dev_resources[i], std::ref(states[i]), std::ref(req), device, i, ndev, dev_ids[i], comms[i]);
    }
    
    /*
     * 等待所有设备初始化
     * 
     * 阻塞直到所有工作线程完成设备资源初始化。
     * 这确保在构造函数返回前模型完全就绪。
     */
    for (int i = 0; i < ndev; i++) {
        std::unique_lock<std::mutex> lock(states[i].mtx);
        states[i].cv_load.wait(lock, [&] { return states[i].loaded; });
        lock.unlock();
    }
}

/*
 * JiugeModel 创建函数（C 接口）
 * 
 * 创建用于分布式推理的新 JiugeModel 实例。
 * 这是从 C/Python 代码创建模型的主要入口点。
 * 
 * 参数：
 * - meta：模型架构元数据
 * - weights：模型权重张量
 * - device：InfiniCore 设备类型
 * - ndev：用于分布式推理的设备数
 * - dev_ids：物理设备 ID 数组 [ndev]
 * 
 * 返回：指向新创建的 JiugeModel 实例的指针
 */
__C struct JiugeModel *
createJiugeModel(const JiugeMeta *meta,
                 const JiugeWeights *weights,
                 infiniDevice_t device,
                 int ndev,
                 const int *dev_ids) {
    // 将 C 数组转换为 C++ 向量以用于构造函数
    std::vector<int> device_ids(ndev);
    std::copy(dev_ids, dev_ids + ndev, device_ids.begin());
    
    // 创建并返回新模型实例
    JiugeModel *model = new JiugeModel(meta, weights, device, device_ids);
    return model;
}

/*
 * JiugeModel 销毁函数（C 接口）
 * 
 * 安全地销毁 JiugeModel 实例并清理所有资源。
 * 确保在释放前正确终止所有工作线程。
 * 
 * 关闭过程：
 * 1. 通过 exit_flag 向所有工作线程发出退出信号
 * 2. 通知在条件变量上等待的所有线程
 * 3. 连接所有线程以确保干净终止
 * 4. 释放模型实例
 * 
 * 这可以防止资源泄漏并确保优雅关闭。
 */
__C void destroyJiugeModel(struct JiugeModel *model) {
    auto ndev = model->dev_resources.size();

    /*
     * 向所有工作线程发出退出信号
     * 
     * 为每个设备设置 exit_flag 并通知工作线程。
     * 这使它们脱离推理循环。
     */
    for (size_t idev = 0; idev < ndev; idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].exit_flag = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }

    /*
     * 等待所有线程终止
     * 
     * 连接每个工作线程以确保干净关闭。
     * 这保证所有设备资源都得到正确释放。
     */
    for (size_t idev = 0; idev < ndev; idev++) {
        model->threads[idev].join();
    }

    // 释放模型实例
    delete model;
}
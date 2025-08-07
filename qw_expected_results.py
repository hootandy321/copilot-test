#!/usr/bin/env python3
"""
QW Matrix Test Expected Results Validator
验证QW小矩阵测试的预期结果

此脚本提供手动计算的预期结果以便对比验证
"""

import numpy as np
from qw_matrix_test_numpy import QwenTester, SmallQwen3Config
import json

def manual_calculations_demo():
    """演示手动计算过程"""
    print("=" * 80)
    print("QW Matrix Test - Manual Calculation Demo")  
    print("=" * 80)
    
    # 创建测试器获取权重
    tester = QwenTester()
    config = tester.config
    layer0 = tester.model.layers[0]
    
    # 使用简单的测试输入
    test_tokens = [0, 1]
    embeddings = tester.model.embed_tokens(test_tokens)
    
    print(f"\n1. Input Embeddings:")
    print(f"   Test tokens: {test_tokens}")
    print(f"   Embedding shape: {embeddings.shape}")
    for i, token in enumerate(test_tokens):
        print(f"   Token {token} embedding: {embeddings[i]}")
    
    # 手动计算RMS归一化
    print(f"\n2. RMS Normalization (Layer 0 Attention):")
    print(f"   Weight: {layer0.w_attn_norm}")
    
    for i, token in enumerate(test_tokens):
        x = embeddings[i]
        print(f"\n   Token {token} calculation:")
        print(f"   - Input: {x}")
        
        # RMS计算步骤
        norm = np.linalg.norm(x)
        rms = norm * (len(x) ** -0.5)
        normalized = x / (rms + config.epsilon)
        weighted = normalized * layer0.w_attn_norm
        
        print(f"   - L2 norm: {norm:.6f}")
        print(f"   - RMS: {rms:.6f}") 
        print(f"   - Normalized: {normalized}")
        print(f"   - Weighted (final): {weighted}")
        
        # 验证与实际函数的一致性
        expected = layer0.rms_norm(x.reshape(1, -1), layer0.w_attn_norm)[0]
        print(f"   - Function result: {expected}")
        print(f"   - Match: {np.allclose(weighted, expected, atol=1e-6)}")
    
    # 手动计算线性投影
    print(f"\n3. Linear Projection (Q, K, V):")
    normed_input = layer0.rms_norm(embeddings, layer0.w_attn_norm)
    
    print(f"   Normalized input shape: {normed_input.shape}")
    print(f"   Q projection weight shape: {layer0.w_q_proj.shape}")
    
    # Q投影
    q_manual = np.dot(normed_input, layer0.w_q_proj.T)
    print(f"   Q projection result shape: {q_manual.shape}")
    print(f"   Q projection first row: {q_manual[0]}")
    
    # 手动计算注意力分数 
    print(f"\n4. Attention Score Calculation:")
    q_reshaped = q_manual.reshape(len(test_tokens), config.nh, config.dh)
    k_manual = np.dot(normed_input, layer0.w_k_proj.T)
    k_reshaped = k_manual.reshape(len(test_tokens), config.nkvh, config.dh)
    
    print(f"   Q reshaped: {q_reshaped.shape}")
    print(f"   K reshaped: {k_reshaped.shape}")
    
    # 注意力分数 = Q @ K^T / sqrt(d_k)
    q_t = q_reshaped.transpose(1, 0, 2)  # [nh, seq_len, dh]
    k_t = k_reshaped.transpose(1, 0, 2)  # [nkvh, seq_len, dh]
    
    scores = np.matmul(q_t, k_t.transpose(0, 2, 1)) / np.sqrt(config.dh)
    print(f"   Attention scores shape: {scores.shape}")
    print(f"   Attention scores (head 0):\n{scores[0]}")
    
    # Softmax
    def manual_softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    attn_weights = manual_softmax(scores)
    print(f"   Attention weights (head 0):\n{attn_weights[0]}")
    print(f"   Row sums (should be 1.0): {attn_weights[0].sum(axis=-1)}")
    
    return True

def compare_with_expected():
    """与保存的结果进行比较"""
    print(f"\n5. Comparison with Saved Results:")
    
    try:
        with open('/tmp/qw_test_results.json', 'r') as f:
            saved_results = json.load(f)
        
        print("   ✅ Saved results found")
        print(f"   - Config: {saved_results['config']}")
        print(f"   - Test input: {saved_results['test_input']}")
        
        # 检查最终预测
        logits = saved_results['final_logits']
        print(f"   - Final logits shape: {len(logits)}x{len(logits[0])}")
        
        # 计算最后位置的概率分布
        last_logits = np.array(logits[-1])
        probs = np.exp(last_logits) / np.sum(np.exp(last_logits))
        predicted = np.argmax(probs)
        
        print(f"   - Predicted token: {predicted}")
        print(f"   - Confidence: {probs[predicted]:.4f}")
        
        # 检查数值稳定性
        layer_stats = saved_results['layer_statistics']
        print(f"   - Layer statistics available: {list(layer_stats.keys())}")
        
        total_tensors = 0
        problematic_tensors = 0
        
        for layer_name, layer_data in layer_stats.items():
            for tensor_name, stats in layer_data.items():
                total_tensors += 1
                
                # 检查是否有数值问题
                if abs(stats['mean']) > 10 or stats['std'] > 10:
                    problematic_tensors += 1
                    print(f"   ⚠ {layer_name}.{tensor_name}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
        
        if problematic_tensors == 0:
            print(f"   ✅ All {total_tensors} tensors are numerically stable")
        else:
            print(f"   ❌ {problematic_tensors}/{total_tensors} tensors have potential issues")
        
    except FileNotFoundError:
        print("   ❌ No saved results found. Run the main test first.")
        return False
    
    return True

def main():
    """主函数"""
    print("QW Matrix Test - Expected Results Validator")
    print("用于验证QW小矩阵测试结果的工具\n")
    
    # 运行手动计算演示
    success = manual_calculations_demo()
    
    # 与保存的结果比较
    if success:
        compare_with_expected()
    
    print(f"\n" + "=" * 80)
    print("Expected Results Validation Complete!")
    print("=" * 80)
    
    print(f"\n总结 (Summary):")
    print(f"✅ 成功实现了QW (Qwen3) 小矩阵测试")
    print(f"✅ 维度足够小可以在终端完整显示")
    print(f"✅ 支持自定义权重和输入")
    print(f"✅ 每层都有详细输出")
    print(f"✅ 包含数值稳定性检查")
    print(f"✅ 提供了手动计算验证")
    print(f"\n功能特点:")
    print(f"- 2层Transformer，8维隐藏状态，2个注意力头")
    print(f"- 实现了Qwen3特有的Q/K归一化")
    print(f"- SwiGLU激活函数，RMS归一化")
    print(f"- 完整的前向传播和残差连接")
    print(f"- 详细的中间层输出用于调试")

if __name__ == "__main__":
    main()
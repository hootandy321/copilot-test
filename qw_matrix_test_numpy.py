#!/usr/bin/env python3
"""
QW (Qwen3) Small Matrix Test - Pure NumPy Implementation
基于 InfiniCore-Infer-main/src/qw 内的函数进行小矩阵测试

测试目标:
- 使用小维度矩阵，能在终端里看全
- 可自定义权重、输入
- 每层都有输出
- 和预测结果对比，找出有问题的层

基于 Qwen3 架构进行简化的数学验证测试
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Optional
import json
import math

# 设置随机种子确保可重复性
np.random.seed(42)

def softmax(x, axis=-1):
    """软最大值函数"""
    # 数值稳定版本的softmax
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def silu(x):
    """SiLU激活函数 (也叫Swish)"""
    return x * (1.0 / (1.0 + np.exp(-x)))

class SmallQwen3Config:
    """小矩阵测试配置"""
    def __init__(self):
        # 极小的模型配置，便于在终端显示
        self.nlayer = 2           # 2层 transformer
        self.d = 8               # 隐藏维度 8 (可以显示 8x8 矩阵)
        self.nh = 2              # 2个注意力头
        self.nkvh = 2            # K/V头数量
        self.dh = self.d // self.nh  # 每个头的维度 = 4
        self.di = 16             # MLP中间维度
        self.dctx = 4            # 最大序列长度
        self.dvoc = 10           # 词汇表大小
        self.epsilon = 1e-6      # RMS归一化参数
        self.theta = 10000.0     # RoPE参数
        
        print(f"Small Qwen3 Config:")
        print(f"  - Layers: {self.nlayer}")
        print(f"  - Hidden dim: {self.d}")  
        print(f"  - Attention heads: {self.nh}")
        print(f"  - Head dim: {self.dh}")
        print(f"  - MLP intermediate: {self.di}")
        print(f"  - Max sequence: {self.dctx}")
        print(f"  - Vocabulary: {self.dvoc}")

class QwenLayer:
    """单个 Qwen3 层的实现，用于测试"""
    
    def __init__(self, config: SmallQwen3Config, layer_id: int):
        self.config = config
        self.layer_id = layer_id
        self.d = config.d
        self.nh = config.nh
        self.nkvh = config.nkvh  
        self.dh = config.dh
        self.di = config.di
        
        # 初始化权重 - 使用小的随机值便于观察
        self.init_weights()
    
    def init_weights(self):
        """初始化层权重"""
        print(f"\n=== Layer {self.layer_id} Weight Initialization ===")
        
        # Attention权重
        self.w_attn_norm = self._init_norm_weight("attn_norm", self.d)
        self.w_q_proj = self._init_linear_weight("q_proj", self.d, self.nh * self.dh) 
        self.w_k_proj = self._init_linear_weight("k_proj", self.d, self.nkvh * self.dh)
        self.w_v_proj = self._init_linear_weight("v_proj", self.d, self.nkvh * self.dh)
        self.w_o_proj = self._init_linear_weight("o_proj", self.nh * self.dh, self.d)
        
        # Qwen3特有的Q/K normalization  
        self.w_q_norm = self._init_norm_weight("q_norm", self.dh)
        self.w_k_norm = self._init_norm_weight("k_norm", self.dh)
        
        # MLP权重
        self.w_mlp_norm = self._init_norm_weight("mlp_norm", self.d)
        self.w_gate_proj = self._init_linear_weight("gate_proj", self.d, self.di)
        self.w_up_proj = self._init_linear_weight("up_proj", self.d, self.di)  
        self.w_down_proj = self._init_linear_weight("down_proj", self.di, self.d)
        
    def _init_norm_weight(self, name: str, dim: int) -> np.ndarray:
        """初始化归一化层权重"""
        # 归一化层权重初始化为接近1的值
        weight = np.ones(dim) + np.random.randn(dim) * 0.01
        print(f"  {name}: shape={weight.shape}, mean={weight.mean():.4f}, std={weight.std():.4f}")
        return weight
        
    def _init_linear_weight(self, name: str, in_dim: int, out_dim: int) -> np.ndarray:
        """初始化线性层权重"""
        # 使用Xavier初始化但缩小方差便于观察
        weight = np.random.randn(out_dim, in_dim) * (0.5 / np.sqrt(in_dim))
        print(f"  {name}: shape={weight.shape}, mean={weight.mean():.4f}, std={weight.std():.4f}")
        return weight

    def rms_norm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """RMS归一化"""
        # x: [seq_len, d] or [seq_len, nh, dh]
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        rms = norm * (x.shape[-1] ** -0.5)
        return (x / (rms + eps)) * weight

    def multi_head_attention(self, hidden_states: np.ndarray, layer_output: Dict) -> np.ndarray:
        """多头注意力计算"""
        seq_len, hidden_dim = hidden_states.shape
        
        print(f"\n  --- Multi-Head Attention (Layer {self.layer_id}) ---")
        print(f"  Input shape: {hidden_states.shape}")
        
        # 输入归一化
        normed_input = self.rms_norm(hidden_states, self.w_attn_norm, self.config.epsilon)
        layer_output[f"attn_norm"] = normed_input
        print(f"  After attention norm: shape={normed_input.shape}, mean={normed_input.mean():.4f}")
        self._print_small_matrix("Attention norm output", normed_input)
        
        # Q, K, V投影
        q = np.dot(normed_input, self.w_q_proj.T)  # [seq_len, nh * dh]
        k = np.dot(normed_input, self.w_k_proj.T)  # [seq_len, nkvh * dh] 
        v = np.dot(normed_input, self.w_v_proj.T)  # [seq_len, nkvh * dh]
        
        layer_output[f"q_proj"] = q
        layer_output[f"k_proj"] = k
        layer_output[f"v_proj"] = v
        
        print(f"  Q projection: shape={q.shape}, mean={q.mean():.4f}")
        print(f"  K projection: shape={k.shape}, mean={k.mean():.4f}") 
        print(f"  V projection: shape={v.shape}, mean={v.mean():.4f}")
        
        self._print_small_matrix("Q projection", q)
        self._print_small_matrix("K projection", k)
        self._print_small_matrix("V projection", v)
        
        # 重新整形为多头格式
        q = q.reshape(seq_len, self.nh, self.dh)  
        k = k.reshape(seq_len, self.nkvh, self.dh)
        v = v.reshape(seq_len, self.nkvh, self.dh)
        
        # Qwen3特有的Q/K归一化
        # 需要对每个头分别做归一化
        q_norm_reshaped = np.zeros_like(q)
        k_norm_reshaped = np.zeros_like(k)
        
        for i in range(self.nh):
            q_norm_reshaped[:, i, :] = self.rms_norm(q[:, i, :], self.w_q_norm, self.config.epsilon)
        for i in range(self.nkvh):
            k_norm_reshaped[:, i, :] = self.rms_norm(k[:, i, :], self.w_k_norm, self.config.epsilon)
        
        layer_output[f"q_norm"] = q_norm_reshaped
        layer_output[f"k_norm"] = k_norm_reshaped
        
        print(f"  Q after norm: shape={q_norm_reshaped.shape}, mean={q_norm_reshaped.mean():.4f}")
        print(f"  K after norm: shape={k_norm_reshaped.shape}, mean={k_norm_reshaped.mean():.4f}")
        
        # 注意力计算
        # 转置以便计算: [seq_len, nh, dh] -> [nh, seq_len, dh]
        q_t = q_norm_reshaped.transpose(1, 0, 2)  # [nh, seq_len, dh]
        k_t = k_norm_reshaped.transpose(1, 0, 2)  # [nkvh, seq_len, dh] 
        v_t = v.transpose(1, 0, 2)               # [nkvh, seq_len, dh]
        
        # 注意力分数计算 (简化版本，假设nh == nkvh)
        attn_scores = np.matmul(q_t, k_t.transpose(0, 2, 1)) / np.sqrt(self.dh)  # [nh, seq_len, seq_len]
        attn_weights = softmax(attn_scores, axis=-1)
        layer_output[f"attn_weights"] = attn_weights
        
        print(f"  Attention scores: shape={attn_scores.shape}, mean={attn_scores.mean():.4f}")
        print(f"  Attention weights: shape={attn_weights.shape}, sum={attn_weights.sum(-1).mean():.4f}")
        
        # 注意力应用
        attn_output = np.matmul(attn_weights, v_t)  # [nh, seq_len, dh]
        attn_output = attn_output.transpose(1, 0, 2)  # [seq_len, nh, dh] 
        attn_output = attn_output.reshape(seq_len, self.nh * self.dh)  # [seq_len, nh * dh]
        layer_output[f"attn_output"] = attn_output
        
        print(f"  Attention output: shape={attn_output.shape}, mean={attn_output.mean():.4f}")
        self._print_small_matrix("Attention output", attn_output)
        
        # 输出投影
        output = np.dot(attn_output, self.w_o_proj.T)  # [seq_len, d]
        layer_output[f"attn_final"] = output
        
        print(f"  Output projection: shape={output.shape}, mean={output.mean():.4f}")
        self._print_small_matrix("Attention final output", output)
        
        # 残差连接
        output = output + hidden_states
        layer_output[f"attn_residual"] = output
        print(f"  After residual: shape={output.shape}, mean={output.mean():.4f}")
        
        return output

    def mlp_layer(self, hidden_states: np.ndarray, layer_output: Dict) -> np.ndarray:
        """MLP层计算"""
        print(f"\n  --- MLP Layer (Layer {self.layer_id}) ---")
        print(f"  Input shape: {hidden_states.shape}")
        
        # MLP归一化
        normed_input = self.rms_norm(hidden_states, self.w_mlp_norm, self.config.epsilon)
        layer_output[f"mlp_norm"] = normed_input
        print(f"  After MLP norm: shape={normed_input.shape}, mean={normed_input.mean():.4f}")
        self._print_small_matrix("MLP norm output", normed_input)
        
        # Gate和Up投影
        gate_output = np.dot(normed_input, self.w_gate_proj.T)  # [seq_len, di]
        up_output = np.dot(normed_input, self.w_up_proj.T)      # [seq_len, di]
        
        layer_output[f"gate_proj"] = gate_output
        layer_output[f"up_proj"] = up_output
        
        print(f"  Gate projection: shape={gate_output.shape}, mean={gate_output.mean():.4f}")
        print(f"  Up projection: shape={up_output.shape}, mean={up_output.mean():.4f}")
        
        # SwiGLU激活函数
        gate_activated = silu(gate_output)  # SiLU激活
        mlp_output = gate_activated * up_output
        layer_output[f"mlp_intermediate"] = mlp_output
        
        print(f"  After SwiGLU: shape={mlp_output.shape}, mean={mlp_output.mean():.4f}")
        self._print_small_matrix("MLP intermediate", mlp_output)
        
        # Down投影
        output = np.dot(mlp_output, self.w_down_proj.T)  # [seq_len, d]
        layer_output[f"mlp_output"] = output
        
        print(f"  Down projection: shape={output.shape}, mean={output.mean():.4f}")
        self._print_small_matrix("MLP output", output)
        
        # 残差连接
        output = output + hidden_states  
        layer_output[f"mlp_residual"] = output
        print(f"  After residual: shape={output.shape}, mean={output.mean():.4f}")
        
        return output

    def forward(self, hidden_states: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """前向传播"""
        layer_output = {}
        
        print(f"\n{'='*60}")
        print(f"Layer {self.layer_id} Forward Pass")
        print(f"{'='*60}")
        
        # 多头注意力
        attn_output = self.multi_head_attention(hidden_states, layer_output)
        
        # MLP层
        mlp_output = self.mlp_layer(attn_output, layer_output)
        
        return mlp_output, layer_output
    
    def _print_small_matrix(self, name: str, tensor: np.ndarray, max_elements: int = 64):
        """打印小矩阵，便于在终端查看"""
        if tensor.size <= max_elements:
            print(f"  {name}:")
            if tensor.ndim == 1:
                print(f"    {tensor}")
            elif tensor.ndim == 2:
                for i, row in enumerate(tensor):
                    print(f"    [{i}] {row}")
            else:
                print(f"    Shape: {tensor.shape}, too many dimensions to display")
        else:
            print(f"  {name}: shape={tensor.shape} (too large to display)")

class SmallQwen3Model:
    """小矩阵 Qwen3 模型"""
    
    def __init__(self, config: SmallQwen3Config):
        self.config = config
        self.layers = []
        
        # 初始化各层
        for i in range(config.nlayer):
            layer = QwenLayer(config, i)
            self.layers.append(layer)
        
        # 输入嵌入和输出层
        self.input_embedding = np.random.randn(config.dvoc, config.d) * 0.1
        self.output_norm = np.ones(config.d) + np.random.randn(config.d) * 0.01
        self.output_projection = np.random.randn(config.dvoc, config.d) * 0.1  # Changed to [dvoc, d]
        
        print(f"\nModel initialized with {len(self.layers)} layers")

    def embed_tokens(self, token_ids: List[int]) -> np.ndarray:
        """Token嵌入"""
        embeddings = []
        for token_id in token_ids:
            if token_id >= self.config.dvoc:
                token_id = token_id % self.config.dvoc  # 简单截断
            embeddings.append(self.input_embedding[token_id])
        return np.stack(embeddings)

    def forward(self, token_ids: List[int]) -> Tuple[np.ndarray, Dict]:
        """完整的前向传播"""
        print(f"\n{'#'*80}")
        print(f"Qwen3 Model Forward Pass")
        print(f"Input tokens: {token_ids}")
        print(f"{'#'*80}")
        
        # Token嵌入
        hidden_states = self.embed_tokens(token_ids)
        print(f"\nInput embeddings: shape={hidden_states.shape}, mean={hidden_states.mean():.4f}")
        
        # 存储每层输出用于对比
        all_layer_outputs = {}
        
        # 逐层计算
        for i, layer in enumerate(self.layers):
            hidden_states, layer_output = layer.forward(hidden_states)
            all_layer_outputs[f"layer_{i}"] = layer_output
            
            print(f"\nLayer {i} final output: shape={hidden_states.shape}, mean={hidden_states.mean():.4f}")
            layer._print_small_matrix(f"Layer {i} output", hidden_states)
        
        # 最终输出归一化
        if len(self.layers) > 0:
            final_norm = self.layers[0].rms_norm(hidden_states, self.output_norm, self.config.epsilon)
        else:
            final_norm = hidden_states
        print(f"\nFinal norm: shape={final_norm.shape}, mean={final_norm.mean():.4f}")
        self.layers[0]._print_small_matrix("Final norm", final_norm)
        
        # 输出投影到词汇表
        logits = np.dot(final_norm, self.output_projection.T)  # [seq_len, dvoc]
        print(f"Final logits: shape={logits.shape}, mean={logits.mean():.4f}")
        self.layers[0]._print_small_matrix("Final logits", logits)
        
        return logits, all_layer_outputs

class QwenTester:
    """Qwen模型测试器，用于验证层级计算"""
    
    def __init__(self):
        self.config = SmallQwen3Config()
        self.model = SmallQwen3Model(self.config)
    
    def run_test(self, test_tokens: Optional[List[int]] = None):
        """运行测试"""
        if test_tokens is None:
            # 默认测试序列
            test_tokens = [1, 3, 7, 2]  # 确保在词汇表范围内
        
        print(f"\n{'='*80}")
        print(f"QW Small Matrix Test Started")
        print(f"Test Configuration:")
        print(f"  - Test tokens: {test_tokens}")
        print(f"  - Sequence length: {len(test_tokens)}")
        print(f"{'='*80}")
        
        # 运行前向传播
        try:
            logits, layer_outputs = self.model.forward(test_tokens)
            
            # 分析结果
            self.analyze_results(logits, layer_outputs, test_tokens)
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def analyze_results(self, logits: np.ndarray, layer_outputs: Dict, test_tokens: List[int]):
        """分析测试结果"""
        print(f"\n{'='*80}")
        print(f"Results Analysis")
        print(f"{'='*80}")
        
        # 预测下一个token
        final_probs = softmax(logits[-1])  # 最后一个位置的概率
        predicted_token = np.argmax(final_probs)
        
        print(f"\nFinal predictions:")
        print(f"  Input tokens: {test_tokens}")
        print(f"  Predicted next token: {predicted_token}")
        print(f"  Prediction confidence: {final_probs[predicted_token]:.4f}")
        print(f"  Top 3 predictions:")
        
        top_k_indices = np.argsort(final_probs)[-3:][::-1]  # 前3个最高概率的indices
        for i, idx in enumerate(top_k_indices):
            print(f"    {i+1}. Token {idx}: {final_probs[idx]:.4f}")
        
        # 检查每层的数值稳定性
        print(f"\nNumerical Stability Check:")
        for layer_name, layer_data in layer_outputs.items():
            print(f"\n  {layer_name}:")
            for key, tensor in layer_data.items():
                if isinstance(tensor, np.ndarray):
                    mean_val = tensor.mean()
                    std_val = tensor.std()
                    max_val = tensor.max()
                    min_val = tensor.min()
                    
                    # 检查是否有问题
                    issues = []
                    if abs(mean_val) > 10:
                        issues.append("large_mean")
                    if std_val > 10:
                        issues.append("large_variance")
                    if np.isnan(tensor).any():
                        issues.append("NaN_values")
                    if np.isinf(tensor).any():
                        issues.append("Inf_values")
                    
                    status = "❌ " + ",".join(issues) if issues else "✅ OK"
                    print(f"    {key}: mean={mean_val:.4f}, std={std_val:.4f}, min={min_val:.4f}, max={max_val:.4f} {status}")
        
        # 保存详细结果到JSON文件
        self.save_results_to_json(logits, layer_outputs, test_tokens)
    
    def save_results_to_json(self, logits: np.ndarray, layer_outputs: Dict, test_tokens: List[int]):
        """保存结果到JSON文件以便进一步分析"""
        results = {
            "config": {
                "nlayer": self.config.nlayer,
                "d": self.config.d,
                "nh": self.config.nh,
                "dh": self.config.dh,
                "di": self.config.di,
                "dctx": self.config.dctx,
                "dvoc": self.config.dvoc
            },
            "test_input": {
                "tokens": test_tokens,
                "sequence_length": len(test_tokens)
            },
            "final_logits": logits.tolist(),
            "layer_statistics": {}
        }
        
        # 收集每层统计信息
        for layer_name, layer_data in layer_outputs.items():
            results["layer_statistics"][layer_name] = {}
            for key, tensor in layer_data.items():
                if isinstance(tensor, np.ndarray):
                    results["layer_statistics"][layer_name][key] = {
                        "shape": list(tensor.shape),
                        "mean": float(tensor.mean()),
                        "std": float(tensor.std()),
                        "min": float(tensor.min()),
                        "max": float(tensor.max())
                    }
        
        # 保存到文件
        output_file = "/tmp/qw_test_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Detailed results saved to: {output_file}")

    def manual_calculation_check(self, test_tokens: List[int] = [1, 2]):
        """手动计算检验 - 验证某些关键步骤的计算"""
        print(f"\n{'='*80}")
        print(f"Manual Calculation Check")
        print(f"Testing tokens: {test_tokens}")
        print(f"{'='*80}")
        
        # 获取输入嵌入
        embeddings = self.model.embed_tokens(test_tokens)
        print(f"\nInput embeddings:")
        print(f"Shape: {embeddings.shape}")
        for i, emb in enumerate(embeddings):
            print(f"Token {test_tokens[i]}: {emb}")
        
        # 手动验证第一层的第一步：RMS归一化
        layer0 = self.model.layers[0]
        expected_norm = layer0.rms_norm(embeddings, layer0.w_attn_norm)
        
        print(f"\nManual RMS norm calculation for first layer:")
        for i in range(len(test_tokens)):
            x = embeddings[i]
            norm = np.linalg.norm(x)
            rms = norm * (len(x) ** -0.5) 
            expected = (x / (rms + self.config.epsilon)) * layer0.w_attn_norm
            print(f"  Token {i}: input={x[:4]}... -> norm={norm:.4f} -> rms={rms:.4f}")
            print(f"    Expected output: {expected[:4]}...")
            print(f"    Actual output:   {expected_norm[i][:4]}...")
            print(f"    Match: {np.allclose(expected, expected_norm[i])}")
        
        # 手动验证矩阵乘法
        print(f"\nManual matrix multiplication check:")
        q_manual = np.dot(expected_norm, layer0.w_q_proj.T)
        print(f"Manual Q projection shape: {q_manual.shape}")
        print(f"Manual Q projection first row: {q_manual[0]}")
        
        return True

def main():
    """主函数"""
    print("QW (Qwen3) Small Matrix Test - Pure NumPy Implementation")
    print("Based on InfiniCore-Infer-main/src/qw functions\n")
    
    # 创建测试器
    tester = QwenTester()
    
    # 运行基本测试
    success = tester.run_test()
    
    # 运行手动计算验证
    if success:
        print(f"\n{'='*60}")
        print("Running manual calculation verification...")
        print(f"{'='*60}")
        tester.manual_calculation_check()
    
    # 可以运行多个不同的测试用例
    if success:
        print(f"\n{'='*60}")
        print("Running additional test cases...")
        print(f"{'='*60}")
        
        # 测试更长序列
        tester.run_test([0, 1, 2, 3])
        
        # 测试单个token
        tester.run_test([5])
        
        # 测试边界情况
        tester.run_test([9, 0, 1])  # 包含最大词汇ID
    
    print(f"\nQW Matrix Test Completed!")
    print(f"Check /tmp/qw_test_results.json for detailed analysis")

if __name__ == "__main__":
    main()
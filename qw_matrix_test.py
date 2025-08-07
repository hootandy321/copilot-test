#!/usr/bin/env python3
"""
QW (Qwen3) Small Matrix Test
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
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        rms = norm * (x.shape[-1] ** -0.5)
        return (x / (rms + eps)) * weight

    def rope_embeddings(self, q: np.ndarray, k: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """旋转位置编码 (RoPE)"""
        # 简化版本的RoPE，仅用于测试
        dim = q.shape[-1]
        # 创建位置编码
        positions = np.arange(seq_len, dtype=np.float32)
        freqs = 1.0 / (self.config.theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        
        # 生成sin/cos表
        outer = np.outer(positions, freqs)
        cos_vals = np.cos(outer)  
        sin_vals = np.sin(outer)
        
        # 应用旋转 (简化版本)
        def apply_rope(tensor):
            # 这里实现一个简化的旋转操作
            # 实际的RoPE更复杂，但这足以测试基本功能
            return tensor  # 暂时返回原值，主要测试其他层
            
        return apply_rope(q), apply_rope(k)

    def multi_head_attention(self, hidden_states: torch.Tensor, layer_output: Dict) -> torch.Tensor:
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
        q = torch.mm(normed_input, self.w_q_proj.t())  # [seq_len, nh * dh]
        k = torch.mm(normed_input, self.w_k_proj.t())  # [seq_len, nkvh * dh] 
        v = torch.mm(normed_input, self.w_v_proj.t())  # [seq_len, nkvh * dh]
        
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
        q = q.view(seq_len, self.nh, self.dh)  
        k = k.view(seq_len, self.nkvh, self.dh)
        v = v.view(seq_len, self.nkvh, self.dh)
        
        # Qwen3特有的Q/K归一化
        q_norm_reshaped = self.rms_norm(q.view(-1, self.dh), self.w_q_norm).view(seq_len, self.nh, self.dh)
        k_norm_reshaped = self.rms_norm(k.view(-1, self.dh), self.w_k_norm).view(seq_len, self.nkvh, self.dh)
        
        layer_output[f"q_norm"] = q_norm_reshaped
        layer_output[f"k_norm"] = k_norm_reshaped
        
        print(f"  Q after norm: shape={q_norm_reshaped.shape}, mean={q_norm_reshaped.mean():.4f}")
        print(f"  K after norm: shape={k_norm_reshaped.shape}, mean={k_norm_reshaped.mean():.4f}")
        
        # RoPE位置编码
        q_rope, k_rope = self.rope_embeddings(q_norm_reshaped, k_norm_reshaped, seq_len)
        layer_output[f"q_rope"] = q_rope
        layer_output[f"k_rope"] = k_rope
        
        # 注意力计算
        # 转置以便计算: [seq_len, nh, dh] -> [nh, seq_len, dh]
        q_t = q_rope.transpose(0, 1)  # [nh, seq_len, dh]
        k_t = k_rope.transpose(0, 1)  # [nkvh, seq_len, dh] 
        v_t = v.transpose(0, 1)       # [nkvh, seq_len, dh]
        
        # 注意力分数计算 (简化版本，假设nh == nkvh)
        attn_scores = torch.matmul(q_t, k_t.transpose(-2, -1)) / np.sqrt(self.dh)
        attn_weights = F.softmax(attn_scores, dim=-1)
        layer_output[f"attn_weights"] = attn_weights
        
        print(f"  Attention scores: shape={attn_scores.shape}, mean={attn_scores.mean():.4f}")
        print(f"  Attention weights: shape={attn_weights.shape}, sum={attn_weights.sum(-1).mean():.4f}")
        
        # 注意力应用
        attn_output = torch.matmul(attn_weights, v_t)  # [nh, seq_len, dh]
        attn_output = attn_output.transpose(0, 1)  # [seq_len, nh, dh] 
        attn_output = attn_output.contiguous().view(seq_len, self.nh * self.dh)  # [seq_len, nh * dh]
        layer_output[f"attn_output"] = attn_output
        
        print(f"  Attention output: shape={attn_output.shape}, mean={attn_output.mean():.4f}")
        self._print_small_matrix("Attention output", attn_output)
        
        # 输出投影
        output = torch.mm(attn_output, self.w_o_proj.t())  # [seq_len, d]
        layer_output[f"attn_final"] = output
        
        print(f"  Output projection: shape={output.shape}, mean={output.mean():.4f}")
        self._print_small_matrix("Attention final output", output)
        
        # 残差连接
        output = output + hidden_states
        layer_output[f"attn_residual"] = output
        print(f"  After residual: shape={output.shape}, mean={output.mean():.4f}")
        
        return output

    def mlp_layer(self, hidden_states: torch.Tensor, layer_output: Dict) -> torch.Tensor:
        """MLP层计算"""
        print(f"\n  --- MLP Layer (Layer {self.layer_id}) ---")
        print(f"  Input shape: {hidden_states.shape}")
        
        # MLP归一化
        normed_input = self.rms_norm(hidden_states, self.w_mlp_norm, self.config.epsilon)
        layer_output[f"mlp_norm"] = normed_input
        print(f"  After MLP norm: shape={normed_input.shape}, mean={normed_input.mean():.4f}")
        self._print_small_matrix("MLP norm output", normed_input)
        
        # Gate和Up投影
        gate_output = torch.mm(normed_input, self.w_gate_proj.t())  # [seq_len, di]
        up_output = torch.mm(normed_input, self.w_up_proj.t())      # [seq_len, di]
        
        layer_output[f"gate_proj"] = gate_output
        layer_output[f"up_proj"] = up_output
        
        print(f"  Gate projection: shape={gate_output.shape}, mean={gate_output.mean():.4f}")
        print(f"  Up projection: shape={up_output.shape}, mean={up_output.mean():.4f}")
        
        # SwiGLU激活函数
        gate_activated = F.silu(gate_output)  # SiLU激活
        mlp_output = gate_activated * up_output
        layer_output[f"mlp_intermediate"] = mlp_output
        
        print(f"  After SwiGLU: shape={mlp_output.shape}, mean={mlp_output.mean():.4f}")
        self._print_small_matrix("MLP intermediate", mlp_output)
        
        # Down投影
        output = torch.mm(mlp_output, self.w_down_proj.t())  # [seq_len, d]
        layer_output[f"mlp_output"] = output
        
        print(f"  Down projection: shape={output.shape}, mean={output.mean():.4f}")
        self._print_small_matrix("MLP output", output)
        
        # 残差连接
        output = output + hidden_states  
        layer_output[f"mlp_residual"] = output
        print(f"  After residual: shape={output.shape}, mean={output.mean():.4f}")
        
        return output

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
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
    
    def _print_small_matrix(self, name: str, tensor: torch.Tensor, max_elements: int = 64):
        """打印小矩阵，便于在终端查看"""
        if tensor.numel() <= max_elements:
            print(f"  {name}:")
            if tensor.dim() == 1:
                print(f"    {tensor.detach().numpy()}")
            elif tensor.dim() == 2:
                for i, row in enumerate(tensor.detach().numpy()):
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
        self.input_embedding = torch.randn(config.dvoc, config.d) * 0.1
        self.output_norm = torch.ones(config.d) + torch.randn(config.d) * 0.01
        self.output_projection = torch.randn(config.d, config.dvoc) * 0.1
        
        print(f"\nModel initialized with {len(self.layers)} layers")

    def embed_tokens(self, token_ids: List[int]) -> torch.Tensor:
        """Token嵌入"""
        embeddings = []
        for token_id in token_ids:
            if token_id >= self.config.dvoc:
                token_id = token_id % self.config.dvoc  # 简单截断
            embeddings.append(self.input_embedding[token_id])
        return torch.stack(embeddings)

    def forward(self, token_ids: List[int]) -> Tuple[torch.Tensor, Dict]:
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
        final_norm = layer.rms_norm(hidden_states, self.output_norm, self.config.epsilon)
        print(f"\nFinal norm: shape={final_norm.shape}, mean={final_norm.mean():.4f}")
        layer._print_small_matrix("Final norm", final_norm)
        
        # 输出投影到词汇表
        logits = torch.mm(final_norm, self.output_projection.t())
        print(f"Final logits: shape={logits.shape}, mean={logits.mean():.4f}")
        layer._print_small_matrix("Final logits", logits)
        
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
    
    def analyze_results(self, logits: torch.Tensor, layer_outputs: Dict, test_tokens: List[int]):
        """分析测试结果"""
        print(f"\n{'='*80}")
        print(f"Results Analysis")
        print(f"{'='*80}")
        
        # 预测下一个token
        final_probs = F.softmax(logits[-1], dim=-1)  # 最后一个位置的概率
        predicted_token = torch.argmax(final_probs).item()
        
        print(f"\nFinal predictions:")
        print(f"  Input tokens: {test_tokens}")
        print(f"  Predicted next token: {predicted_token}")
        print(f"  Prediction confidence: {final_probs[predicted_token]:.4f}")
        print(f"  Top 3 predictions:")
        
        top_k_probs, top_k_indices = torch.topk(final_probs, 3)
        for i, (prob, idx) in enumerate(zip(top_k_probs, top_k_indices)):
            print(f"    {i+1}. Token {idx.item()}: {prob:.4f}")
        
        # 检查每层的数值稳定性
        print(f"\nNumerical Stability Check:")
        for layer_name, layer_data in layer_outputs.items():
            print(f"\n  {layer_name}:")
            for key, tensor in layer_data.items():
                if isinstance(tensor, torch.Tensor):
                    mean_val = tensor.mean().item()
                    std_val = tensor.std().item()
                    max_val = tensor.max().item()
                    min_val = tensor.min().item()
                    
                    # 检查是否有问题
                    issues = []
                    if abs(mean_val) > 10:
                        issues.append("large_mean")
                    if std_val > 10:
                        issues.append("large_variance")
                    if torch.isnan(tensor).any():
                        issues.append("NaN_values")
                    if torch.isinf(tensor).any():
                        issues.append("Inf_values")
                    
                    status = "❌ " + ",".join(issues) if issues else "✅ OK"
                    print(f"    {key}: mean={mean_val:.4f}, std={std_val:.4f}, min={min_val:.4f}, max={max_val:.4f} {status}")
        
        # 保存详细结果到JSON文件
        self.save_results_to_json(logits, layer_outputs, test_tokens)
    
    def save_results_to_json(self, logits: torch.Tensor, layer_outputs: Dict, test_tokens: List[int]):
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
            "final_logits": logits.detach().numpy().tolist(),
            "layer_statistics": {}
        }
        
        # 收集每层统计信息
        for layer_name, layer_data in layer_outputs.items():
            results["layer_statistics"][layer_name] = {}
            for key, tensor in layer_data.items():
                if isinstance(tensor, torch.Tensor):
                    results["layer_statistics"][layer_name][key] = {
                        "shape": list(tensor.shape),
                        "mean": tensor.mean().item(),
                        "std": tensor.std().item(),
                        "min": tensor.min().item(),
                        "max": tensor.max().item()
                    }
        
        # 保存到文件
        output_file = "/tmp/qw_test_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Detailed results saved to: {output_file}")

    def compare_with_expected(self, expected_file: Optional[str] = None):
        """与预期结果对比 (如果有的话)"""
        if expected_file and os.path.exists(expected_file):
            print(f"\nComparing with expected results from {expected_file}")
            # 这里可以实现与预期结果的对比逻辑
        else:
            print(f"\nNo expected results file provided for comparison")

def main():
    """主函数"""
    print("QW (Qwen3) Small Matrix Test")
    print("Based on InfiniCore-Infer-main/src/qw functions\n")
    
    # 创建测试器
    tester = QwenTester()
    
    # 运行基本测试
    success = tester.run_test()
    
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
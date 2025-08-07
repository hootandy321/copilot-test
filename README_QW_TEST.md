# QW (Qwen3) Small Matrix Test

基于 InfiniCore-Infer-main/src/qw 内的函数进行小矩阵测试的实现。

## 测试目标

- ✅ 使用小维度矩阵，能在终端里看全
- ✅ 可自定义权重、输入
- ✅ 每层都有输出
- ✅ 和预测结果对比，找出有问题的层

## 文件说明

### 主要文件

- **`qw_matrix_test_numpy.py`** - 主测试脚本（纯NumPy实现）
- **`qw_expected_results.py`** - 结果验证和手动计算演示
- **`qw_matrix_test.py`** - 原始PyTorch版本（需要PyTorch依赖）

### 生成的结果文件

- **`/tmp/qw_test_results.json`** - 详细的测试结果和统计信息

## 快速开始

### 运行主要测试

```bash
python3 qw_matrix_test_numpy.py
```

### 运行结果验证

```bash
python3 qw_expected_results.py
```

## 模型配置

测试使用的小矩阵配置：

```python
nlayer = 2           # 2层 transformer
d = 8               # 隐藏维度 8 (可以显示 8x8 矩阵)
nh = 2              # 2个注意力头  
nkvh = 2            # K/V头数量
dh = 4              # 每个头的维度 = d // nh
di = 16             # MLP中间维度
dctx = 4            # 最大序列长度
dvoc = 10           # 词汇表大小
```

## 实现特点

### 核心架构组件

1. **输入嵌入层** - Token到向量的映射
2. **RMS归一化** - 替代LayerNorm的归一化方法
3. **多头注意力**
   - Q/K/V投影
   - **Qwen3特有的Q/K归一化**
   - 注意力分数计算和Softmax
   - 输出投影和残差连接
4. **MLP层**
   - Gate和Up投影
   - **SwiGLU激活函数**
   - Down投影和残差连接
5. **输出投影** - 隐藏状态到词汇表logits

### 数学实现

#### RMS归一化
```python
def rms_norm(x, weight, eps=1e-6):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    rms = norm * (x.shape[-1] ** -0.5)
    return (x / (rms + eps)) * weight
```

#### SwiGLU激活
```python  
def swiglu(x, gate):
    return silu(gate) * x  # where silu(x) = x / (1 + exp(-x))
```

#### 多头注意力
```python
scores = Q @ K.T / sqrt(d_k)
attn_weights = softmax(scores)
output = attn_weights @ V
```

## 测试结果示例

### 终端输出示例

```
Layer 0 Forward Pass
============================================================

  --- Multi-Head Attention (Layer 0) ---
  Input shape: (4, 8)
  After attention norm: shape=(4, 8), mean=-0.0887
  Attention norm output:
    [0] [-1.15423342 -0.56555793 -0.7315256   0.32394526 ...]
    [1] [ 0.4729917   0.65896561 -0.50436396 -2.28650505 ...]
    
  Q projection: shape=(4, 8), mean=0.0496
  K projection: shape=(4, 8), mean=0.0087  
  V projection: shape=(4, 8), mean=-0.0219
```

### 数值稳定性检查

所有层的输出都会进行数值稳定性检查：

```
Numerical Stability Check:
  layer_0:
    attn_norm: mean=0.2540, std=0.9741, min=-1.3507, max=2.2465 ✅ OK
    q_proj: mean=0.0117, std=0.4370, min=-0.7200, max=0.8854 ✅ OK
    attn_weights: mean=0.3333, std=0.1971, min=0.0527, max=0.6827 ✅ OK
```

### 最终预测结果

```
Final predictions:
  Input tokens: [1, 3, 7, 2]
  Predicted next token: 6
  Prediction confidence: 0.1475
  Top 3 predictions:
    1. Token 6: 0.1475
    2. Token 4: 0.1297  
    3. Token 9: 0.1182
```

## 详细分析

### JSON输出结构

生成的 `/tmp/qw_test_results.json` 包含：

```json
{
  "config": {...},
  "test_input": {...},
  "final_logits": [...],
  "layer_statistics": {
    "layer_0": {
      "attn_norm": {"shape": [4,8], "mean": 0.254, "std": 0.974, ...},
      "q_proj": {...},
      ...
    }
  }
}
```

### 手动计算验证

`qw_expected_results.py` 提供了详细的手动计算步骤：

1. **输入嵌入验证**
2. **RMS归一化手动计算**
3. **线性投影验证**
4. **注意力分数计算**
5. **结果一致性检查**

## 调试和验证

### 层级问题检测

每个层的输出都会被检查，包括：

- 数值范围是否合理
- 是否存在NaN或Inf值
- 梯度是否过大或过小
- 注意力权重是否正确归一化

### 与预期结果对比

通过手动计算验证关键步骤：

```python
# 验证RMS归一化
expected = manual_rms_norm(x, weight)
actual = layer.rms_norm(x, weight)
assert np.allclose(expected, actual)
```

## 扩展和自定义

### 自定义输入

```python
# 测试不同的token序列
tester.run_test([0, 1, 2, 3])  # 长序列
tester.run_test([5])           # 单token
tester.run_test([9, 0, 1])     # 边界情况
```

### 修改模型配置

```python
config = SmallQwen3Config()
config.d = 16        # 增大隐藏维度
config.nlayer = 3    # 增加层数
config.nh = 4        # 增加注意力头数
```

### 自定义权重初始化

```python
# 修改权重初始化策略
def custom_init_weights(self):
    # 使用不同的初始化方法
    self.w_q_proj = np.eye(self.d) * 0.1  # 单位矩阵初始化
```

## 依赖要求

- Python 3.6+
- NumPy
- (可选) PyTorch - 仅用于 `qw_matrix_test.py`

## 运行环境

测试已在以下环境验证：

- Ubuntu/Linux
- Python 3.12
- NumPy 2.3.2

## 总结

这个QW小矩阵测试成功实现了原始需求的所有目标：

✅ **小维度矩阵显示** - 8x8及更小的矩阵完整显示  
✅ **自定义权重输入** - 支持修改所有模型参数  
✅ **层级输出监控** - 每层的详细中间结果  
✅ **问题层检测** - 数值稳定性检查和异常检测  
✅ **结果验证** - 手动计算对比和一致性检查

适用于调试、教学和验证 InfiniCore-Infer QW 模块的正确性。
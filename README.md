# Qwen3-1.7B Model Adaptation for InfiniCore

## 项目概述

本项目旨在将 Qwen3-1.7B 模型适配到 InfiniCore 跨平台统一编程工具集中。InfiniCore 为不同芯片平台的功能（包括计算、运行时、通信等）提供统一 C 语言接口。

## 任务目标

基于已有的 jiuge 模型适配实现，为 qwen3-1.7B 模型创建完整的适配层，包括：
1. C++ 核心推理引擎实现
2. Python 绑定接口
3. 模型权重处理和加载
4. 推理流程实现

## 项目结构

```
src/models/qwen3/
├── qwen3.cpp           # 核心推理实现
├── qwen3.py            # Python 接口层
├── qwen3_impl.hpp      # 实现头文件
├── qwen3_weight.hpp    # 权重处理头文件
└── CMakeLists.txt      # 构建配置
```

## 技术规格

### 模型架构特性
- **模型名称**: Qwen3-1.7B
- **隐藏层维度**: 5120
- **注意力头数**: 64 (需要确认 KV 头数)
- **中间层维度**: 25600  
- **词汇表大小**: 151936
- **最大序列长度**: 待确认
- **层数**: 待确认

### 核心计算流程【具体流程需要去阅读qwen3的原型文件，在./qwen3/内】
1. **输入嵌入**: (B,S,5120)
2. **Transformer 层** (重复 N 次):
   - RMS LayerNorm
   - Multi-Head Attention
   - 残差连接
   - RMS LayerNorm  
   - MLP (Gate + Up + SiLU + Down)
   - 残差连接
3. **输出层**:
   - RMS LayerNorm
   - Token Linear
   - ArgMax (B,S,151936)

### 注意力机制细节
- Q、K、V 线性变换: (B,S,64,128)
- Q、K 使用 RMS LayerNorm、Transpose、RoPE
- 注意力计算: MatMul → Divide → Mask → Softmax → MatMul(V)
- 输出: Reshape → O-Linear → (B,S,5120)

### MLP 模块细节
- Gate-Linear 和 Up-Linear 并行: (B,S,25600)
- Gate 使用 SiLU 激活函数
- Element-wise 乘法: Gate(SiLU) * Up
- Down-Linear: (B,S,5120)

## 实现要求

### 1. 核心推理引擎 (qwen3.cpp)

**参考**: `src/models/jiuge/jiuge.cpp`

**必须实现的函数**:
```cpp
// 设备资源管理
void createDeviceResource(DeviceResource *rsrc, const Qwen3Meta *meta,
                          const Qwen3Weights *weights, ...);
void releaseDeviceResource(DeviceResource &res);

// 批量推理
void inferDeviceBatch(const Qwen3Meta &meta, DeviceResource &rsrc, ...);

// C 接口
extern "C" {
    struct Qwen3Model* createQwen3Model(const Qwen3Meta *meta, ...);
    void destroyQwen3Model(struct Qwen3Model *model);
    void inferBatch(struct Qwen3Model *model, ...);
}
```

**关键差异点** (相对于 jiuge):
- 词汇表大小: 151936 vs jiuge 的大小
- 隐藏层维度: 5120
- 中间层维度: 25600
- 注意力机制中的 RMS LayerNorm 应用到 Q、K
- 确认是否使用 GQA (Grouped Query Attention)

### 2. 元数据结构 (qwen3_impl.hpp)

```cpp
struct Qwen3Meta {
    infiniDatatype_t dt_logits;
    uint32_t nlayer;        // 层数
    uint32_t d;             // 5120
    uint32_t nh;            // 64
    uint32_t nkvh;          // 待确认，可能与 nh 相同或更小
    uint32_t dh;            // d/nh = 80
    uint32_t di;            // 25600
    uint32_t dctx;          // 最大上下文长度
    uint32_t dvoc;          // 151936
    float epsilon;          // RMS norm epsilon
    float theta;            // RoPE theta
    uint32_t end_token;
    
    // 可能需要的缩放因子
    float scale_input;
    float scale_output;
    float scale_o;
    float scale_down;
};
```

### 3. 权重处理 (qwen3_weight.hpp)

**参考**: `src/models/jiuge/jiuge_weight.hpp`

需要实现的权重获取函数:
```cpp
std::shared_ptr<Tensor> getInEmbd(const Qwen3Meta *meta, const Qwen3Weights *weights);
std::shared_ptr<Tensor> getOutNorm(const Qwen3Meta *meta, const Qwen3Weights *weights);
std::shared_ptr<Tensor> getOutEmbd(const Qwen3Meta *meta, const Qwen3Weights *weights);
std::shared_ptr<Tensor> getAttnNorm(const Qwen3Meta *meta, const Qwen3Weights *weights, size_t layer);
std::shared_ptr<Tensor> getAttnQKV(const Qwen3Meta *meta, const Qwen3Weights *weights, size_t layer, int idev, int ndev);
// ... 其他权重获取函数
```

### 4. Python 接口 (qwen3.py)

**参考**: `src/models/jiuge/jiuge.py`

**必须实现的类**:

```python
class Qwen3WeightsNaming:
    """定义 Qwen3 模型的权重命名规则"""
    def input_embd(self): return "model.embed_tokens.weight"
    def output_norm(self): return "model.norm.weight"
    def output_embd(self): return "lm_head.weight"
    # ... 其他权重命名方法

class Qwen3MetaFromConfig(Qwen3MetaCStruct):
    """从配置文件创建元数据"""
    def __init__(self, config, dtype=torch.float16, max_tokens=None):
        # 解析 Qwen3 特定的配置参数
        
class Qwen3WeightsImpl(Qwen3WeightsCStruct):
    """权重实现，处理权重加载和格式转换"""
    
class Qwen3ForCausalLM:
    """主要的模型接口类"""
    def __init__(self, model_dir_path, device=DeviceType.DEVICE_TYPE_CPU, ndev=1):
    def batch_infer_one_round(self, tasks: List[InferTask]):
    def generate(self, input_content, max_steps, ...):
```

## 实现步骤

### Phase 1: 环境准备和分析
1. **分析 qwen3 Python 模型结构**
   - 确定确切的模型参数 (层数、注意力头配置等)
   - 理解权重命名规则
   - 确认与 jiuge 的具体差异

2. **准备开发环境**
   - 克隆 InfiniCore 仓库
   - 编译现有的 jiuge 模型作为基线
   - 准备 qwen3 模型文件

### Phase 2: 核心实现
1. **创建元数据结构**
   - 定义 `Qwen3Meta` 结构
   - 实现配置文件解析

2. **实现权重处理**
   - 创建 `qwen3_weight.hpp`
   - 实现权重获取和格式转换函数
   - 处理 qwen3 特有的权重布局

3. **核心推理引擎**
   - 基于 jiuge.cpp 创建 qwen3.cpp
   - 适配 qwen3 特有的计算流程
   - 实现注意力机制中的 RMS LayerNorm

### Phase 3: Python 绑定
1. **Python 接口实现**
   - 创建 qwen3.py
   - 实现权重命名类
   - 创建主要的模型接口

2. **集成测试**
   - 编写基本的推理测试
   - 验证输出正确性
   - 性能基准测试

### Phase 4: 优化和文档
1. **性能优化**
   - 内存使用优化
   - 计算效率优化

2. **文档完善**
   - API 文档
   - 使用示例
   - 性能基准

## 关键注意事项

### 1. 与 Jiuge 的主要差异
- **词汇表大小**: 151936 (需要适配输出层)
- **模型维度**: 5120 vs jiuge 的维度
- **注意力机制**: Q、K 使用 RMS LayerNorm (需要确认 jiuge 是否也有)
- **权重命名**: 可能使用不同的命名规则

### 2. 需要确认的技术细节
- **注意力头配置**: 是否使用 GQA？KV 头数是多少？
- **RoPE 参数**: theta 值和具体实现
- **激活函数**: MLP 中使用 SiLU，需要确认 InfiniCore 的支持
- **数据类型**: 权重和激活的精度配置

### 3. 开发规范
- 遵循 InfiniCore 的编码规范
- 使用相同的内存管理策略
- 确保跨平台兼容性
- 添加适当的错误处理

## 测试要求

### 1. 功能测试
```python
# 基本推理测试
model = Qwen3ForCausalLM(model_path, device_type, ndev=1)
output = model.generate("你好，世界", max_steps=50)
print(output)
```

### 2. 正确性验证
- 对比原 Python 实现的输出
- 验证数值精度
- 检查边界情况

### 3. 性能测试
- 推理延迟测试
- 内存使用测试
- 多设备扩展性测试

## 交付要求

1. **完整的源代码**
   - qwen3.cpp, qwen3.py 及相关头文件
   - CMakeLists.txt 构建配置
   - 与现有代码集成

2. **测试代码和结果**
   - 单元测试
   - 集成测试
   - 性能基准报告

3. **文档**
   - 详细的实现文档
   - API 使用指南
   - 与 jiuge 的差异说明

## 需要澄清的问题

在开始实现前，请确认以下信息：

1. **InfiniCore 仓库的具体访问地址**
2. **Qwen3 模型的确切配置参数** (层数、注意力头配置等)
3. **目标硬件平台** (CPU、GPU、NPU等)
4. **性能要求** (延迟、吞吐量目标)
5. **是否需要支持多设备分布式推理**
6. **特定的编译和依赖要求**

## 预期工作量

- **Phase 1**: 1-2 天 (环境准备和分析)
- **Phase 2**: 3-5 天 (核心实现)
- **Phase 3**: 2-3 天 (Python 绑定)
- **Phase 4**: 1-2 天 (优化和文档)

**总计**: 7-12 天，具体取决于模型复杂度和测试要求。

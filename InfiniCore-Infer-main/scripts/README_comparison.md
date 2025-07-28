# Qwen3 模型对比工具

本目录包含了用于对比适配后的Qwen3 C++模型（基于InfiniCore框架）与原始Qwen3 Python模型的推理计算结果和性能的工具。

## 功能特性

### 1. 端到端推理对比 (`qwen3_comparison_simple.py`)

这是主要的对比工具，提供：

- **文本生成对比**: 对比两个模型在相同输入下的输出文本
- **性能对比**: 测量两个模型的推理时间和性能提升
- **准确度评估**: 使用文本相似度评估输出质量
- **批量测试**: 支持多个测试用例的批量对比
- **详细报告**: 生成JSON格式的详细对比报告

### 2. 深度分析对比 (`compare_qwen3_models.py`)

提供更深入的模型内部对比（开发中）：

- **中间层结果对比**: 比较每个transformer层的输出
- **注意力权重分析**: 对比注意力机制的计算结果
- **激活值分析**: 分析各层激活值的差异
- **数值精度评估**: 使用MSE、余弦相似度等指标评估数值精度

## 使用方法

### 基础对比

```bash
cd InfiniCore-Infer-main/scripts

# 基础CPU对比
python qwen3_comparison_simple.py /path/to/qwen3-model

# 指定设备类型
python qwen3_comparison_simple.py /path/to/qwen3-model --device nvidia

# 自定义测试输入
python qwen3_comparison_simple.py /path/to/qwen3-model \
    --test-inputs "Hello world" "山东最高的山是什么？" "解释人工智能"

# 生成详细报告
python qwen3_comparison_simple.py /path/to/qwen3-model \
    --output comparison_report.json \
    --max-tokens 100
```

### 深度分析对比

```bash
# 运行深度分析（需要Python transformers库中的Qwen3实现）
python compare_qwen3_models.py /path/to/qwen3-model \
    --max-layers 10 \
    --output detailed_analysis.json
```

## 参数说明

### qwen3_comparison_simple.py

- `model_path`: Qwen3模型目录路径
- `--device`: 设备类型 (cpu, nvidia, cambricon, ascend, metax, moore, iluvatar)
- `--max-tokens`: 每个测试生成的最大token数 (默认: 50)
- `--output`: 输出JSON报告文件路径
- `--test-inputs`: 自定义测试输入列表

### compare_qwen3_models.py

- `model_path`: Qwen3模型目录路径
- `--device`: 设备类型
- `--max-layers`: 要详细对比的最大层数 (默认: 5)
- `--output`: 输出详细分析报告文件路径
- `--test-inputs`: 测试输入列表

## 输出报告格式

### 简单对比报告 (JSON)

```json
{
  "summary": {
    "total_tests": 5,
    "successful_tests": 5,
    "failed_tests": 0,
    "avg_cpp_time": 0.123,
    "avg_python_time": 0.456,
    "avg_similarity": 0.85
  },
  "results": [
    {
      "test_input": "Hello",
      "cpp_output": "Hello! How can I help you today?",
      "python_output": "Hello! How may I assist you?",
      "cpp_time": 0.120,
      "python_time": 0.450,
      "text_similarity": 0.82,
      "success": true,
      "error_msg": ""
    }
  ]
}
```

### 控制台输出示例

```
================================================================================
QWEN3 MODEL COMPARISON
================================================================================
Model path: /path/to/qwen3-model
Device: DeviceType.DEVICE_TYPE_CPU
Test cases: 5
Max tokens per test: 50

[1/5]
Testing: 'Hello'
------------------------------------------------------------
Running C++ model...
C++ output: 'Hello! How can I help you today?' (0.120s)
Running Python model...
Python output: 'Hello! How may I assist you?' (0.450s)
Text similarity: 0.820

================================================================================
COMPARISON SUMMARY
================================================================================
Total tests: 5
Successful: 5
Failed: 0

Performance:
  Average C++ time: 0.123s
  Average Python time: 0.456s
  C++ speedup: 3.71x

Accuracy:
  Average text similarity: 0.850
```

## 环境要求

### C++ 模型要求

- 已编译的InfiniCore-Infer库
- 正确配置的INFINI_ROOT环境变量
- Qwen3模型权重文件

### Python 模型要求（可选）

- PyTorch
- transformers库
- 原始Qwen3模型实现文件

如果没有Python模型，工具会跳过Python模型对比，只显示C++模型的性能信息。

## 故障排除

### 常见问题

1. **"undefined symbol: createQwen3Model"**
   - 确保已重新编译InfiniCore库: `srun xmake && xmake install`

2. **"Python model not available"**
   - 检查qwen3目录是否包含modeling_qwen3.py
   - 确保transformers库已安装

3. **模型加载失败**
   - 检查模型路径是否正确
   - 确认模型文件完整性
   - 检查权限设置

### 环境设置

```bash
# 设置InfiniCore环境
export INFINI_ROOT=~/.infini

# 编译模型（如果需要）
cd InfiniCore-Infer-main
srun xmake && xmake install
```

## 贡献

如需添加新的对比指标或功能，请：

1. 在相应的Python脚本中添加新的度量函数
2. 更新ComparisonResult数据结构
3. 修改报告生成逻辑
4. 更新本README文档

## 注意事项

- 对比结果可能因模型量化、数值精度等因素存在差异
- C++模型当前主要实现基础功能，某些高级特性可能与Python模型有差异
- 建议在相同硬件环境下进行性能对比以获得准确结果
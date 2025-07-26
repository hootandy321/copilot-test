# Qwen3 模型对比工具快速使用指南

## 快速开始

### 1. 环境准备

```bash
# 设置环境变量
export INFINI_ROOT=~/.infini

# 确保C++模型已编译
cd InfiniCore-Infer-main
srun xmake && xmake install
```

### 2. 运行演示

```bash
cd InfiniCore-Infer-main/scripts

# 运行完整演示（推荐）
python demo_qwen3_comparison.py /path/to/qwen3-model

# 或直接运行对比工具
python qwen3_comparison_simple.py /path/to/qwen3-model
```

## 详细使用方法

### 基础对比

```bash
# CPU设备上的基础对比
python qwen3_comparison_simple.py /home/shared/models/Qwen3-1.7B

# 指定其他设备
python qwen3_comparison_simple.py /home/shared/models/Qwen3-1.7B --device nvidia

# 自定义测试输入
python qwen3_comparison_simple.py /home/shared/models/Qwen3-1.7B \
    --test-inputs "你好" "山东最高的山是什么？" "请介绍人工智能"

# 生成更长的输出并保存报告
python qwen3_comparison_simple.py /home/shared/models/Qwen3-1.7B \
    --max-tokens 100 \
    --output my_comparison_report.json
```

### 支持的设备类型

- `cpu`: CPU设备（默认）
- `nvidia`: NVIDIA GPU
- `cambricon`: 寒武纪MLU
- `ascend`: 华为昇腾
- `metax`: 沐曦GPU
- `moore`: 摩尔线程GPU
- `iluvatar`: 天数智芯GPU

## 输出解读

### 控制台输出示例

```
================================================================================
QWEN3 MODEL COMPARISON
================================================================================
Model path: /home/shared/models/Qwen3-1.7B
Device: DeviceType.DEVICE_TYPE_CPU
Test cases: 3
Max tokens per test: 50

[1/3]
Testing: '你好'
------------------------------------------------------------
Running C++ model...
C++ output: '你好！有什么我可以帮助你的吗？' (0.156s)
Running Python model...
Python output: '你好！我是一个人工智能助手，有什么可以帮到你的吗？' (0.423s)
Text similarity: 0.750

================================================================================
COMPARISON SUMMARY  
================================================================================
Total tests: 3
Successful: 3
Failed: 0

Performance:
  Average C++ time: 0.156s
  Average Python time: 0.423s
  C++ speedup: 2.71x

Accuracy:
  Average text similarity: 0.750
```

### 关键指标说明

- **C++ speedup**: C++模型相对于Python模型的加速比
- **Text similarity**: 文本相似度（0-1，越接近1越相似）
- **Success rate**: 成功对比的测试用例比例

### JSON报告结构

生成的JSON报告包含：

```json
{
  "summary": {
    "total_tests": 3,
    "successful_tests": 3, 
    "avg_cpp_time": 0.156,
    "avg_python_time": 0.423,
    "avg_similarity": 0.750
  },
  "results": [
    {
      "test_input": "你好",
      "cpp_output": "你好！有什么我可以帮助你的吗？",
      "python_output": "你好！我是一个人工智能助手，有什么可以帮到你的吗？",
      "cpp_time": 0.156,
      "python_time": 0.423,
      "text_similarity": 0.750,
      "success": true
    }
  ]
}
```

## 故障排除

### 常见错误

1. **模型加载失败**
   ```
   ✗ Failed to load C++ model: undefined symbol: createQwen3Model
   ```
   **解决方案**: 重新编译InfiniCore库
   ```bash
   cd InfiniCore-Infer-main
   srun xmake clean
   srun xmake && xmake install
   ```

2. **Python模型不可用**
   ```
   ✗ Python model not available
   ```
   **说明**: 这是正常情况，工具会继续运行只测试C++模型性能。

3. **权限错误**
   ```
   PermissionError: [Errno 13] Permission denied
   ```
   **解决方案**: 检查模型文件权限和INFINI_ROOT目录权限

### 环境检查

```bash
# 检查环境变量
echo $INFINI_ROOT

# 检查库文件
ls $INFINI_ROOT/lib/libinfinicore_infer.so

# 测试简单加载
python -c "from qwen3 import Qwen3ForCausalLM"
```

## 进阶使用

### 批量测试脚本

创建自定义测试脚本：

```python
#!/usr/bin/env python3
import subprocess
import sys

test_cases = [
    "数学问题测试",
    "常识问答测试", 
    "代码生成测试",
    "翻译任务测试"
]

for i, test_name in enumerate(test_cases):
    print(f"运行测试 {i+1}: {test_name}")
    # 运行对比工具
    subprocess.run([
        sys.executable, "qwen3_comparison_simple.py",
        "/path/to/model",
        "--output", f"test_{i+1}_report.json"
    ])
```

### 自定义指标

可以修改 `qwen3_comparison_simple.py` 中的 `calculate_text_similarity` 函数来实现自定义的相似度计算，例如：

- BLEU分数
- ROUGE分数  
- 语义相似度（基于embedding）
- 特定任务的评估指标
#!/usr/bin/env python3
"""
Qwen3 模型对比演示脚本

演示如何使用对比工具来评估适配后的Qwen3 C++模型与原始Python模型的差异。
"""

import os
import sys
import subprocess
from pathlib import Path


def run_basic_comparison(model_path: str):
    """运行基础对比演示"""
    print("=" * 60)
    print("运行基础Qwen3模型对比")
    print("=" * 60)
    
    # 准备测试用例
    test_inputs = [
        "你好",
        "山东最高的山是什么？", 
        "请简单介绍一下人工智能",
        "1+1等于几？",
        "What is the capital of China?"
    ]
    
    # 构建命令
    cmd = [
        sys.executable, "qwen3_comparison_simple.py",
        model_path,
        "--device", "cpu",
        "--max-tokens", "30",
        "--output", "demo_comparison_report.json",
        "--test-inputs"
    ] + test_inputs
    
    print(f"执行命令: {' '.join(cmd)}")
    print()
    
    try:
        # 运行对比
        result = subprocess.run(cmd, cwd=Path(__file__).parent, 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✓ 对比完成! 检查 demo_comparison_report.json 获取详细结果")
        else:
            print(f"\n✗ 对比失败，返回码: {result.returncode}")
            
    except Exception as e:
        print(f"运行对比时发生错误: {e}")


def run_performance_test(model_path: str):
    """运行性能测试演示"""
    print("\n" + "=" * 60)
    print("运行性能测试")
    print("=" * 60)
    
    # 性能测试用例 - 较长的输入来测试性能差异
    perf_inputs = [
        "请详细解释什么是深度学习，它与传统机器学习有什么区别？",
        "Please explain the concept of artificial intelligence and its applications in modern technology.",
        "描述一下中国的地理特征，包括主要山脉、河流和气候特点。"
    ]
    
    cmd = [
        sys.executable, "qwen3_comparison_simple.py",
        model_path,
        "--device", "cpu", 
        "--max-tokens", "100",
        "--output", "demo_performance_report.json",
        "--test-inputs"
    ] + perf_inputs
    
    print(f"执行命令: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent,
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✓ 性能测试完成! 检查 demo_performance_report.json 获取详细结果")
        else:
            print(f"\n✗ 性能测试失败，返回码: {result.returncode}")
            
    except Exception as e:
        print(f"运行性能测试时发生错误: {e}")


def check_environment():
    """检查运行环境"""
    print("检查运行环境...")
    
    # 检查必要文件
    script_dir = Path(__file__).parent
    required_files = [
        "qwen3_comparison_simple.py",
        "qwen3.py",
        "libinfinicore_infer.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not (script_dir / file).exists():
            missing_files.append(file)
            
    if missing_files:
        print(f"✗ 缺少必要文件: {missing_files}")
        return False
        
    # 检查环境变量
    if not os.environ.get("INFINI_ROOT"):
        print("⚠ 警告: INFINI_ROOT 环境变量未设置")
        print("  建议执行: export INFINI_ROOT=~/.infini")
        
    print("✓ 环境检查通过")
    return True


def main():
    if len(sys.argv) != 2:
        print("用法: python demo_qwen3_comparison.py <qwen3_model_path>")
        print()
        print("示例:")
        print("  python demo_qwen3_comparison.py /home/shared/models/Qwen3-1.7B")
        sys.exit(1)
        
    model_path = sys.argv[1]
    
    # 检查模型路径
    if not Path(model_path).exists():
        print(f"错误: 模型路径不存在: {model_path}")
        sys.exit(1)
        
    print("Qwen3 模型对比演示")
    print("=" * 60)
    print(f"模型路径: {model_path}")
    print()
    
    # 检查环境
    if not check_environment():
        sys.exit(1)
        
    try:
        # 运行基础对比
        run_basic_comparison(model_path)
        
        # 运行性能测试
        run_performance_test(model_path)
        
        print("\n" + "=" * 60)
        print("演示完成!")
        print("=" * 60)
        print("生成的文件:")
        print("  - demo_comparison_report.json: 基础对比结果")
        print("  - demo_performance_report.json: 性能测试结果")
        print()
        print("您可以查看这些JSON文件来分析详细的对比结果。")
        
    except KeyboardInterrupt:
        print("\n用户中断了演示")
    except Exception as e:
        print(f"\n演示过程中发生错误: {e}")


if __name__ == "__main__":
    main()
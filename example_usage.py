#!/usr/bin/env python3
"""
Example usage of the Qwen3 debug comparison system
This script demonstrates how to use the system to identify computation differences.
"""

import sys
import subprocess
from pathlib import Path

def run_example_comparison():
    """Run a simple example comparison"""
    print("🔬 Qwen3 Debug Comparison Example")
    print("=" * 50)
    
    # Check if we have the required files
    required_files = [
        "debug_qwen3_reference.py",
        "debug_compare.py", 
        "test_debug_system.py",
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
    
    if missing:
        print(f"❌ Missing required files: {', '.join(missing)}")
        return False
    
    # Run validation tests first
    print("Step 1: Running validation tests...")
    result = subprocess.run([sys.executable, "test_debug_system.py"], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ Validation tests failed:")
        print(result.stdout)
        print(result.stderr)
        return False
    
    print("✅ Validation tests passed")
    
    # Show usage help
    print("\nStep 2: Usage examples")
    print("\nTo run a comparison with a real Qwen3 model:")
    print("  python debug_compare.py /path/to/qwen3/model")
    print("\nWith custom input:")
    print("  python debug_compare.py /path/to/qwen3/model --input 'Hello world'")
    print("\nTo skip parts of the comparison:")
    print("  python debug_compare.py /path/to/qwen3/model --skip-python  # Use existing Python files")
    print("  python debug_compare.py /path/to/qwen3/model --skip-cpp     # Use existing C++ files")
    
    # Demonstrate Python reference (without real model)
    print("\nStep 3: Testing Python reference components...")
    try:
        from debug_qwen3_reference import DebugQwen3RMSNorm, save_tensor_debug
        import torch
        
        # Test RMSNorm
        norm = DebugQwen3RMSNorm(hidden_size=64)
        test_input = torch.randn(1, 5, 64)
        output = norm(test_input)
        print(f"✅ RMSNorm test: input {test_input.shape} -> output {output.shape}")
        
        # Test tensor saving
        save_tensor_debug(test_input, "example_tensor", 0)
        if Path("py_layer_0_example_tensor.txt").exists():
            print("✅ Tensor saving works")
            Path("py_layer_0_example_tensor.txt").unlink()  # Cleanup
        
    except Exception as e:
        print(f"⚠ Python reference test failed: {e}")
    
    # Demonstrate comparison functionality
    print("\nStep 4: Testing comparison functions...")
    try:
        from debug_compare import compare_tensors, load_tensor_from_debug_file
        import numpy as np
        
        # Create test tensors
        tensor1 = np.random.randn(3, 4).astype(np.float32)
        tensor2 = tensor1 + np.random.randn(3, 4).astype(np.float32) * 0.001  # Small diff
        
        stats = compare_tensors(tensor1, tensor2, "example_comparison")
        print(f"✅ Tensor comparison: max_abs_diff={stats['max_abs_diff']:.2e}, "
              f"cos_sim={stats['cosine_similarity']:.6f}")
        
    except Exception as e:
        print(f"⚠ Comparison test failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Example completed successfully!")
    print("\nNext steps:")
    print("1. Obtain a Qwen3 model (huggingface format)")
    print("2. Build the C++ library with debug support")
    print("3. Run: python debug_compare.py /path/to/model")
    print("4. Analyze the comparison results")
    
    return True

def show_system_requirements():
    """Show system requirements and setup instructions"""
    print("📋 System Requirements")
    print("=" * 30)
    print("\nPython Dependencies:")
    print("  pip install numpy torch transformers safetensors")
    
    print("\nRequired Files:")
    files = [
        "debug_qwen3_reference.py - Python reference implementation",
        "debug_compare.py - Main comparison script", 
        "test_debug_system.py - Validation tests",
        "NewInfiniCore-Infer-main/src/models/qw/qwen3.cpp - Modified C++ implementation",
        "NewInfiniCore-Infer-main/include/infinicore_infer/models/qwen3.h - C++ headers",
    ]
    for f in files:
        print(f"  ✓ {f}")
    
    print("\nC++ Build Requirements:")
    print("  - InfiniCore library and headers")
    print("  - xmake build system")
    print("  - CUDA toolkit (for GPU support)")
    
    print("\nModel Requirements:")
    print("  - Qwen3 model in HuggingFace format")
    print("  - Compatible with modeling_qwen3.py architecture")
    print("  - Supports separate Q/K/V projections and Q/K normalization")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--requirements":
        show_system_requirements()
    else:
        run_example_comparison()
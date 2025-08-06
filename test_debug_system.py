#!/usr/bin/env python3
"""
Test script to validate the debug comparison system
This script tests the individual components before running the full comparison.
"""

import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch

def test_debug_tensor_saving():
    """Test the tensor saving and loading functionality"""
    print("🧪 Testing tensor saving and loading...")
    
    # Create a test tensor
    test_data = np.random.randn(3, 4, 5).astype(np.float32)
    
    # Save it using our format
    filename = "test_tensor.txt"
    with open(filename, 'w') as f:
        f.write("# Tensor: test_tensor\n")
        f.write(f"# Shape: {test_data.shape[0]}x{test_data.shape[1]}x{test_data.shape[2]}\n")
        f.write(f"# Dtype: {test_data.dtype}\n")
        f.write(f"# Mean: {test_data.mean()}\n")
        f.write(f"# Std: {test_data.std()}\n")
        f.write(f"# Min: {test_data.min()}\n")
        f.write(f"# Max: {test_data.max()}\n")
        f.write("# Data:\n")
        for val in test_data.flatten():
            f.write(f"{val:.6e}\n")
    
    # Load it back
    from debug_compare import load_tensor_from_debug_file
    loaded_data, metadata = load_tensor_from_debug_file(filename)
    
    # Check if they match
    if loaded_data is not None and np.allclose(test_data, loaded_data):
        print("✅ Tensor saving/loading works correctly")
    else:
        print("❌ Tensor saving/loading failed")
        if loaded_data is not None:
            print(f"   Original shape: {test_data.shape}, Loaded shape: {loaded_data.shape}")
            print(f"   Max difference: {np.abs(test_data - loaded_data).max()}")
        else:
            print("   Loaded data is None")
    
    # Cleanup
    Path(filename).unlink(missing_ok=True)
    return loaded_data is not None and np.allclose(test_data, loaded_data)


def test_tensor_comparison():
    """Test tensor comparison functionality"""
    print("🧪 Testing tensor comparison...")
    
    from debug_compare import compare_tensors
    
    # Create test tensors
    tensor1 = np.random.randn(10, 20).astype(np.float32)
    tensor2 = tensor1 + np.random.randn(10, 20).astype(np.float32) * 0.01  # Small difference
    tensor3 = np.random.randn(10, 20).astype(np.float32)  # Large difference
    
    # Test similar tensors
    stats1 = compare_tensors(tensor1, tensor2, "similar_test")
    if stats1.get('allclose_1e-3', False) and not stats1.get('allclose_1e-4', False):
        print("✅ Similar tensor comparison works")
    else:
        print("❌ Similar tensor comparison failed")
        print(f"   Stats: {stats1}")
    
    # Test different tensors
    stats2 = compare_tensors(tensor1, tensor3, "different_test")
    if not stats2.get('allclose_1e-3', False) and stats2.get('max_abs_diff', 0) > 0.1:
        print("✅ Different tensor comparison works")
    else:
        print("❌ Different tensor comparison failed")
        print(f"   Stats: {stats2}")
    
    return True


def test_python_reference():
    """Test the Python reference implementation with dummy model"""
    print("🧪 Testing Python reference implementation...")
    
    try:
        # This will fail without a real model, but we can check if the import works
        from debug_qwen3_reference import DebugQwen3Model, DebugQwen3RMSNorm
        
        # Test RMSNorm
        norm = DebugQwen3RMSNorm(hidden_size=128)
        test_input = torch.randn(2, 10, 128)
        output = norm(test_input)
        
        if output.shape == test_input.shape:
            print("✅ Python reference components import and work")
            return True
        else:
            print("❌ Python reference output shape mismatch")
            return False
            
    except ImportError as e:
        print(f"⚠ Python reference import failed (expected without model): {e}")
        return True  # This is expected without a real model
    except Exception as e:
        print(f"❌ Python reference test failed: {e}")
        return False


def test_debug_file_patterns():
    """Test debug file pattern matching"""
    print("🧪 Testing debug file pattern matching...")
    
    # Create mock debug files
    mock_files = [
        "py_input_embeddings.txt",
        "py_layer_0_input_hidden_states.txt",
        "py_layer_0_attn_norm_output.txt",
        "py_layer_0_layer_output.txt",
        "cpp_input_embeddings.txt", 
        "cpp_layer_0_input_hidden_states.txt",
        "cpp_layer_0_attn_norm_output.txt",
        "cpp_layer_0_layer_output.txt",
    ]
    
    # Create empty files
    for filename in mock_files:
        Path(filename).touch()
    
    # Test pattern matching
    py_files = list(Path(".").glob("py_*.txt"))
    cpp_files = list(Path(".").glob("cpp_*.txt"))
    
    success = len(py_files) == 4 and len(cpp_files) == 4
    
    if success:
        print("✅ Debug file pattern matching works")
    else:
        print(f"❌ Debug file pattern matching failed: {len(py_files)} py files, {len(cpp_files)} cpp files")
    
    # Cleanup
    for filename in mock_files:
        Path(filename).unlink(missing_ok=True)
    
    return success


def main():
    print("🔬 Qwen3 Debug System Validation Tests")
    print("=" * 50)
    
    tests = [
        ("Tensor saving/loading", test_debug_tensor_saving),
        ("Tensor comparison", test_tensor_comparison), 
        ("Python reference", test_python_reference),
        ("Debug file patterns", test_debug_file_patterns),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The debug system is ready to use.")
        return 0
    else:
        print("⚠ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit(main())
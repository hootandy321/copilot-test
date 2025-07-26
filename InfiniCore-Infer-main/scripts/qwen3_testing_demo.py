#!/usr/bin/env python3
"""
Qwen3 Testing Demo

This script demonstrates how to use the Qwen3 testing and verification tools
to validate the C++ implementation against Python reference implementations.
"""

import sys
import os
import subprocess
from pathlib import Path

def print_header(title):
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def run_command(cmd, description):
    print(f"\n🔨 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✓ Command completed successfully")
        else:
            print(f"✗ Command failed with return code {result.returncode}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("✗ Command timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"✗ Command failed: {e}")
        return False

def main():
    print_header("QWEN3 C++ IMPLEMENTATION TESTING DEMO")
    print("""
This demo will guide you through testing the Qwen3 C++ implementation
against Python reference implementations to validate computational accuracy.

The demo includes:
1. Simple model verification (C++ vs PyTorch reference)
2. Layer-by-layer comparison (for detailed debugging)
3. Performance benchmarking

Prerequisites:
- Qwen3-1.7B model files in a local directory
- C++ implementation compiled and installed
- Python dependencies installed
""")
    
    # Check for model path
    if len(sys.argv) < 2:
        print("\nUsage: python qwen3_testing_demo.py <model_path>")
        print("\nExample:")
        print("  python qwen3_testing_demo.py /home/shared/models/Qwen3-1.7B")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    if not os.path.exists(model_path):
        print(f"✗ Model path does not exist: {model_path}")
        sys.exit(1)
    
    print(f"Model path: {model_path}")
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    print_header("TEST 1: BASIC MODEL VERIFICATION")
    print("""
This test verifies that the simplified Qwen3 model can load and generate text.
It uses the updated qwen3.py that properly tries to use Qwen3 C++ API
with graceful fallback to jiuge API.
""")
    
    # Test 1: Basic model verification
    success1 = run_command([
        "python", str(script_dir / "qwen3.py"), model_path, "cpu"
    ], "Testing basic Qwen3 model loading and generation")
    
    print_header("TEST 2: C++ vs PYTORCH VERIFICATION")
    print("""
This test compares the C++ implementation against a PyTorch reference
implementation to validate computational accuracy. It tests:
- Token-level accuracy
- Text similarity
- Performance comparison
""")
    
    # Test 2: C++ vs PyTorch verification
    success2 = run_command([
        "python", str(script_dir / "qwen3_verification.py"), model_path,
        "--test-inputs", "Hello", "你好", "What is the capital", "山东最高的山"
    ], "Running C++ vs PyTorch verification")
    
    print_header("TEST 3: LAYER-BY-LAYER COMPARISON")
    print("""
This test performs detailed layer-by-layer comparison to identify
where computational differences occur. Useful for debugging
specific implementation issues.
""")
    
    # Test 3: Layer-by-layer comparison
    success3 = run_command([
        "python", str(script_dir / "qwen3_layer_comparison.py"), model_path,
        "--test-inputs", "Hello", "What is"
    ], "Running detailed layer-by-layer comparison")
    
    print_header("TEST 4: BUILD STATUS CHECK")
    print("""
Checking if the C++ qwen3 implementation compiles correctly.
This helps identify compilation issues that might affect testing.
""")
    
    # Test 4: Check build status
    infini_dir = script_dir.parent
    success4 = run_command([
        "bash", "-c", f"cd {infini_dir} && xmake"
    ], "Checking C++ build status")
    
    print_header("SUMMARY")
    
    tests = [
        ("Basic Model Loading", success1),
        ("C++ vs PyTorch Verification", success2),
        ("Layer-by-Layer Comparison", success3),
        ("C++ Build Status", success4),
    ]
    
    print("\nTest Results:")
    for test_name, success in tests:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    total_passed = sum(success for _, success in tests)
    print(f"\nOverall: {total_passed}/{len(tests)} tests passed")
    
    if total_passed == len(tests):
        print("\n🎉 All tests passed! The Qwen3 implementation appears to be working correctly.")
    else:
        print(f"\n⚠ {len(tests) - total_passed} test(s) failed. Review the output above for details.")
    
    print("\nGenerated files:")
    output_files = [
        "/tmp/qwen3_verification_results.png",
        "/tmp/qwen3_comparison_*.png",
        "/tmp/qwen3_comparison_report.json",
    ]
    
    for file_pattern in output_files:
        print(f"  {file_pattern}")
    
    print("\nNext steps:")
    if not success1:
        print("- Fix basic model loading issues first")
        print("- Check if qwen3 C++ API is compiled and available")
        print("- Verify model files are in correct format")
    
    if not success2:
        print("- Compare detailed error messages from verification test")
        print("- Check if computational differences are significant")
        print("- Review tokenizer configuration")
    
    if not success3:
        print("- Use layer comparison to identify specific computation differences")
        print("- Focus on layers with highest error rates")
        print("- Compare intermediate tensor values")
    
    if not success4:
        print("- Fix C++ compilation errors first")
        print("- Check missing dependencies or header files")
        print("- Review xmake configuration")
    
    print("\nFor detailed analysis, check the generated visualization files.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Mock demonstration of the Qwen3 debug comparison system
This script simulates what a real comparison would look like using synthetic data.
"""

import numpy as np
from pathlib import Path
import sys

def create_mock_debug_files():
    """Create mock debug files to simulate a real comparison"""
    print("🎭 Creating mock debug files...")
    
    # Simulate 2 layers + input/output
    layers = [-1, 0, 1, "output"]  # -1 for input, "output" for final
    tensor_names = [
        "input_embeddings",
        "input_hidden_states", 
        "attn_norm_output",
        "attn_q_proj_raw",
        "attn_k_proj_raw", 
        "attn_v_proj_raw",
        "attn_q_normed",
        "attn_k_normed",
        "attn_residual_output",
        "mlp_norm_output", 
        "mlp_gate_proj",
        "mlp_up_proj",
        "mlp_intermediate",
        "layer_output",
        "final_norm_output"
    ]
    
    # Model dimensions (small for demo)
    batch_size, seq_len, hidden_dim = 1, 3, 128
    num_heads, head_dim = 8, 16
    intermediate_dim = 512
    vocab_size = 1000
    
    files_created = []
    
    for layer in layers:
        layer_tensors = tensor_names.copy()
        
        # Skip some tensors for input/output layers
        if layer == -1:  # Input layer
            layer_tensors = ["input_embeddings"]
        elif layer == "output":  # Output layer  
            layer_tensors = ["final_norm_output"]
        
        for tensor_name in layer_tensors:
            # Determine tensor shape based on name
            if "embeddings" in tensor_name or "hidden_states" in tensor_name or "norm_output" in tensor_name:
                shape = (batch_size, seq_len, hidden_dim)
            elif "q_proj" in tensor_name or "q_normed" in tensor_name:
                shape = (batch_size, seq_len, num_heads * head_dim)
            elif "k_proj" in tensor_name or "k_normed" in tensor_name or "v_proj" in tensor_name:
                shape = (batch_size, seq_len, num_heads * head_dim)  # Simplified
            elif "gate_proj" in tensor_name or "up_proj" in tensor_name or "intermediate" in tensor_name:
                shape = (batch_size, seq_len, intermediate_dim)
            else:
                shape = (batch_size, seq_len, hidden_dim)  # Default
            
            # Create synthetic data
            base_data = np.random.randn(*shape).astype(np.float32) * 0.1
            
            # Add small differences between Python and C++ versions
            py_data = base_data
            cpp_data = base_data.copy()
            
            # Simulate different types of discrepancies based on layer
            if layer == 0 and "q_proj" in tensor_name:
                # Simulate a significant difference in layer 0 Q projection
                cpp_data += np.random.randn(*shape).astype(np.float32) * 0.05
            elif isinstance(layer, int) and layer > 0:
                # Propagated small differences in later layers
                cpp_data += np.random.randn(*shape).astype(np.float32) * 0.001
            else:
                # Very small differences for most tensors
                cpp_data += np.random.randn(*shape).astype(np.float32) * 0.0001
            
            # Save files
            for prefix, data in [("py", py_data), ("cpp", cpp_data)]:
                if layer == -1:
                    filename = f"{prefix}_{tensor_name}.txt"
                elif layer == "output":
                    filename = f"{prefix}_{tensor_name}.txt"
                else:
                    filename = f"{prefix}_layer_{layer}_{tensor_name}.txt"
                
                save_mock_tensor(data, filename, tensor_name)
                files_created.append(filename)
    
    print(f"✅ Created {len(files_created)} mock debug files")
    return files_created


def save_mock_tensor(data: np.ndarray, filename: str, tensor_name: str):
    """Save a tensor in the debug file format"""
    with open(filename, 'w') as f:
        f.write(f"# Tensor: {tensor_name}\n")
        f.write(f"# Shape: {'x'.join(map(str, data.shape))}\n")
        f.write(f"# Dtype: {data.dtype}\n")
        f.write(f"# Mean: {data.mean()}\n")
        f.write(f"# Std: {data.std()}\n")
        f.write(f"# Min: {data.min()}\n")
        f.write(f"# Max: {data.max()}\n")
        f.write("# Data:\n")
        
        # For demo, save first 100 elements 
        flat_data = data.flatten()
        for i, val in enumerate(flat_data[:100]):
            f.write(f"{val:.6e}\n")


def run_mock_comparison():
    """Run the comparison system on mock data"""
    print("🔬 Running mock comparison...")
    
    try:
        from debug_compare import compare_debug_outputs, print_comparison_summary
        
        # Run comparison
        comparisons = compare_debug_outputs()
        
        if not comparisons:
            print("❌ No comparisons found")
            return False
        
        # Print results
        print_comparison_summary(comparisons)
        return True
        
    except ImportError as e:
        print(f"❌ Could not import comparison functions: {e}")
        return False
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup_mock_files(files: list):
    """Clean up mock files"""
    print("\n🧹 Cleaning up mock files...")
    for file in files:
        Path(file).unlink(missing_ok=True)
    print("✅ Cleanup completed")


def main():
    print("🎭 Qwen3 Debug Comparison Mock Demo")
    print("=" * 50)
    print("This demo simulates what a real comparison looks like using synthetic data.")
    print("It demonstrates the workflow and expected output format.\n")
    
    # Create mock data
    files = create_mock_debug_files()
    
    try:
        # Run comparison
        success = run_mock_comparison()
        
        if success:
            print("\n🎉 Mock demonstration completed successfully!")
            print("\nThis shows what you would see with a real model comparison.")
            print("Key observations from this mock run:")
            print("- Most tensors show excellent/good agreement")
            print("- Layer 0 Q projection shows significant differences")
            print("- Later layers show propagated small differences")
            print("- The system correctly identifies the first problematic layer")
        else:
            print("\n❌ Mock demonstration failed")
            return 1
            
    finally:
        # Cleanup
        cleanup_mock_files(files)
    
    print("\n📚 To run with real data:")
    print("1. Obtain a Qwen3 model")
    print("2. Build C++ implementation with debug support") 
    print("3. Run: python debug_compare.py /path/to/model")
    
    return 0


if __name__ == "__main__":
    exit(main())
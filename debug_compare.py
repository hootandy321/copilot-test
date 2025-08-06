#!/usr/bin/env python3
"""
Debug comparison script for Qwen3 C++ vs Python implementations
This script compares the intermediate results layer-by-layer to identify where discrepancies occur.

Usage: python debug_compare.py <model_path> [input_text]
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
import subprocess
import tempfile
from typing import Dict, List, Tuple
import argparse

# Import our debug implementations  
from debug_qwen3_reference import run_debug_inference

def clear_debug_files():
    """Clear any existing debug output files"""
    for pattern in ["py_*.txt", "cpp_*.txt"]:
        for f in Path(".").glob(pattern):
            f.unlink(missing_ok=True)
    print("Cleared existing debug files")


def load_tensor_from_debug_file(filepath: str) -> Tuple[np.ndarray, Dict]:
    """Load tensor data from debug output file"""
    if not Path(filepath).exists():
        return None, {}
    
    metadata = {}
    data_lines = []
    reading_data = False
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "# Data:":
                reading_data = True
            elif line.startswith('# ') and not reading_data:
                # Parse metadata
                if ':' in line:
                    key, value = line[2:].split(':', 1)
                    metadata[key.strip()] = value.strip()
            elif reading_data and line and not line.startswith('# '):
                try:
                    data_lines.append(float(line))
                except ValueError:
                    continue
    
    if not data_lines:
        return None, metadata
    
    # Convert to numpy array
    data = np.array(data_lines, dtype=np.float32)
    
    # Try to parse shape and reshape
    if 'Shape' in metadata:
        shape_str = metadata['Shape']
        try:
            # Parse shape like "3x4x5" or "(3, 4, 5)"
            shape_str = shape_str.replace('(', '').replace(')', '').replace(',', 'x')
            dims = [int(d) for d in shape_str.split('x') if d.strip()]
            
            # Only reshape if we have the expected number of elements
            expected_size = np.prod(dims)
            if len(data) >= expected_size:
                data = data[:expected_size].reshape(dims)
            
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not reshape {filepath}: {e}")
            pass
    
    return data, metadata


def compare_tensors(py_tensor: np.ndarray, cpp_tensor: np.ndarray, 
                   tensor_name: str, layer: int = -1) -> Dict:
    """Compare two tensors and return comparison statistics"""
    if py_tensor is None or cpp_tensor is None:
        return {
            'error': f"One tensor is None (py: {py_tensor is not None}, cpp: {cpp_tensor is not None})"
        }
    
    # Ensure both are float32 for comparison
    if py_tensor.dtype != np.float32:
        py_tensor = py_tensor.astype(np.float32)
    if cpp_tensor.dtype != np.float32:
        cpp_tensor = cpp_tensor.astype(np.float32)
    
    # Check shapes
    if py_tensor.shape != cpp_tensor.shape:
        return {
            'error': f"Shape mismatch: py {py_tensor.shape} vs cpp {cpp_tensor.shape}"
        }
    
    # Flatten for easier comparison
    py_flat = py_tensor.flatten()
    cpp_flat = cpp_tensor.flatten()
    
    # Compute statistics
    abs_diff = np.abs(py_flat - cpp_flat)
    rel_diff = abs_diff / (np.abs(py_flat) + 1e-8)  # Avoid division by zero
    
    stats = {
        'layer': layer,
        'tensor_name': tensor_name,
        'shape': py_tensor.shape,
        'py_mean': float(py_flat.mean()),
        'cpp_mean': float(cpp_flat.mean()),
        'py_std': float(py_flat.std()),
        'cpp_std': float(cpp_flat.std()),
        'max_abs_diff': float(abs_diff.max()),
        'mean_abs_diff': float(abs_diff.mean()),
        'max_rel_diff': float(rel_diff.max()),
        'mean_rel_diff': float(rel_diff.mean()),
        'cosine_similarity': float(np.dot(py_flat, cpp_flat) / (np.linalg.norm(py_flat) * np.linalg.norm(cpp_flat) + 1e-8)),
        'allclose_1e-3': bool(np.allclose(py_flat, cpp_flat, rtol=1e-3, atol=1e-3)),
        'allclose_1e-4': bool(np.allclose(py_flat, cpp_flat, rtol=1e-4, atol=1e-4)),
        'allclose_1e-5': bool(np.allclose(py_flat, cpp_flat, rtol=1e-5, atol=1e-5)),
    }
    
    return stats


def run_cpp_debug_inference(model_path: str, input_text: str = None) -> bool:
    """Run the C++ inference with debug enabled"""
    print("Running C++ debug inference...")
    
    try:
        # Import the Qwen3 wrapper and enable debug mode
        sys.path.append(str(Path(__file__).parent / "NewInfiniCore-Infer-main" / "scripts"))
        from qwen3 import QwenForCausalLM
        
        # Create model instance
        model = QwenForCausalLM(model_path, device_type="cpu", ndev=1)
        
        # Enable debug mode via the C++ API
        try:
            # Try to access the debug function through the C library
            from libinfinicore_infer import setQwen3DebugMode
            setQwen3DebugMode(1)  # Enable debug mode
            print("✓ C++ debug mode enabled")
        except ImportError:
            print("⚠ Could not enable C++ debug mode - function not available")
            return False
        
        # Run simple inference
        if input_text is None:
            # Use the same fixed input as Python version
            test_tokens = [1, 15339, 1079]  # Simple token sequence for reproducibility
            print(f"Using fixed input tokens: {test_tokens}")
            
            from infer_task import Qwen3InferTask
            from qwen3 import Qwen3KVCache
            
            task = Qwen3InferTask(
                tokens=test_tokens,
                position=0,
                temperature=0.0,  # Deterministic
                topk=1,
                topp=1.0,
                end_tokens=[2],
                max_tokens=int(model.meta.dctx),
                task_id=0
            )
            task.bind_kvcache(Qwen3KVCache(model))
            
            # Run single inference step
            output = model.batch_infer_one_round([task])
            print(f"C++ output token: {output[0]}")
            
            # Cleanup
            task._kv_cache.drop()
            
        else:
            # Run text generation
            output_text, avg_time = model.generate_simple(input_text, max_steps=1)
            print(f"C++ output: '{output_text}'")
        
        # Disable debug mode
        setQwen3DebugMode(0)
        model.destroy_model_instance()
        
        return True
        
    except Exception as e:
        print(f"❌ C++ inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_debug_outputs() -> List[Dict]:
    """Compare Python and C++ debug outputs"""
    print("\nComparing debug outputs...")
    
    # Find all Python debug files
    py_files = list(Path(".").glob("py_*.txt"))
    py_files.sort()
    
    # Find all C++ debug files  
    cpp_files = list(Path(".").glob("cpp_*.txt"))
    cpp_files.sort()
    
    print(f"Found {len(py_files)} Python debug files and {len(cpp_files)} C++ debug files")
    
    if not py_files or not cpp_files:
        print("❌ Missing debug files - both Python and C++ runs must complete successfully")
        return []
    
    comparisons = []
    
    # Compare matching files
    for py_file in py_files:
        # Find corresponding C++ file
        cpp_file_name = py_file.name.replace("py_", "cpp_")
        cpp_file = Path(".") / cpp_file_name
        
        if not cpp_file.exists():
            print(f"⚠ Missing C++ counterpart for {py_file.name}")
            continue
        
        # Load tensors
        py_tensor, py_meta = load_tensor_from_debug_file(py_file)
        cpp_tensor, cpp_meta = load_tensor_from_debug_file(cpp_file)
        
        if py_tensor is None:
            print(f"⚠ Could not load Python tensor from {py_file.name}")
            continue
            
        if cpp_tensor is None:
            print(f"⚠ Could not load C++ tensor from {cpp_file.name}")
            continue
        
        # Extract layer and tensor name from filename
        parts = py_file.stem.split('_')
        if len(parts) >= 3 and parts[1] == "layer":
            layer = int(parts[2])
            tensor_name = '_'.join(parts[3:])
        else:
            layer = -1
            tensor_name = '_'.join(parts[1:])
        
        # Compare tensors
        stats = compare_tensors(py_tensor, cpp_tensor, tensor_name, layer)
        comparisons.append(stats)
    
    return comparisons


def print_comparison_summary(comparisons: List[Dict]):
    """Print a summary of tensor comparisons"""
    if not comparisons:
        print("No comparisons available")
        return
    
    print(f"\n{'='*80}")
    print("TENSOR COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    # Sort by layer, then by tensor name
    comparisons.sort(key=lambda x: (x.get('layer', -1), x.get('tensor_name', '')))
    
    first_major_diff_layer = None
    
    for i, comp in enumerate(comparisons):
        if 'error' in comp:
            print(f"❌ {comp.get('tensor_name', 'unknown')} (layer {comp.get('layer', -1)}): {comp['error']}")
            continue
        
        layer = comp.get('layer', -1)
        name = comp.get('tensor_name', 'unknown')
        shape = comp.get('shape', 'unknown')
        
        # Determine if this is a significant difference
        max_abs_diff = comp.get('max_abs_diff', float('inf'))
        max_rel_diff = comp.get('max_rel_diff', float('inf'))
        cosine_sim = comp.get('cosine_similarity', 0.0)
        allclose_1e3 = comp.get('allclose_1e-3', False)
        allclose_1e4 = comp.get('allclose_1e-4', False)
        
        # Status determination
        if allclose_1e4:
            status = "✅ EXCELLENT"
        elif allclose_1e3:
            status = "✅ GOOD"
        elif max_abs_diff < 1e-2 and cosine_sim > 0.99:
            status = "⚠ ACCEPTABLE"
        else:
            status = "❌ SIGNIFICANT DIFF"
            if first_major_diff_layer is None and layer >= 0:
                first_major_diff_layer = layer
        
        layer_str = f"L{layer}" if layer >= 0 else "Input"
        
        print(f"{status} {layer_str:>6} {name:<25} {str(shape):<15} "
              f"max_abs:{max_abs_diff:.2e} max_rel:{max_rel_diff:.2e} cos_sim:{cosine_sim:.6f}")
    
    print(f"\n{'='*80}")
    
    if first_major_diff_layer is not None:
        print(f"🔍 FIRST MAJOR DIFFERENCE DETECTED AT LAYER {first_major_diff_layer}")
        print(f"   This suggests the issue begins in layer {first_major_diff_layer}")
    else:
        print("🎉 ALL COMPARISONS LOOK REASONABLE")
    
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Compare Qwen3 C++ vs Python implementations layer-by-layer")
    parser.add_argument("model_path", help="Path to the Qwen3 model directory")
    parser.add_argument("--input", help="Input text for inference (default: use fixed tokens)", default=None)
    parser.add_argument("--skip-python", action="store_true", help="Skip Python reference run (use existing files)")
    parser.add_argument("--skip-cpp", action="store_true", help="Skip C++ run (use existing files)")
    
    args = parser.parse_args()
    
    model_path = args.model_path
    if not Path(model_path).exists():
        print(f"❌ Model path does not exist: {model_path}")
        return 1
    
    print(f"🔬 Qwen3 C++ vs Python Debug Comparison")
    print(f"   Model: {model_path}")
    print(f"   Input: {args.input if args.input else 'Fixed tokens [1, 15339, 1079]'}")
    print()
    
    # Clear previous debug files
    clear_debug_files()
    
    # Step 1: Run Python reference implementation
    if not args.skip_python:
        print("Step 1: Running Python reference implementation...")
        try:
            hidden_states = run_debug_inference(model_path, args.input)
            if hidden_states is not None:
                print(f"✅ Python reference completed. Final shape: {hidden_states.shape}")
            else:
                print("❌ Python reference failed")
                return 1
        except Exception as e:
            print(f"❌ Python reference failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print("Step 1: Skipping Python reference (using existing files)")
    
    # Step 2: Run C++ implementation
    if not args.skip_cpp:
        print("\nStep 2: Running C++ implementation...")
        if not run_cpp_debug_inference(model_path, args.input):
            print("❌ C++ inference failed")
            return 1
        print("✅ C++ inference completed")
    else:
        print("\nStep 2: Skipping C++ implementation (using existing files)")
    
    # Step 3: Compare outputs
    print("\nStep 3: Comparing outputs...")
    comparisons = compare_debug_outputs()
    
    if not comparisons:
        print("❌ No successful comparisons - check debug file generation")
        return 1
    
    # Step 4: Print summary
    print_comparison_summary(comparisons)
    
    return 0


if __name__ == "__main__":
    exit(main())
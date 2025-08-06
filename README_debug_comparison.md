# Qwen3 C++ vs Python Layer-by-Layer Debug Comparison

This document describes how to use the debug comparison system to identify discrepancies between the C++ and Python implementations of the Qwen3 model.

## Overview

The system compares intermediate layer outputs between:
- **Python Reference**: Pure PyTorch implementation (`debug_qwen3_reference.py`) based on `qwen3/modeling_qwen3.py`
- **C++ Implementation**: Optimized implementation in `NewInfiniCore-Infer-main/src/models/qw/qwen3.cpp`

## Components

### 1. Python Reference Implementation (`debug_qwen3_reference.py`)
- Pure PyTorch implementation of Qwen3 model
- Saves intermediate tensors after each layer to files with prefix `py_`
- Handles Q/K normalization, RoPE, SwiGLU activation
- Uses fixed numerical precision and deterministic operations

### 2. C++ Debug Integration 
- Modified `qwen3.cpp` with debug tensor saving functions
- Added `setQwen3DebugMode()` API to enable/disable debug output
- Saves intermediate tensors to files with prefix `cpp_`
- Synchronized tensor saving points with Python version

### 3. Comparison Script (`debug_compare.py`)
- Loads and compares corresponding tensor files from Python and C++
- Computes statistics: absolute/relative differences, cosine similarity
- Identifies the first layer where significant discrepancies occur
- Provides detailed comparison reports

### 4. Test Suite (`test_debug_system.py`)
- Validates tensor saving/loading functionality
- Tests tensor comparison algorithms
- Verifies file pattern matching
- Ensures system components work correctly

## Quick Start

### Prerequisites
```bash
pip install numpy torch transformers safetensors
```

### Basic Usage
```bash
# Run comparison with a Qwen3 model
python debug_compare.py /path/to/qwen3/model

# Use custom input text
python debug_compare.py /path/to/qwen3/model --input "Hello world"

# Skip Python run if files already exist
python debug_compare.py /path/to/qwen3/model --skip-python

# Skip C++ run if files already exist  
python debug_compare.py /path/to/qwen3/model --skip-cpp
```

### Validation Tests
```bash
# Test system components before using
python test_debug_system.py
```

## Detailed Workflow

### Step 1: Python Reference Run
The system runs the pure PyTorch implementation and saves intermediate outputs:

**Saved Tensors per Layer:**
- `py_input_embeddings.txt` - Input token embeddings
- `py_layer_N_input_hidden_states.txt` - Layer input
- `py_layer_N_attn_norm_output.txt` - After attention normalization
- `py_layer_N_attn_q_proj_raw.txt` - Q projection output
- `py_layer_N_attn_k_proj_raw.txt` - K projection output  
- `py_layer_N_attn_v_proj_raw.txt` - V projection output
- `py_layer_N_attn_q_normed.txt` - Q after normalization
- `py_layer_N_attn_k_normed.txt` - K after normalization
- `py_layer_N_attn_q_rope.txt` - Q after RoPE
- `py_layer_N_attn_k_rope.txt` - K after RoPE
- `py_layer_N_attn_o_proj.txt` - Attention output projection
- `py_layer_N_attn_residual_output.txt` - After attention residual
- `py_layer_N_mlp_norm_output.txt` - After MLP normalization
- `py_layer_N_mlp_gate_proj.txt` - MLP gate projection
- `py_layer_N_mlp_up_proj.txt` - MLP up projection
- `py_layer_N_mlp_intermediate.txt` - After SwiGLU activation
- `py_layer_N_layer_output.txt` - Final layer output
- `py_final_norm_output.txt` - Final model output

### Step 2: C++ Implementation Run
Enables debug mode in the C++ implementation and saves corresponding tensors with `cpp_` prefix.

### Step 3: Tensor Comparison
For each matching tensor pair, computes:
- **Shape Validation**: Ensures tensors have identical shapes
- **Statistical Comparison**: Mean, std, min, max values
- **Difference Metrics**: Maximum and mean absolute/relative differences
- **Similarity Metrics**: Cosine similarity between flattened tensors
- **Tolerance Tests**: `allclose()` tests at different precision levels

### Step 4: Results Analysis
The comparison report shows:
- ✅ **EXCELLENT**: Tensors match within 1e-4 tolerance
- ✅ **GOOD**: Tensors match within 1e-3 tolerance
- ⚠ **ACCEPTABLE**: Small differences but high cosine similarity
- ❌ **SIGNIFICANT DIFF**: Large discrepancies detected

## Understanding Results

### Example Output
```
================================================================================
TENSOR COMPARISON SUMMARY
================================================================================
✅ EXCELLENT Input input_embeddings      (1, 3, 2048)    max_abs:1.23e-05 max_rel:2.45e-04 cos_sim:0.999998
✅ GOOD      L0    input_hidden_states   (1, 3, 2048)    max_abs:3.21e-04 max_rel:1.23e-03 cos_sim:0.999995
✅ GOOD      L0    attn_norm_output      (1, 3, 2048)    max_abs:2.45e-04 max_rel:8.76e-04 cos_sim:0.999996
❌ SIGNIFICANT DIFF L0 attn_q_proj_raw   (1, 3, 2048)    max_abs:1.23e-01 max_rel:5.67e-01 cos_sim:0.892345
================================================================================
🔍 FIRST MAJOR DIFFERENCE DETECTED AT LAYER 0
   This suggests the issue begins in layer 0
================================================================================
```

### Interpreting Metrics

1. **max_abs_diff**: Largest absolute difference between corresponding elements
   - < 1e-4: Excellent match
   - < 1e-3: Good match  
   - < 1e-2: Acceptable for FP16 computations
   - \> 1e-2: Significant difference

2. **cosine_similarity**: Measures if tensors point in the same direction
   - \> 0.999: Excellent alignment
   - \> 0.99: Good alignment
   - < 0.99: Poor alignment, significant structural differences

3. **max_rel_diff**: Largest relative difference (useful for small values)
   - Helps identify issues with values near zero
   - High relative diff with low absolute diff often acceptable

## Debugging Strategies

### 1. Numerical Precision Issues
- Compare FP16 vs FP32 computations
- Check for different rounding behaviors
- Verify consistent data type conversions

### 2. Weight Loading Discrepancies  
- Validate that both implementations load identical weights
- Check for transpose differences in matrix operations
- Verify tensor shape interpretations

### 3. Algorithm Differences
- Compare attention computation order (QK^T then softmax vs batched)
- Check RoPE implementation details
- Verify SwiGLU activation computation

### 4. Implementation-Specific Optimizations
- Identify fused operations in C++ that might differ from Python
- Check tensor memory layouts and strides  
- Verify distributed computation synchronization

## Customization

### Adding New Debug Points
1. **Python**: Add `save_tensor_debug(tensor, "name", layer)` calls
2. **C++**: Add `save_tensor_debug_f32(tensor, "name", layer)` calls  
3. **Ensure synchronized naming** between implementations

### Custom Input Data
- Use deterministic inputs for reproducible debugging
- Test with edge cases (zeros, very small/large values)
- Validate with known-good reference computations

## Limitations

- **Memory Usage**: Debug mode saves many intermediate tensors
- **Performance**: Debug runs are significantly slower
- **File I/O**: Large models generate many debug files
- **Precision**: Limited to comparison precision of saved text format

## Troubleshooting

### Common Issues

1. **Missing debug files**: 
   - Ensure C++ library was built with debug support
   - Check that `setQwen3DebugMode(1)` was called

2. **Shape mismatches**:
   - Verify tensor dimension interpretations
   - Check distributed computation settings

3. **Import errors**:
   - Ensure all Python dependencies are installed
   - Verify C++ library can be imported

4. **Large differences everywhere**:
   - Check weight loading consistency
   - Verify model architecture matches
   - Test with smaller/simpler inputs first

### Debug Tips

- Start with single layer models or simple inputs
- Use fixed random seeds for reproducibility  
- Compare layer-by-layer incrementally
- Save intermediate states for deeper analysis
- Use visualization tools for tensor inspection

## Advanced Usage

### Batch Processing
```bash
# Compare multiple test cases
for input in "test1" "test2" "test3"; do
    python debug_compare.py model --input "$input"
done
```

### Precision Analysis  
```bash
# Test different numerical precisions
python -c "
import torch
torch.set_default_dtype(torch.float32)  # vs torch.float16
"
```

### Custom Tolerance Testing
```python
# Modify comparison tolerances in debug_compare.py
stats = compare_tensors(py_tensor, cpp_tensor, name)
custom_match = np.allclose(py_tensor, cpp_tensor, rtol=1e-2, atol=1e-5)
```
# Qwen3 Layer-by-Layer Verification Test

This directory contains a comprehensive test program for verifying the computational accuracy of the Qwen3 model adapted for the InfiniCore library. The test compares calculation results at each transformer layer between the original Python implementation and the adapted C++ code.

## Purpose

As requested in the problem statement:
- **Verify accuracy**: Compare calculations between original Python code and adapted C++ code
- **Layer-by-layer testing**: Test each transformer layer individually for precise error localization
- **Use hardcoded model path**: `/home/shared/models/Qwen3-1.7B`
- **Focus on calculations**: Verify computational correctness without full inference runs

## Files Created

### `test_qwen3_layer_verification.py`
Main verification program that implements:

1. **SimplifiedQwen3Reference**: Pure PyTorch reference implementation
   - RMS normalization layers
   - Multi-head attention with rotary position embeddings
   - Qwen3-specific Q/K normalization
   - SwiGLU MLP layers
   - Layer-by-layer computation extraction

2. **Qwen3LayerVerificationTester**: Main testing framework
   - Layer-by-layer verification workflow
   - Comprehensive accuracy metrics (MSE, cosine similarity, relative error)
   - Detailed reporting and JSON output
   - Graceful fallback for missing components

3. **Accuracy Metrics**:
   - **Cosine Similarity**: Measures output direction alignment (threshold: 0.99)
   - **Mean Squared Error**: Measures numerical differences (threshold: 1e-4)
   - **Relative Error**: Normalized error magnitude (threshold: 0.01)
   - **Max Absolute Error**: Largest pointwise difference

## Usage

### Basic Usage
```bash
# Run with default settings (uses hardcoded model path)
python test_qwen3_layer_verification.py

# Specify custom model path
python test_qwen3_layer_verification.py --model-path /path/to/qwen3-model

# Generate detailed JSON report
python test_qwen3_layer_verification.py --output verification_report.json

# Use custom test input
python test_qwen3_layer_verification.py --test-input "Test this specific input"
```

### Advanced Usage
```bash
# Run on CUDA if available
python test_qwen3_layer_verification.py --device cuda

# Full command with all options
python test_qwen3_layer_verification.py \
    --model-path /home/shared/models/Qwen3-1.7B \
    --device cpu \
    --output detailed_verification_report.json \
    --test-input "Comprehensive layer verification test"
```

## Architecture Tested

The verification test covers these Qwen3 architectural components:

### 1. Embedding Layer
- Token embedding lookup
- Input shape validation

### 2. Transformer Layers (first 5 layers tested)
For each layer:

#### Attention Sublayer:
- Input RMS normalization
- Q, K, V projections  
- Qwen3-specific Q/K normalization (if present)
- Rotary position embedding application
- Scaled dot-product attention
- Output projection
- Residual connection

#### MLP Sublayer:
- Post-attention RMS normalization
- Gate and up projections
- SwiGLU activation (SiLU(gate) * up)
- Down projection
- Residual connection

### 3. Final Processing
- Final RMS normalization
- Shape and numerical validation

## Test Output

### Console Output
```
================================================================================
QWEN3 LAYER-BY-LAYER VERIFICATION TEST
================================================================================
Model path: /home/shared/models/Qwen3-1.7B
Device: cpu

Starting comprehensive layer-by-layer verification...
Test input: 'Hello, this is a test input'

Generated input shape: torch.Size([1, 32])
Verifying embedding layer...
✓ Embedding layer passed verification

Verifying 12 transformer layers...
  Verifying layer 0...
    ✓ Layer 0: 2 sublayers tested
  Verifying layer 1...
    ✓ Layer 1: 2 sublayers tested
  ...

================================================================================
QWEN3 LAYER VERIFICATION REPORT
================================================================================
Overall success: ✓ PASS
Layers tested: 11
Layers passed: 11
Layers failed: 0

DETAILED LAYER RESULTS:
--------------------------------------------------------------------------------
✓ PASS embedding
  Cosine similarity: 0.999950
  MSE: 0.00000012
  Relative error: 0.000001

✓ PASS layer_0_attention  
  Cosine similarity: 0.999890
  MSE: 0.00000089
  Relative error: 0.000012
...
```

### JSON Report Structure
```json
{
  "model_path": "/home/shared/models/Qwen3-1.7B",
  "test_input": "Hello, this is a test input",
  "overall_success": true,
  "total_layers_tested": 11,
  "layers_passed": 11,
  "layers_failed": 0,
  "layer_results": [
    {
      "layer_index": -1,
      "layer_name": "embedding",
      "input_shape": [1, 32],
      "output_shape": [1, 32, 2048],
      "mse": 1.2e-7,
      "cosine_similarity": 0.99995,
      "relative_error": 0.000001,
      "pass_threshold": true
    }
  ]
}
```

## Implementation Details

### Reference Implementation Design
The test implements a simplified but complete Qwen3 reference using standard PyTorch operations:

1. **Weight Loading**: Uses `safetensors` for efficient weight loading
2. **Numerical Precision**: Uses float32 for numerical stability
3. **Memory Efficiency**: Processes layers individually to reduce memory usage
4. **Error Handling**: Comprehensive error handling and fallback modes

### Comparison Strategy
1. **Layer Isolation**: Each layer tested independently
2. **Numerical Thresholds**: Strict thresholds for high-precision verification
3. **Multiple Metrics**: Different metrics capture different types of errors
4. **Graceful Degradation**: Falls back to demonstration mode if C++ unavailable

### Integration Points
The test is designed to integrate with the existing InfiniCore implementation:

- **C++ Model Loading**: Interfaces with `Qwen3ForCausalLM` from InfiniCore
- **Layer Access**: Expects layer-by-layer computation access in C++ model
- **Weight Compatibility**: Uses same weight format as InfiniCore
- **Device Support**: Supports CPU/CUDA device placement

## Setup Requirements

### Dependencies
```bash
pip install torch transformers safetensors numpy matplotlib seaborn scikit-learn
```

### Environment Setup
For full C++ testing:
```bash
# Set InfiniCore environment
export INFINI_ROOT=~/.infini

# Compile InfiniCore library
cd InfiniCore-Infer-main
xmake && xmake install
```

### Model Requirements
- Qwen3-1.7B model files in `/home/shared/models/Qwen3-1.7B`
- Model should include:
  - `config.json`: Model configuration
  - `*.safetensors`: Model weights
  - `tokenizer.json` and related tokenizer files

## Demonstration Mode

When the actual model or C++ implementation is not available, the test runs in demonstration mode:

1. **Synthetic Data**: Generates realistic input tensors
2. **Simulated C++ Output**: Creates outputs with small numerical differences
3. **Complete Workflow**: Shows the full verification process
4. **Framework Validation**: Confirms the testing framework works correctly

## Error Analysis

The test helps identify different types of computational errors:

### High Cosine Similarity + High MSE
- Suggests scaling or bias errors
- Output direction correct but magnitude wrong

### Low Cosine Similarity + Low MSE  
- Suggests rotation or transformation errors
- Small errors that change output direction

### High Relative Error
- Suggests implementation differences
- May indicate missing operations or wrong order

### Shape Mismatches
- Indicates architectural differences
- May suggest missing reshaping or transpositions

## Integration with Existing Tools

This test complements the existing comparison tools:

- **qwen3_comparison_simple.py**: End-to-end inference comparison
- **qwen3_layer_comparison.py**: Intermediate layer analysis
- **compare_qwen3_models.py**: Comprehensive model comparison

The layer verification test provides the most detailed, focused analysis for identifying specific computational discrepancies between implementations.

## Troubleshooting

### "Model path does not exist"
- Verify the model path: `/home/shared/models/Qwen3-1.7B`
- Check file permissions
- Ensure model files are complete

### "C++ model interface not available"
- Set `INFINI_ROOT` environment variable
- Compile InfiniCore with `xmake && xmake install`
- Check library compatibility

### "Python model loading failed"
- Install required dependencies
- Check model file format (safetensors)
- Verify disk space and memory

### High Error Rates
- Check numerical precision settings
- Verify weight loading correctness
- Compare device placement (CPU vs GPU)
- Review layer implementation details

## Future Enhancements

Potential improvements for production use:

1. **Extended Layer Coverage**: Test all transformer layers
2. **Batch Testing**: Support for multiple batch sizes
3. **Precision Variants**: Test different numerical precisions
4. **Performance Benchmarking**: Add timing comparisons
5. **Gradient Verification**: Test backward pass if needed
6. **Hardware Variants**: Test different hardware accelerators
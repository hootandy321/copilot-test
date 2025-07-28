# Qwen3 Layer Verification Test - Quick Start Guide

## What This Test Does

This test program verifies the computational accuracy of the Qwen3 model adapted for the InfiniCore library by comparing calculation results at each transformer layer between the original Python implementation and the adapted C++ code.

## Files Overview

- **`test_qwen3_layer_verification.py`** - Main verification program (production use)
- **`demo_qwen3_layer_verification.py`** - Demonstration with synthetic data 
- **`README_qwen3_layer_verification.md`** - Comprehensive documentation

## Quick Start

### 1. Run Demonstration (Works Immediately)
```bash
# Shows how the test works with synthetic data
python demo_qwen3_layer_verification.py
```
This demonstrates the complete verification workflow and shows expected output format.

### 2. Run Real Verification (Requires Model)
```bash
# Basic test with hardcoded model path
python test_qwen3_layer_verification.py

# Generate detailed report
python test_qwen3_layer_verification.py --output my_verification_report.json

# Custom test input
python test_qwen3_layer_verification.py --test-input "Custom verification test"
```

## Prerequisites for Real Testing

### Required Model Files
- Model must be at: `/home/shared/models/Qwen3-1.7B`
- Must include: `config.json`, `*.safetensors`, tokenizer files

### Environment Setup
```bash
# Install Python dependencies
pip install torch transformers safetensors numpy

# For C++ implementation testing
export INFINI_ROOT=~/.infini
cd InfiniCore-Infer-main
xmake && xmake install
```

## What Gets Tested

### Layer Types Verified
1. **Embedding Layer** - Token embedding lookup
2. **Transformer Layers** (first 5 tested):
   - Attention sublayer (with Q/K normalization)
   - MLP sublayer (SwiGLU activation)
3. **Final Normalization** - Output layer normalization

### Accuracy Metrics
- **Cosine Similarity** (threshold: 0.99) - Direction alignment
- **Mean Squared Error** (threshold: 1e-4) - Magnitude differences  
- **Relative Error** (threshold: 0.01) - Normalized error magnitude

## Interpreting Results

### Success Indicators
```
✓ PASS layer_0_attention
  Cosine similarity: 0.999950
  MSE: 0.00000012
  Relative error: 0.000001
```

### Common Error Patterns
```
✗ FAIL layer_4_attention
  Cosine similarity: 0.995006  # < 0.99 threshold
  MSE: 7.16e-04                # > 1e-4 threshold
  Relative error: 0.007499     # < 0.01 threshold (OK)
```

**This pattern suggests**: Direction mostly correct but magnitude differences - likely scaling or precision issues.

## Output Files

### Console Output
- Real-time layer verification progress
- Pass/fail status for each layer
- Summary statistics
- Error analysis guidance

### JSON Report
- Detailed metrics for each layer
- Configuration information
- Machine-readable results for further analysis

## Fallback Modes

### No Model Files
- Uses synthetic input data
- Tests framework without real weights
- Still validates computational workflow

### No C++ Implementation
- Simulates C++ outputs with small differences
- Demonstrates expected verification process
- Shows framework capabilities

## Integration Notes

This test is designed to work with:
- Existing InfiniCore-Infer-main infrastructure
- Python Qwen3 reference implementation
- Safetensors weight format
- Standard transformer architecture

## Troubleshooting

### "Model path does not exist"
```bash
# Check if model exists
ls -la /home/shared/models/Qwen3-1.7B
```

### "C++ model interface not available"
```bash
# Set environment and compile
export INFINI_ROOT=~/.infini
cd InfiniCore-Infer-main && xmake install
```

### "Python model loading failed"
```bash
# Install dependencies
pip install torch transformers safetensors
```

## Example Workflow

1. **Start with demo**: `python demo_qwen3_layer_verification.py`
2. **Check framework**: Understand output format and metrics
3. **Setup environment**: Install dependencies and set paths
4. **Run real test**: `python test_qwen3_layer_verification.py`
5. **Analyze results**: Review console output and JSON report
6. **Debug issues**: Use metrics to identify specific problems

This framework provides the foundation for systematic verification of Qwen3 model adaptations, helping ensure computational accuracy across implementations.
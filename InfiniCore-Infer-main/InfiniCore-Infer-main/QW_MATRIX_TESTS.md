# QW Model Matrix Test Suite

This directory contains comprehensive matrix tests for the QW (Qwen3) C++ model implementation. The tests use tiny dimensions that can be fully displayed and verified in the terminal, allowing manual verification of all computations.

## Overview

The QW model test suite provides three levels of testing:

1. **Basic Structure Test** (`qw_simple_test.cpp`) - Weight structure and data integrity
2. **Tensor Operations Test** (`qw_tensor_test.cpp`) - Advanced tensor operations (requires InfiniCore)  
3. **Complete Forward Pass Test** (`qw_complete_test.cpp`) - Full transformer computation manually implemented

## Test Configurations

All tests use minimal dimensions for complete visibility:

- **Layers**: 1-2 layers
- **Hidden Dimension**: 3-4
- **Attention Heads**: 1-2 (query/key-value)
- **Vocabulary Size**: 6-8
- **Sequence Length**: 2-3 tokens
- **Intermediate Dimension**: 4-6

These small dimensions allow every matrix element and computation to be displayed and manually verified in the terminal.

## Tests Description

### 1. Basic Structure Test (`qw_simple_test.cpp`)

**Purpose**: Verify weight structure, memory layout, and basic matrix operations

**Features**:
- ✅ Weight structure initialization with predictable patterns
- ✅ Full matrix display in terminal (all elements visible)
- ✅ Manual computation verification:
  - Token embedding lookup
  - RMS normalization
  - Matrix multiplication (Q projection)
  - SwiGLU activation
- ✅ Data integrity and pointer validation
- ✅ Memory layout verification

**Sample Output**:
```
=== Model Configuration ===
Layers: 2, Hidden: 4, Heads: 2/2, Vocab: 8

--- Input Embedding [8x4] ---
    0.100   0.101   0.102   0.103 
    0.110   0.111   0.112   0.113 
    ...

--- Manual Q Projection Test ---
Input vector: [1.00, 0.50, -0.20, 0.80]
Q projection result: [0.211, 0.211, 0.211, 0.212]

✅ All weight matrices and operations verified!
```

**Compilation & Run**:
```bash
g++ -std=c++17 -O2 -o qw_simple_test qw_simple_test.cpp
./qw_simple_test
```

### 2. Tensor Operations Test (`qw_tensor_test.cpp`)

**Purpose**: Test advanced tensor operations using QW internal tensor system

**Features**:
- Uses internal QW tensor creation and manipulation
- Tests embedding lookup with tensor API
- Advanced matrix operations
- Tensor shape and data access validation

**Note**: Requires InfiniCore libraries to compile. Currently prepared but not runnable without dependencies.

### 3. Complete Forward Pass Test (`qw_complete_test.cpp`)

**Purpose**: Simulate complete transformer forward pass with manual implementation

**Features**:
- ✅ **Complete forward pass**: Token embedding → Attention → MLP → Output prediction
- ✅ **Attention mechanism**: 
  - Pre-attention RMS normalization
  - QKV projections with separate Q/K normalization (Qwen3-specific)
  - Self-attention computation with softmax
  - Output projection with residual connection
- ✅ **MLP block**:
  - Pre-MLP RMS normalization  
  - Gate/Up projections
  - SwiGLU activation
  - Down projection with residual connection
- ✅ **Output generation**:
  - Final RMS normalization
  - Vocabulary projection
  - Next token prediction (argmax)

**Sample Output**:
```
🔍 Step 1: Token Embedding Lookup
Token 1 -> embedding: [0.20, 0.21, 0.22]
Token 3 -> embedding: [0.40, 0.41, 0.42]

🔍 Step 2: Attention Block
Q matrix [2x3]:
  [ 0.93,  1.00,  1.07]
  [ 0.95,  1.00,  1.05]
Attention weights (after softmax):
  [ 0.500,  0.500]
  [ 0.500,  0.500]

🔍 Step 3: MLP Block
SwiGLU output:
  [ 0.085,  0.127,  0.168,  0.067]

🔍 Step 4: Output Prediction
Final normalized state: [0.977, 1.098, 1.229]
Output logits: [0.379, 0.412, 0.445, 0.478, 0.511, 0.544]

✨ FINAL RESULT ✨
Input sequence: [1, 3]
Predicted next token: 5 (logit: 0.544)
```

**Compilation & Run**:
```bash
g++ -std=c++17 -O2 -o qw_complete_test qw_complete_test.cpp  
./qw_complete_test
```

## Key Features

### Manual Verification Capability
All matrices and intermediate results are displayed with full precision, allowing manual verification of computations using a calculator.

### Qwen3-Specific Features Tested
- **Q/K Normalization**: Qwen3's unique Q and K normalization after projection
- **SwiGLU Activation**: Proper gate * silu(up) computation  
- **RMS Normalization**: Correct RMS norm implementation with learnable scaling
- **Separate QKV Projections**: Individual Q, K, V projection matrices

### Computational Correctness
- **Matrix Multiplication**: Verified element-by-element
- **Attention Mechanism**: Softmax and weighted sum properly implemented
- **Residual Connections**: Proper addition of residuals
- **Normalization**: RMS norm with correct scaling and epsilon

## Weight Initialization Patterns

Weights use predictable patterns for easy verification:

- **Input Embeddings**: `token_id * 0.1 + dim * 0.01`
- **Attention Projections**: Identity-like matrices with small off-diagonals
- **MLP Projections**: Small values to prevent numerical explosion
- **Normalization Weights**: Close to 1.0 for stability

## Mathematical Verification

The tests demonstrate key transformer computations:

1. **RMS Normalization**: `y = x / sqrt(mean(x²) + ε) * weight`
2. **Self-Attention**: `softmax(QK^T/√d) * V`  
3. **SwiGLU**: `gate * silu(up)` where `silu(x) = x * sigmoid(x)`
4. **Matrix Multiplication**: Standard GEMM operations
5. **Residual Connections**: Element-wise addition

## Usage

These tests serve multiple purposes:

1. **Development**: Verify QW implementation correctness during development
2. **Debugging**: Trace computation steps when issues arise  
3. **Documentation**: Understand QW model architecture and data flow
4. **Validation**: Confirm mathematical operations are implemented correctly

## Integration with QW Implementation

The test structure mirrors the actual QW C++ implementation:

- Weight structures match `Qwen3Weights` and `Qwen3Meta` from `qwen3.h`
- Computation order follows the actual inference pipeline
- Data types and dimensions are consistent with real model constraints
- Memory layout matches the expected C++ implementation

## Extending the Tests

To extend these tests:

1. **Add new operations**: Follow the pattern of displaying inputs, computation, and outputs
2. **Test edge cases**: Add boundary conditions and special values
3. **Scale up dimensions**: Increase size while maintaining terminal visibility
4. **Add more layers**: Test multi-layer behavior
5. **Integrate with actual QW**: Replace manual implementations with QW function calls

## Build System

The tests can be built using:

1. **Direct compilation**: `g++ -std=c++17 -O2 -o test_name test_name.cpp`
2. **XMake** (when available): `xmake build qw_matrix_test` 
3. **Integration**: Tests can be integrated into existing build systems

## Conclusion

This test suite provides comprehensive verification of QW model matrix operations with complete visibility into all computations. The small dimensions ensure every step can be manually verified, making it an excellent tool for development, debugging, and validation of the QW C++ implementation.
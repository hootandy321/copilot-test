#!/usr/bin/env python3
"""
Parameter Flow Test for Qwen3 Implementation
This script tests whether parameters can be successfully passed from Python to the C++ layer.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

# Add the scripts directory to path
sys.path.append('/home/runner/work/copilot-test/copilot-test/NewInfiniCore-Infer-main/scripts')

# Set INFINI_ROOT environment variable if not set
if "INFINI_ROOT" not in os.environ:
    # The library is directly in the build directory, not in lib subdirectory
    build_dir = "/home/runner/work/copilot-test/copilot-test/NewInfiniCore-Infer-main/build/linux/x86_64/release"
    os.environ["INFINI_ROOT"] = build_dir
    # Add to LD_LIBRARY_PATH as well for runtime loading
    os.environ["LD_LIBRARY_PATH"] = build_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")

try:
    # Try direct library loading first
    import ctypes
    from ctypes import c_size_t, c_uint, c_int, c_float, c_void_p, POINTER, byref
    
    lib_path = "/home/runner/work/copilot-test/copilot-test/NewInfiniCore-Infer-main/build/linux/x86_64/release/libinfinicore_infer.so"
    
    if not os.path.exists(lib_path):
        print(f"✗ Library not found at {lib_path}")
        exit(1)
    
    # Define data types and structures directly
    class DataType(ctypes.c_int):
        INFINI_DTYPE_F16 = 12
        INFINI_DTYPE_F32 = 13
        INFINI_DTYPE_BF16 = 19

    class DeviceType(ctypes.c_int):
        DEVICE_TYPE_CPU = 0
        DEVICE_TYPE_NVIDIA = 1

    class Qwen3MetaCStruct(ctypes.Structure):
        _fields_ = [
            ("dt_logits", DataType),
            ("nlayer", c_size_t), ("d", c_size_t), ("nh", c_size_t), ("nkvh", c_size_t),
            ("dh", c_size_t), ("di", c_size_t), ("dctx", c_size_t), ("dvoc", c_size_t),
            ("epsilon", c_float), ("theta", c_float), ("end_token", c_uint),
            ("sliding_windows", POINTER(c_uint)), ("layer_types", POINTER(c_uint)),
        ]

    class Qwen3WeightsCStruct(ctypes.Structure):
        _fields_ = [
            ("nlayer", c_size_t), ("dt_norm", DataType), ("dt_mat", DataType),
            ("transpose_linear_weights", c_int),
            ("input_embd", c_void_p), ("output_norm", c_void_p), ("output_embd", c_void_p),
            ("attn_norm", POINTER(c_void_p)), ("attn_qkv", POINTER(c_void_p)),
            ("attn_qkv_b", POINTER(c_void_p)), ("attn_o", POINTER(c_void_p)),
            ("ffn_norm", POINTER(c_void_p)), ("ffn_gate_up", POINTER(c_void_p)),
            ("ffn_down", POINTER(c_void_p)), ("q_norm", POINTER(c_void_p)), ("k_norm", POINTER(c_void_p)),
        ]

    class Qwen3ModelCStruct(ctypes.Structure):
        pass

    class KVCacheCStruct(ctypes.Structure):
        pass

    # Load library
    lib = ctypes.CDLL(lib_path)
    
    # Setup function signatures
    lib.createQwen3Model.restype = POINTER(Qwen3ModelCStruct)
    lib.createQwen3Model.argtypes = [POINTER(Qwen3MetaCStruct), POINTER(Qwen3WeightsCStruct), DeviceType, c_int, POINTER(c_int)]
    lib.destroyQwen3Model.argtypes = [POINTER(Qwen3ModelCStruct)]
    lib.createQwen3KVCache.argtypes = [POINTER(Qwen3ModelCStruct)]
    lib.createQwen3KVCache.restype = POINTER(KVCacheCStruct)
    lib.dropQwen3KVCache.argtypes = [POINTER(Qwen3ModelCStruct), POINTER(KVCacheCStruct)]
    lib.inferQwen3Batch.restype = None
    lib.inferQwen3Batch.argtypes = [
        POINTER(Qwen3ModelCStruct), POINTER(c_uint), c_uint, POINTER(c_uint), c_uint, POINTER(c_uint),
        POINTER(POINTER(KVCacheCStruct)), POINTER(c_float), POINTER(c_uint), POINTER(c_float), POINTER(c_uint)
    ]
    
    # Create function wrappers
    create_qwen3_model = lib.createQwen3Model
    destroy_qwen3_model = lib.destroyQwen3Model
    create_qwen3_kv_cache = lib.createQwen3KVCache
    drop_qwen3_kv_cache = lib.dropQwen3KVCache
    infer_qwen3_batch = lib.inferQwen3Batch
    
    print("✓ Successfully loaded Qwen3 API directly from library")
    QWEN3_API_AVAILABLE = True
except Exception as e:
    print(f"✗ Failed to load Qwen3 API: {e}")
    import traceback
    traceback.print_exc()
    QWEN3_API_AVAILABLE = False
    exit(1)

from ctypes import c_int, c_uint, c_float, c_void_p, POINTER, byref

class ParameterFlowTest:
    """Test parameter passing from Python to C++"""
    
    def __init__(self):
        print("🧪 Starting Parameter Flow Test")
        
    def create_minimal_qwen3_meta(self):
        """Create minimal Qwen3Meta structure for testing"""
        meta = Qwen3MetaCStruct()
        meta.dt_logits = DataType.INFINI_DTYPE_F16
        meta.nlayer = 2  # Minimal layers for testing
        meta.d = 512     # Small hidden size
        meta.nh = 8      # Attention heads  
        meta.nkvh = 4    # Key-value heads
        meta.dh = 64     # Head dimension (d/nh = 512/8 = 64)
        meta.di = 1536   # Intermediate size (3*d = 3*512 = 1536)
        meta.dctx = 128  # Small context length
        meta.dvoc = 1000 # Small vocabulary
        meta.epsilon = 1e-6
        meta.theta = 10000.0
        meta.end_token = 0
        meta.sliding_windows = None
        meta.layer_types = None
        
        print(f"✓ Created Qwen3Meta: {meta.nlayer} layers, {meta.d} hidden_size")
        return meta
    
    def create_minimal_qwen3_weights(self, meta):
        """Create minimal Qwen3Weights structure with dummy data"""
        from ctypes import c_void_p
        
        # Create dummy weight tensors
        def create_dummy_tensor(shape, dtype=torch.float16):
            """Create a dummy tensor with the given shape"""
            tensor = torch.randn(shape, dtype=dtype)
            return tensor
        
        nlayer = meta.nlayer
        d = meta.d
        nh = meta.nh  
        nkvh = meta.nkvh
        dh = meta.dh
        di = meta.di
        dvoc = meta.dvoc
        
        # Store tensors to prevent garbage collection
        self.weight_tensors = {}
        
        # Basic weights
        self.weight_tensors['input_embd'] = create_dummy_tensor((dvoc, d))
        self.weight_tensors['output_norm'] = create_dummy_tensor((d,))
        self.weight_tensors['output_embd'] = create_dummy_tensor((d, dvoc))
        
        # Per-layer weights
        self.weight_tensors['attn_norm'] = [create_dummy_tensor((d,)) for _ in range(nlayer)]
        self.weight_tensors['attn_qkv'] = [create_dummy_tensor((d, (nh + 2*nkvh)*dh)) for _ in range(nlayer)]
        self.weight_tensors['attn_o'] = [create_dummy_tensor((nh*dh, d)) for _ in range(nlayer)]
        self.weight_tensors['ffn_norm'] = [create_dummy_tensor((d,)) for _ in range(nlayer)]
        self.weight_tensors['ffn_gate_up'] = [create_dummy_tensor((d, 2*di)) for _ in range(nlayer)]
        self.weight_tensors['ffn_down'] = [create_dummy_tensor((di, d)) for _ in range(nlayer)]
        
        # Q/K normalization weights (Qwen3 specific)
        self.weight_tensors['q_norm'] = [create_dummy_tensor((dh,)) for _ in range(nlayer)]
        self.weight_tensors['k_norm'] = [create_dummy_tensor((dh,)) for _ in range(nlayer)]
        
        # Create the C structure
        weights = Qwen3WeightsCStruct()
        weights.nlayer = nlayer
        weights.dt_norm = DataType.INFINI_DTYPE_F16
        weights.dt_mat = DataType.INFINI_DTYPE_F16
        weights.transpose_linear_weights = 1  # PyTorch format
        
        # Basic weight pointers
        weights.input_embd = self.weight_tensors['input_embd'].data_ptr()
        weights.output_norm = self.weight_tensors['output_norm'].data_ptr()
        weights.output_embd = self.weight_tensors['output_embd'].data_ptr()
        
        # Per-layer weight pointer arrays
        weights.attn_norm = (c_void_p * nlayer)(
            *[tensor.data_ptr() for tensor in self.weight_tensors['attn_norm']]
        )
        weights.attn_qkv = (c_void_p * nlayer)(
            *[tensor.data_ptr() for tensor in self.weight_tensors['attn_qkv']]
        )
        weights.attn_qkv_b = None  # No bias for this test
        weights.attn_o = (c_void_p * nlayer)(
            *[tensor.data_ptr() for tensor in self.weight_tensors['attn_o']]
        )
        weights.ffn_norm = (c_void_p * nlayer)(
            *[tensor.data_ptr() for tensor in self.weight_tensors['ffn_norm']]
        )
        weights.ffn_gate_up = (c_void_p * nlayer)(
            *[tensor.data_ptr() for tensor in self.weight_tensors['ffn_gate_up']]
        )
        weights.ffn_down = (c_void_p * nlayer)(
            *[tensor.data_ptr() for tensor in self.weight_tensors['ffn_down']]
        )
        
        # Q/K normalization weights
        weights.q_norm = (c_void_p * nlayer)(
            *[tensor.data_ptr() for tensor in self.weight_tensors['q_norm']]
        )
        weights.k_norm = (c_void_p * nlayer)(
            *[tensor.data_ptr() for tensor in self.weight_tensors['k_norm']]
        )
        
        print(f"✓ Created Qwen3Weights with {nlayer} layers of dummy data")
        return weights
    
    def test_model_creation(self):
        """Test Qwen3Model creation and destruction"""
        print("\n🧪 Testing Qwen3Model creation...")
        
        # Create metadata and weights
        meta = self.create_minimal_qwen3_meta()
        weights = self.create_minimal_qwen3_weights(meta)
        
        try:
            # Device configuration
            device = DeviceType.DEVICE_TYPE_CPU
            ndev = 1
            dev_ids = (c_int * ndev)(0)
            
            print("🔧 Calling createQwen3Model...")
            model = create_qwen3_model(
                byref(meta),
                byref(weights), 
                device,
                ndev,
                dev_ids
            )
            
            if model:
                print("✓ Qwen3Model created successfully!")
                
                # Test KV cache creation
                print("🔧 Testing KV cache creation...")
                kv_cache = create_qwen3_kv_cache(model)
                
                if kv_cache:
                    print("✓ Qwen3KVCache created successfully!")
                    
                    # Clean up KV cache
                    drop_qwen3_kv_cache(model, kv_cache)
                    print("✓ Qwen3KVCache dropped successfully!")
                else:
                    print("✗ Failed to create Qwen3KVCache")
                
                # Clean up model
                destroy_qwen3_model(model)
                print("✓ Qwen3Model destroyed successfully!")
                
                return True
            else:
                print("✗ Failed to create Qwen3Model")
                return False
                
        except Exception as e:
            print(f"✗ Exception during model creation: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_minimal_inference(self):
        """Test minimal inference with dummy data"""
        print("\n🧪 Testing minimal inference...")
        
        # Create metadata and weights
        meta = self.create_minimal_qwen3_meta()
        weights = self.create_minimal_qwen3_weights(meta)
        
        try:
            # Device configuration
            device = DeviceType.DEVICE_TYPE_CPU
            ndev = 1
            dev_ids = (c_int * ndev)(0)
            
            # Create model
            model = create_qwen3_model(byref(meta), byref(weights), device, ndev, dev_ids)
            if not model:
                print("✗ Failed to create model for inference test")
                return False
            
            # Create KV cache
            kv_cache = create_qwen3_kv_cache(model)
            if not kv_cache:
                print("✗ Failed to create KV cache for inference test")
                destroy_qwen3_model(model)
                return False
            
            # Prepare minimal inference data
            ntok = 3
            nreq = 1
            tokens = (c_uint * ntok)(1, 2, 3)  # Dummy token IDs
            req_lens = (c_uint * nreq)(ntok)   # Single request with all tokens
            req_pos = (c_uint * nreq)(0)       # Starting at position 0
            kv_caches = (POINTER(type(kv_cache)) * nreq)(kv_cache)
            temperature = (c_float * nreq)(1.0)
            topk = (c_uint * nreq)(1)
            topp = (c_float * nreq)(1.0)
            output = (c_uint * nreq)()
            
            print(f"🔧 Calling inferQwen3Batch with {ntok} tokens, {nreq} requests...")
            infer_qwen3_batch(
                model,
                tokens, ntok,
                req_lens, nreq, req_pos,
                kv_caches,
                temperature, topk, topp,
                output
            )
            
            print(f"✓ Inference completed! Output token: {output[0]}")
            
            # Clean up
            drop_qwen3_kv_cache(model, kv_cache)
            destroy_qwen3_model(model)
            
            return True
            
        except Exception as e:
            print(f"✗ Exception during inference: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_tests(self):
        """Run all parameter flow tests"""
        print("🚀 Running Parameter Flow Tests for Qwen3")
        print("=" * 60)
        
        tests = [
            ("Model Creation", self.test_model_creation),
            ("Minimal Inference", self.test_minimal_inference),
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n📋 Test: {test_name}")
            print("-" * 40)
            success = test_func()
            results.append((test_name, success))
            print(f"Result: {'✅ PASS' if success else '❌ FAIL'}")
        
        print("\n" + "=" * 60)
        print("📊 Test Summary:")
        passed = sum(1 for _, success in results if success)
        total = len(results)
        
        for test_name, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {test_name}: {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All parameter flow tests PASSED!")
            return True
        else:
            print("💥 Some parameter flow tests FAILED!")
            return False


def main():
    """Main test function"""
    if not QWEN3_API_AVAILABLE:
        print("❌ Qwen3 API not available, cannot run tests")
        return False
    
    # Set torch default device
    torch.set_default_device("cpu")
    
    # Create and run tests
    test = ParameterFlowTest()
    return test.run_all_tests()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
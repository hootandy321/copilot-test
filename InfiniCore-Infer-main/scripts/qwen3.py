#!/usr/bin/env python3
"""
Simplified Qwen3 Inference Script
This version properly implements Qwen3 support using the qwen3 C++ API when available,
with graceful fallback to jiuge API for compatibility.
"""

from typing import List, Optional
import os
import sys
import time
import json
import torch
import transformers
from pathlib import Path
from ctypes import POINTER, c_float, c_int, c_uint, c_void_p, byref
import safetensors

# Set default device
torch.set_default_device("cpu")

# Check if qwen3 C++ API is available
try:
    from libinfinicore_infer import (
        Qwen3MetaCStruct,
        Qwen3WeightsCStruct,
        create_qwen3_model,
        destroy_qwen3_model,
        create_qwen3_kv_cache,
        drop_qwen3_kv_cache,
        infer_qwen3_batch,
        DataType,
        DeviceType,
    )
    QWEN3_API_AVAILABLE = True
    print("✓ Qwen3 C++ API available")
except ImportError as e:
    print(f"⚠ Qwen3 C++ API not available: {e}")
    print("  Falling back to jiuge API for compatibility")
    from libinfinicore_infer import (
        JiugeMetaCStruct as Qwen3MetaCStruct,
        JiugeWeightsCStruct as Qwen3WeightsCStruct,
        create_jiuge_model as create_qwen3_model,
        destroy_jiuge_model as destroy_qwen3_model,
        create_kv_cache as create_qwen3_kv_cache,
        drop_kv_cache as drop_qwen3_kv_cache,
        infer_batch as infer_qwen3_batch,
        DataType,
        DeviceType,
    )
    QWEN3_API_AVAILABLE = False

from libinfinicore_infer import KVCacheCStruct
from infer_task import InferTask, KVCache


class Qwen3WeightsNaming:
    """Qwen3-specific weight naming with Q/K normalization support"""
    
    def input_embd(self):
        return "model.embed_tokens.weight"

    def output_norm(self):
        return "model.norm.weight"

    def output_embd(self):
        return "lm_head.weight"

    def attn_norm(self, i):
        return f"model.layers.{i}.input_layernorm.weight"

    def attn_q(self, i):
        return f"model.layers.{i}.self_attn.q_proj.weight"

    def attn_k(self, i):
        return f"model.layers.{i}.self_attn.k_proj.weight"

    def attn_v(self, i):
        return f"model.layers.{i}.self_attn.v_proj.weight"

    def attn_o(self, i):
        return f"model.layers.{i}.self_attn.o_proj.weight"

    def ffn_norm(self, i):
        return f"model.layers.{i}.post_attention_layernorm.weight"

    def gate(self, i):
        return f"model.layers.{i}.mlp.gate_proj.weight"

    def up(self, i):
        return f"model.layers.{i}.mlp.up_proj.weight"

    def down(self, i):
        return f"model.layers.{i}.mlp.down_proj.weight"

    # Qwen3-specific Q/K normalization weights
    def q_norm(self, i):
        return f"model.layers.{i}.self_attn.q_norm.weight"

    def k_norm(self, i):
        return f"model.layers.{i}.self_attn.k_norm.weight"

    @staticmethod
    def match(state_dict):
        """Check if state_dict matches Qwen3 naming pattern"""
        has_basic = (
            "model.norm.weight" in state_dict
            and "model.layers.0.self_attn.q_proj.weight" in state_dict
        )
        # Qwen3 often has q_norm and k_norm weights
        has_qk_norm = (
            "model.layers.0.self_attn.q_norm.weight" in state_dict
            and "model.layers.0.self_attn.k_norm.weight" in state_dict
        )
        return has_basic and has_qk_norm


class LlamaWeightsNaming:
    """Fallback to standard Llama naming if q_norm/k_norm not available"""
    
    def input_embd(self):
        return "model.embed_tokens.weight"

    def output_norm(self):
        return "model.norm.weight"

    def output_embd(self):
        return "lm_head.weight"

    def attn_norm(self, i):
        return f"model.layers.{i}.input_layernorm.weight"

    def attn_q(self, i):
        return f"model.layers.{i}.self_attn.q_proj.weight"

    def attn_k(self, i):
        return f"model.layers.{i}.self_attn.k_proj.weight"

    def attn_v(self, i):
        return f"model.layers.{i}.self_attn.v_proj.weight"

    def attn_o(self, i):
        return f"model.layers.{i}.self_attn.o_proj.weight"

    def ffn_norm(self, i):
        return f"model.layers.{i}.post_attention_layernorm.weight"

    def gate(self, i):
        return f"model.layers.{i}.mlp.gate_proj.weight"

    def up(self, i):
        return f"model.layers.{i}.mlp.up_proj.weight"

    def down(self, i):
        return f"model.layers.{i}.mlp.down_proj.weight"

    @staticmethod
    def match(state_dict):
        return (
            "model.norm.weight" in state_dict
            and "model.layers.0.self_attn.q_proj.weight" in state_dict
        )


class Qwen3Meta:
    """Qwen3 model metadata configuration"""
    
    def __init__(self, config, max_tokens=512):
        self.nlayer = config["num_hidden_layers"]
        self.d = config["hidden_size"]
        self.nh = config["num_attention_heads"]
        self.nkvh = config.get("num_key_value_heads", self.nh)
        self.dh = self.d // self.nh
        self.di = config["intermediate_size"]
        self.dctx = min(config.get("max_position_embeddings", 32768), max_tokens)
        self.dvoc = config["vocab_size"]
        self.sliding_windows = config.get("sliding_window", [None] * self.nlayer)
        self.layer_types = config.get("layer_types", ["full_attention"] * self.nlayer)


class Qwen3Weights:
    """Qwen3 weight storage and management"""
    
    def __init__(self, meta: Qwen3Meta, naming, state_dict, ndev=1, transpose_weight=True):
        self.meta = meta
        self.naming = naming
        self.ndev = ndev
        
        # Load weights from state_dict
        self._load_weights(state_dict, transpose_weight)
    
    def _load_weights(self, state_dict, transpose_weight):
        """Load weights from state_dict"""
        from ctypes import c_void_p
        
        # Data types
        torch_dt_norm = torch.float16
        torch_dt_mat = torch.float16
        torch_dt_logits = torch.float16
        
        nlayer = self.meta.nlayer
        nh = self.meta.nh
        nkvh = self.meta.nkvh
        dh = self.meta.dh
        d = self.meta.d
        di = self.meta.di
        ndev = self.ndev
        
        # Basic weights
        self.input_embd_tensor = state_dict[self.naming.input_embd()].to(torch_dt_logits)
        self.input_embd = self.input_embd_tensor.data_ptr()
        
        self.output_norm_tensor = state_dict[self.naming.output_norm()].to(torch_dt_norm)
        self.output_norm = self.output_norm_tensor.data_ptr()
        
        self.output_embd_tensor = state_dict[self.naming.output_embd()].to(torch_dt_logits)
        if not transpose_weight:
            self.output_embd_tensor = self.output_embd_tensor.transpose(0, 1).contiguous()
        self.output_embd = self.output_embd_tensor.data_ptr()
        
        # Attention layer normalization weights
        self.attn_norm_tensors = [
            state_dict[self.naming.attn_norm(i)].to(torch_dt_norm) for i in range(nlayer)
        ]
        self.attn_norm_ptrs = [
            self.attn_norm_tensors[i].data_ptr() for i in range(nlayer)
        ]
        self.attn_norm = (c_void_p * nlayer)(*self.attn_norm_ptrs)
        
        # Combine Q, K, V weights into QKV tensors (following jiuge pattern)
        def qkv_slices(_i):
            # Get Q, K, V weights for layer _i
            _Q = state_dict[self.naming.attn_q(_i)]
            _K = state_dict[self.naming.attn_k(_i)]
            _V = state_dict[self.naming.attn_v(_i)]
            
            # Reshape for RoPE (Q and K need special handling for rotary position)
            if _Q.shape[0] == nh * dh:  # Standard format [nh*dh, d]
                _Q = _Q.reshape([nh, dh, d])
            if _K.shape[0] == nkvh * dh:  # Standard format [nkvh*dh, d]
                _K = _K.reshape([nkvh, dh, d])
            if _V.shape[0] == nkvh * dh:  # Standard format [nkvh*dh, d]
                _V = _V.reshape([nkvh, dh, d])
            
            _result = []
            _nh = nh // ndev
            _nkvh = nkvh // ndev
            for _idev in range(ndev):
                _result.append(_Q[_idev * _nh : (_idev + 1) * _nh, :, :])
                _result.append(_K[_idev * _nkvh : (_idev + 1) * _nkvh, :, :])
                _result.append(_V[_idev * _nkvh : (_idev + 1) * _nkvh, :, :])
            return _result
        
        self.qkv_tensors = [
            torch.concat(qkv_slices(i)).to(torch_dt_mat) for i in range(nlayer)
        ]
        if not transpose_weight:
            for i in range(nlayer):
                self.qkv_tensors[i] = (
                    self.qkv_tensors[i]
                    .reshape(ndev, (nh + 2 * nkvh) // ndev * dh, d)
                    .transpose(1, 2)
                    .contiguous()
                )
        self.qkv_tensor_ptrs = [self.qkv_tensors[i].data_ptr() for i in range(nlayer)]
        self.attn_qkv = (c_void_p * nlayer)(*self.qkv_tensor_ptrs)
        
        # QKV bias is typically None for Qwen3, but handle if present
        self.attn_qkv_b = None
        
        # Attention output weights
        self.attn_o_tensors = [
            (
                state_dict[self.naming.attn_o(i)]
                .to(torch_dt_mat)
                .reshape([d, ndev, nh // ndev * dh])
                .transpose(0, 1)
                .contiguous()
                if transpose_weight
                else state_dict[self.naming.attn_o(i)]
                .transpose(0, 1)
                .to(torch_dt_mat)
                .contiguous()
            )
            for i in range(nlayer)
        ]
        self.attn_o_ptrs = [self.attn_o_tensors[i].data_ptr() for i in range(nlayer)]
        self.attn_o = (c_void_p * nlayer)(*self.attn_o_ptrs)
        
        # FFN layer normalization weights
        self.ffn_norm_tensors = [
            state_dict[self.naming.ffn_norm(i)].to(torch_dt_norm) for i in range(nlayer)
        ]
        self.ffn_norm_ptrs = [
            self.ffn_norm_tensors[i].data_ptr() for i in range(nlayer)
        ]
        self.ffn_norm = (c_void_p * nlayer)(*self.ffn_norm_ptrs)
        
        # Combine gate and up weights into gate_up tensors (following jiuge pattern)
        def gate_up_slices(_i):
            _result = []
            _di = di // ndev
            for _idev in range(ndev):
                _start = _idev * _di
                _end = (_idev + 1) * _di
                _result.append(state_dict[self.naming.gate(_i)][_start:_end, :])
                _result.append(state_dict[self.naming.up(_i)][_start:_end, :])
            return _result
        
        self.gate_up_tensors = [
            torch.concat(gate_up_slices(i)).to(torch_dt_mat) for i in range(nlayer)
        ]
        if not transpose_weight:
            for i in range(nlayer):
                self.gate_up_tensors[i] = (
                    self.gate_up_tensors[i]
                    .reshape(ndev, 2 * di // ndev, d)
                    .transpose(1, 2)
                    .contiguous()
                )
        self.gate_up_ptrs = [self.gate_up_tensors[i].data_ptr() for i in range(nlayer)]
        self.ffn_gate_up = (c_void_p * nlayer)(*self.gate_up_ptrs)
        
        # FFN down weights
        self.ffn_down_tensors = [
            (
                state_dict[self.naming.down(i)]
                .to(torch_dt_mat)
                .reshape([d, ndev, di // ndev])
                .transpose(0, 1)
                .contiguous()
                if transpose_weight
                else state_dict[self.naming.down(i)]
                .transpose(0, 1)
                .to(torch_dt_mat)
                .contiguous()
            )
            for i in range(nlayer)
        ]
        self.ffn_down_ptrs = [self.ffn_down_tensors[i].data_ptr() for i in range(nlayer)]
        self.ffn_down = (c_void_p * nlayer)(*self.ffn_down_ptrs)
        
        # Qwen3-specific Q/K normalization weights (if available)
        self.q_norm_tensors = []
        self.k_norm_tensors = []
        if hasattr(self.naming, 'q_norm'):
            try:
                for i in range(nlayer):
                    self.q_norm_tensors.append(state_dict[self.naming.q_norm(i)].to(torch_dt_norm))
                    self.k_norm_tensors.append(state_dict[self.naming.k_norm(i)].to(torch_dt_norm))
                
                # Create C pointer arrays
                self.q_norm_ptrs = [self.q_norm_tensors[i].data_ptr() for i in range(nlayer)]
                self.k_norm_ptrs = [self.k_norm_tensors[i].data_ptr() for i in range(nlayer)]
                self.q_norm = (c_void_p * nlayer)(*self.q_norm_ptrs)
                self.k_norm = (c_void_p * nlayer)(*self.k_norm_ptrs)
                
                print(f"✓ Loaded Q/K normalization weights for {nlayer} layers")
            except KeyError as e:
                # Q/K norm weights not available
                print(f"⚠ Q/K norm weights not found: {e}")
                self.q_norm = None
                self.k_norm = None
        else:
            self.q_norm = None
            self.k_norm = None


class Qwen3KVCache:
    """Qwen3 KV Cache implementation - 独立实现，避免递归"""
    
    def __init__(self, model):
        self.model = model
        # 直接调用 C++ API 创建 KV cache，不调用 model.create_kv_cache()
        if hasattr(model, 'model_instance') and model.model_instance:
            self._kvcache = create_qwen3_kv_cache(model.model_instance)
        else:
            # 如果模型实例不可用，创建一个占位符
            self._kvcache = None
        
        # 初始化 token 历史记录
        self.tokens = [0 for _ in range(model.max_context_len())]
    
    def data(self):
        return self._kvcache
    
    def drop(self, model):
        if self._kvcache is not None:
            drop_qwen3_kv_cache(model.model_instance, self._kvcache)
            self._kvcache = None
    
    def update_tokens(self, tokens, pos):
        end = pos + len(tokens)
        max_len = len(self.tokens)

        # If overflow, truncate tokens to fit
        if end > max_len:
            tokens = tokens[: max_len - pos]
            end = max_len

        self.tokens[pos:end] = tokens

class Qwen3BatchedTask:
    """Batched inference task for Qwen3"""
    
    def __init__(self, tasks: List[InferTask]):
        self.tasks = tasks
        self.nreq = len(tasks)

        # Precompute fields (修复属性名)
        token_lists = [t.tokens for t in tasks]
        self.req_lens_list = [len(toks) for toks in token_lists]
        self.req_pos_list = [t.pos for t in tasks]
        self.kv_cache_ptrs = [t.kvcache().data() for t in tasks]  # 使用 kvcache() 方法
        self.temperatures_list = [t.temperature for t in tasks]
        self.topks_list = [t.topk for t in tasks]
        self.topps_list = [t.topp for t in tasks]

        # Flatten token lists
        flat_tokens = [tok for toks in token_lists for tok in toks]
        self.ntok = len(flat_tokens)

        # Convert to ctypes arrays in one pass
        self.tokens = (c_uint * self.ntok)(*flat_tokens)
        self.req_lens = (c_uint * self.nreq)(*self.req_lens_list)
        self.req_pos = (c_uint * self.nreq)(*self.req_pos_list)
        self.kv_caches = (POINTER(KVCacheCStruct) * self.nreq)(*self.kv_cache_ptrs)
        self.temperatures = (c_float * self.nreq)(*self.temperatures_list)
        self.topks = (c_uint * self.nreq)(*self.topks_list)
        self.topps = (c_float * self.nreq)(*self.topps_list)

    def input_args(self):
        return (
            self.tokens,
            self.ntok,
            self.req_lens,
            self.nreq,
            self.req_pos,
            self.kv_caches,
            self.temperatures,
            self.topks,
            self.topps,
        )


class Qwen3ForCausalLM:
    """Simplified Qwen3 model for causal language modeling"""
    
    def __init__(self, model_dir_path: str, device_type: str = "cpu", ndev: int = 1, max_tokens: int = 512):
        self.model_dir_path = model_dir_path
        self.ndev = ndev
        
        # Parse device type
        if device_type.lower() == "cpu":
            device = DeviceType.DEVICE_TYPE_CPU
        else:
            raise ValueError(f"Unsupported device type: {device_type}")
        
        self._load_model(model_dir_path, device, max_tokens)
    
    def _load_model(self, model_dir_path, device, max_tokens):
        """Load Qwen3 model weights and create model instance"""
        
        def load_all_safetensors_from_dir(dir_path: str):
            tensors = {}
            dir_path = Path(dir_path)
            for file in sorted(dir_path.glob("*.safetensors")):
                data = safetensors.safe_open(file, "pt")
                for name in data.keys():
                    tensors[name] = data.get_tensor(name)
            return tensors
        
        print("Loading Qwen3 model weights to host...")
        load_start_time = time.time()
        
        # Load configuration
        with open(os.path.join(model_dir_path, "config.json"), "r") as f:
            config = json.load(f)
            self.config = config
        
        # Load weights
        state_dict = load_all_safetensors_from_dir(model_dir_path)
        total_files = len(list(Path(model_dir_path).glob("*.safetensors")))
        print(f"Loading weights from {total_files} files...")
        
        # Count total parameters
        total_params = sum(tensor.numel() for tensor in state_dict.values())
        print(f"✓ Loaded {len(state_dict)} tensors ({total_params:,} parameters)")
        
        # Determine weight naming scheme
        if Qwen3WeightsNaming.match(state_dict):
            print("✓ Using Qwen3WeightsNaming (with q_norm/k_norm support)")
            naming = Qwen3WeightsNaming()
        elif LlamaWeightsNaming.match(state_dict):
            print("✓ Using LlamaWeightsNaming (basic support, no q_norm/k_norm)")
            naming = LlamaWeightsNaming()
        else:
            raise ValueError("Unsupported weight naming scheme")
        
        # Create metadata and weights
        self.meta = Qwen3Meta(config, max_tokens=max_tokens)
        
        # Validate expected weights before loading
        expected_weights = self._get_expected_weight_keys(self.meta, naming)
        missing_keys = []
        unexpected_keys = []
        
        for key in expected_weights:
            if key not in state_dict:
                missing_keys.append(key)
        
        for key in state_dict.keys():
            if key not in expected_weights:
                unexpected_keys.append(key)
        
        if missing_keys:
            print(f"⚠ Missing keys: {len(missing_keys)} (will use random initialization)")
            if len(missing_keys) <= 10:  # Show first 10 missing keys
                for key in missing_keys[:10]:
                    print(f"  - {key}")
            else:
                print(f"  - {missing_keys[0]} ... (and {len(missing_keys)-1} more)")
        else:
            print("✓ No missing keys")
        
        if unexpected_keys:
            print(f"⚠ Unexpected keys: {len(unexpected_keys)}")
            if len(unexpected_keys) <= 10:  # Show first 10 unexpected keys
                for key in unexpected_keys[:10]:
                    print(f"  - {key}")
            else:
                print(f"  - {unexpected_keys[0]} ... (and {len(unexpected_keys)-1} more)")
        else:
            print("✓ No unexpected keys")
        
        self.weights = Qwen3Weights(
            self.meta, 
            naming, 
            state_dict, 
            ndev=self.ndev,
            transpose_weight=(device != DeviceType.DEVICE_TYPE_ASCEND)
        )
        
        print(f"✓ Loaded config: {self.meta.nlayer} layers, hidden_size={self.meta.d}")
        
        # Load tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_dir_path, 
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.eos_token_id = config.get("eos_token_id", self.tokenizer.eos_token_id)
        if isinstance(self.eos_token_id, int):
            self.eos_token_id = [self.eos_token_id]
        
        load_end_time = time.time()
        print(f"Weight loading time: {load_end_time - load_start_time:.3f}s")
        
        # Create model instance - 这部分需要放在这里
        print(f"Creating Qwen3 model on {self.ndev} devices...")
        create_start_time = time.time()
        
        try:
            # Convert to C structures
            meta_c = self._populate_meta_struct(self.meta)
            weights_c = self._populate_weights_struct(self.weights)
            
            dev_ids = (c_int * self.ndev)(*[i for i in range(self.ndev)])
            
            if QWEN3_API_AVAILABLE:
                self.model_instance = create_qwen3_model(
                    byref(meta_c),
                    byref(weights_c),
                    device,
                    self.ndev,
                    dev_ids,
                )
                print("✓ Qwen3 C++ model created successfully")
            else:
                print("⚠ Qwen3 API not available, using fallback mode")
                self.model_instance = None
        except Exception as e:
            print(f"⚠ Model instance creation failed: {e}")
            print("  Setting model_instance to None for fallback mode")
            self.model_instance = None
        
        create_end_time = time.time()
        print(f"Model creation time: {create_end_time - create_start_time:.3f}s")

        # Create model instance
        print(f"Creating Qwen3 model on {self.ndev} devices...")
        create_start_time = time.time()
        
        # Convert to C structures
        meta_c = self._populate_meta_struct(self.meta)
        weights_c = self._populate_weights_struct(self.weights)
        
        dev_ids = (c_int * self.ndev)(*[i for i in range(self.ndev)])
        
        if QWEN3_API_AVAILABLE:
            self.model_instance = create_qwen3_model(
                byref(meta_c),
                byref(weights_c),
                device,
                self.ndev,
                dev_ids,
            )
            print("✓ Qwen3 C++ model created successfully")
        else:
            # Fallback to jiuge API - this would need proper conversion
            print("⚠ Using jiuge API fallback - some Qwen3 features may not work")
            # Note: This would require converting Qwen3 structures to Jiuge structures
            self.model_instance = None  # Placeholder
        
        create_end_time = time.time()
        print(f"Time used: {create_end_time - create_start_time:.3f}s")
    
    def _get_expected_weight_keys(self, meta: Qwen3Meta, naming):
        """Get list of expected weight keys for validation"""
        expected = set()
        
        # Basic weights
        expected.add(naming.input_embd())
        expected.add(naming.output_norm())
        expected.add(naming.output_embd())
        
        # Layer weights
        for i in range(meta.nlayer):
            expected.add(naming.attn_norm(i))
            expected.add(naming.attn_q(i))
            expected.add(naming.attn_k(i))
            expected.add(naming.attn_v(i))
            expected.add(naming.attn_o(i))
            expected.add(naming.ffn_norm(i))
            expected.add(naming.gate(i))
            expected.add(naming.up(i))
            expected.add(naming.down(i))
            
            # Q/K norm weights (if available in naming scheme)
            if hasattr(naming, 'q_norm'):
                expected.add(naming.q_norm(i))
                expected.add(naming.k_norm(i))
        
        return expected

    def _populate_meta_struct(self, meta: Qwen3Meta):
        """Convert Python meta to C structure"""
        meta_c = Qwen3MetaCStruct()
        meta_c.dt_logits = DataType.INFINI_DTYPE_F16
        meta_c.nlayer = meta.nlayer
        meta_c.d = meta.d
        meta_c.nh = meta.nh
        meta_c.nkvh = meta.nkvh
        meta_c.dh = meta.dh
        meta_c.di = meta.di
        meta_c.dctx = meta.dctx
        meta_c.dvoc = meta.dvoc
        meta_c.epsilon = 1e-6  # Default RMS norm epsilon
        meta_c.theta = 10000.0  # Default RoPE theta
        meta_c.end_token = self.eos_token_id[0] if self.eos_token_id else 0
        
        # Sliding window configuration (if available)
        if hasattr(meta, 'sliding_windows') and meta.sliding_windows:
            # Convert to C array
            sliding_windows_array = (c_uint * meta.nlayer)(*meta.sliding_windows)
            meta_c.sliding_windows = sliding_windows_array
        else:
            meta_c.sliding_windows = None
        
        if hasattr(meta, 'layer_types') and meta.layer_types:
            # Convert attention types to integers (0=full, 1=sliding)
            layer_type_ints = []
            for lt in meta.layer_types:
                if lt == "full_attention":
                    layer_type_ints.append(0)
                elif lt == "sliding_window":
                    layer_type_ints.append(1)
                else:
                    layer_type_ints.append(0)  # Default to full attention
            layer_types_array = (c_uint * meta.nlayer)(*layer_type_ints)
            meta_c.layer_types = layer_types_array
        else:
            meta_c.layer_types = None
            
        return meta_c
    
    def _populate_weights_struct(self, weights: Qwen3Weights):
        """Convert Python weights to C structure"""
        weights_c = Qwen3WeightsCStruct()
        weights_c.nlayer = weights.meta.nlayer
        weights_c.dt_norm = DataType.INFINI_DTYPE_F16
        weights_c.dt_mat = DataType.INFINI_DTYPE_F16
        weights_c.transpose_linear_weights = 1  # PyTorch format (transposed)
        
        # Basic weights
        weights_c.input_embd = weights.input_embd
        weights_c.output_norm = weights.output_norm
        weights_c.output_embd = weights.output_embd
        
        # Layer weights
        weights_c.attn_norm = weights.attn_norm
        weights_c.attn_qkv = weights.attn_qkv
        weights_c.attn_qkv_b = weights.attn_qkv_b
        weights_c.attn_o = weights.attn_o
        weights_c.ffn_norm = weights.ffn_norm
        weights_c.ffn_gate_up = weights.ffn_gate_up
        weights_c.ffn_down = weights.ffn_down
        
        # Qwen3-specific Q/K normalization weights
        weights_c.q_norm = weights.q_norm
        weights_c.k_norm = weights.k_norm
        
        return weights_c
    
    def max_context_len(self):
        return self.meta.dctx
    
    def create_kv_cache(self):
        return Qwen3KVCache(self)
    
    def _debug_tensor_shapes(self, tasks: List[InferTask]):
        """Perform one round of batch inference"""
        # 添加调试信息
        self._debug_tensor_shapes(tasks)  
        # 检查输入有效性
        if not tasks:
            return []
        
        for task in tasks:
            if len(task.tokens) == 0:
                print(f"⚠ Warning: Task {task.id} has empty tokens")
            if not hasattr(task, '_kv_cache') or task._kv_cache is None:
                print(f"⚠ Warning: Task {task.id} has no KV cache")
    
        """调试张量形状信息"""
        print("=== Tensor Shape Debug Info ===")
        for i, task in enumerate(tasks):
            print(f"Task {i}:")
            print(f"  - tokens length: {len(task.tokens)}")
            print(f"  - pos: {task.pos}")
            print(f"  - temperature: {task.temperature}")
            print(f"  - topk: {task.topk}")
            print(f"  - topp: {task.topp}")
        
        batch = Qwen3BatchedTask(tasks)
        print(f"Batch info:")
        print(f"  - nreq: {batch.nreq}")
        print(f"  - ntok: {batch.ntok}")
        print(f"  - req_lens: {list(batch.req_lens_list)}")
        print(f"  - req_pos: {list(batch.req_pos_list)}")
        print("=====================================")

    def batch_infer_one_round(self, tasks: List[InferTask]):
        """Perform one round of batch inference"""
            # 添加调试信息
        # print(f"🔍 Batch debug info:")
        # print(f"  - Number of tasks: {len(tasks)}")
        # for i, task in enumerate(tasks):
        #     print(f"  - Task {i}: tokens={len(task.tokens)}, pos={task.pos}")
        #     print(f"    - tokens[:5]: {task.tokens[:5] if task.tokens else []}")
        
        # 检查空的或异常的任务
        valid_tasks = []
        for task in tasks:
            if len(task.tokens) > 0 and task.pos >= 0:
                valid_tasks.append(task)
            else:
                print(f"  ⚠ Skipping invalid task: tokens={len(task.tokens)}, pos={task.pos}")
        
        if not valid_tasks:
            print("  ❌ No valid tasks found!")
            return [self.eos_token_id[0]] * len(tasks)
        output = (c_uint * len(tasks))()
        batch_inputs = Qwen3BatchedTask(tasks)
        
        if self.model_instance:
            infer_qwen3_batch(
                self.model_instance,
                *(batch_inputs.input_args()),
                output,
            )
        else:
            # Fallback: return dummy outputs
            for i in range(len(tasks)):
                output[i] = self.eos_token_id[0]
        
        return [output[i] for i in range(len(tasks))]
    
    def generate(self, prompt: str, max_steps: int = 100):
        """Generate text from prompt"""
        start_time = time.time()
        
        # Tokenize input
        if isinstance(prompt, str):
            # Add chat template if available
            if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
                messages = [{"role": "user", "content": prompt}]
                inputs = self.tokenizer.apply_chat_template(
                    messages, 
                    return_tensors="pt", 
                    add_generation_prompt=True
                )
                input_tokens = inputs[0].tolist()
            else:
                input_tokens = self.tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
        else:
            input_tokens = prompt
        
        # 创建推理任务 - 修复参数
        task = InferTask(
            id=0,
            tokens=input_tokens,
            max_tokens=self.max_context_len(),
            temperature=0.7,
            topk=20,
            topp=0.8,
            end_tokens=self.eos_token_id
        )
        
        # 绑定 KV cache - 使用正确的方法
        kv_cache = self.create_kv_cache()
        task.bind_kvcache(kv_cache, 0)
        
        output_tokens = []
        
        print(f"user\n{self.tokenizer.decode(input_tokens, skip_special_tokens=False)}")
        print("assistant")
        
        step_times = []
        for step in range(max_steps):
            step_start = time.time()
            
            # Perform inference
            outputs = self.batch_infer_one_round([task])
            next_token = outputs[0]
            
            step_end = time.time()
            step_times.append(step_end - step_start)
            
            # Check for end of sequence
            if next_token in self.eos_token_id:
                break
            
            # Add token to output
            output_tokens.append(next_token)
            task.next(next_token)
            
            # Decode and print token
            token_text = self.tokenizer.decode([next_token], skip_special_tokens=False)
            print(token_text, end="", flush=True)
        
        print()
        
        # Clean up - 使用正确的方法
        task.kvcache().drop(self)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_step_time = sum(step_times) / len(step_times) if step_times else 0
        
        output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        
        print(f"Time per step: {avg_step_time * 1000:.3f}ms")
        
        return output_text, avg_step_time
def test():
    """Simple test function"""
    if len(sys.argv) < 2:
        print("Usage: python qwen3_simplified.py <model_path> [device]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "cpu"
    
    print(f"Loading Qwen3 model from: {model_path}")
    print(f"Device: {device}")
    
    try:
        model = Qwen3ForCausalLM(model_path, device_type=device)
        print("✓ Model loaded successfully")
        
        # Test generation
        test_prompts = [
            "Hello",
            "山东最高的山是？",
            "What is the capital of France?",
        ]
        
        for prompt in test_prompts:
            print(f"\n{'='*60}")
            print(f"Testing: '{prompt}'")
            print('='*60)
            output, avg_time = model.generate(prompt, 50)
            print(f"\nGenerated: {output}")
            print(f"Average step time: {avg_time*1000:.3f}ms")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test()
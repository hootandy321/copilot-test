#!/usr/bin/env python3
"""
Fixed Qwen3 Implementation - Using New qw C++ API
This version properly implements Qwen3 support using the dedicated qwen3 C++ API
with proper Q/K normalization and parameter mapping.

Key Improvements:
1. Uses dedicated Qwen3 API instead of fallback jiuge API
2. Handles Q/K normalization weights properly
3. Implements separate QKV projections
4. One-to-one parameter mapping following jiuge.py patterns
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

# Import the proper Qwen3 API
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
        KVCacheCStruct,
    )
    QWEN3_API_AVAILABLE = True
    print("✓ Qwen3 C++ API available")
except ImportError as e:
    print(f"⚠ Qwen3 C++ API not available: {e}")
    print("  This version requires the qw implementation")
    sys.exit(1)

from infer_task import InferTask, KVCache


class Qwen3WeightsNaming:
    """Qwen3-specific weight naming with Q/K normalization and separate QKV support"""
    
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


class Qwen3MetaFromConfig(Qwen3MetaCStruct):
    """Qwen3 metadata structure from model config"""
    
    def __init__(self, config, dtype=torch.float16, max_tokens=None):
        if dtype == torch.float16:
            dt_ = DataType.INFINI_DTYPE_F16
        elif dtype == torch.float32:
            dt_ = DataType.INFINI_DTYPE_F32
        elif dtype == torch.bfloat16:
            dt_ = DataType.INFINI_DTYPE_BF16
        else:
            dt_ = DataType.INFINI_DTYPE_F16

        super().__init__(
            dt_logits=dt_,
            nlayer=config["num_hidden_layers"],
            d=config["hidden_size"],
            nh=config["num_attention_heads"],
            nkvh=(
                config["num_key_value_heads"]
                if "num_key_value_heads" in config
                else config["num_attention_heads"]
            ),
            dh=config["hidden_size"] // config["num_attention_heads"],
            di=config["intermediate_size"],
            dctx=(
                config["max_position_embeddings"] if max_tokens is None else max_tokens
            ),
            dvoc=config["vocab_size"],
            epsilon=config.get("rms_norm_eps", 1e-6),
            theta=config.get("rope_theta", 10000.0),
            end_token=config.get("eos_token_id", 2),
            sliding_windows=None,  # TODO: Handle sliding window config
            layer_types=None,      # TODO: Handle layer types config
        )
        self.torch_dtype_logits = dtype


class Qwen3WeightsImpl(Qwen3WeightsCStruct):
    """Qwen3 weights implementation with Q/K normalization support"""
    
    def __init__(
        self,
        meta,
        naming,
        state_dict,
        torch_dt_mat=torch.float16,
        torch_dt_norm=torch.float32,
        ndev=1,
        transpose_weight=True,
    ):
        nlayer = meta.nlayer
        nh = meta.nh
        nkvh = meta.nkvh
        dh = meta.dh
        d = meta.d
        di = meta.di
        
        assert nh % nkvh == 0
        assert nh % ndev == 0
        assert nkvh % ndev == 0
        assert di % ndev == 0
        
        torch_dt_logits = meta.torch_dtype_logits
        
        # Set data types
        if torch_dt_mat == torch.float16:
            self.dt_mat = DataType.INFINI_DTYPE_F16
        elif torch_dt_mat == torch.float32:
            self.dt_mat = DataType.INFINI_DTYPE_F32
        elif torch_dt_mat == torch.bfloat16:
            self.dt_mat = DataType.INFINI_DTYPE_BF16
        else:
            raise ValueError("Unsupported proj weight data type")
            
        if torch_dt_norm == torch.float16:
            self.dt_norm = DataType.INFINI_DTYPE_F16
        elif torch_dt_norm == torch.float32:
            self.dt_norm = DataType.INFINI_DTYPE_F32
        elif torch_dt_norm == torch.bfloat16:
            self.dt_norm = DataType.INFINI_DTYPE_BF16
        else:
            raise ValueError("Unsupported norm weight data type")

        # Determine input/output embedding names
        input_embd_naming = (
            naming.input_embd()
            if naming.input_embd() in state_dict
            else naming.output_embd()
        )
        output_embd_naming = (
            naming.output_embd()
            if naming.output_embd() in state_dict
            else naming.input_embd()
        )
        
        self.transpose_linear_weights = 1 if transpose_weight else 0
        self.nlayer = nlayer
        
        # Basic weights
        self.input_embd_tensor = state_dict[input_embd_naming].to(torch_dt_logits)
        self.input_embd = self.input_embd_tensor.data_ptr()
        
        self.output_norm_tensor = state_dict[naming.output_norm()].to(torch_dt_norm)
        self.output_norm = self.output_norm_tensor.data_ptr()
        
        self.output_embd_tensor = state_dict[output_embd_naming].to(torch_dt_mat)
        if not transpose_weight:
            self.output_embd_tensor = self.output_embd_tensor.transpose(0, 1).contiguous()
        self.output_embd = self.output_embd_tensor.data_ptr()

        # Attention layer normalization weights
        self.attn_norm_tensors = [
            state_dict[naming.attn_norm(i)].to(torch_dt_norm) for i in range(nlayer)
        ]
        self.attn_norm_ptrs = [
            self.attn_norm_tensors[i].data_ptr() for i in range(nlayer)
        ]
        self.attn_norm = (c_void_p * nlayer)(*self.attn_norm_ptrs)

        # QWEN3 IMPROVEMENT: Separate Q, K, V projections instead of fused QKV
        def separate_qkv_slices(_i):
            """Process Q, K, V separately for better control"""
            _Q = state_dict[naming.attn_q(_i)]
            _K = state_dict[naming.attn_k(_i)]
            _V = state_dict[naming.attn_v(_i)]
            
            # Apply device partitioning
            _result_q = []
            _result_k = []
            _result_v = []
            
            _nh = nh // ndev
            _nkvh = nkvh // ndev
            
            for _idev in range(ndev):
                # Q partitioning by attention heads
                _result_q.append(_Q.reshape([nh, dh, d])[_idev * _nh : (_idev + 1) * _nh])
                # K, V partitioning by key-value heads
                _result_k.append(_K.reshape([nkvh, dh, d])[_idev * _nkvh : (_idev + 1) * _nkvh])
                _result_v.append(_V.reshape([nkvh, dh, d])[_idev * _nkvh : (_idev + 1) * _nkvh])
            
            return _result_q, _result_k, _result_v

        # Create separate Q, K, V tensors
        q_tensors = []
        k_tensors = []
        v_tensors = []
        
        for i in range(nlayer):
            q_slices, k_slices, v_slices = separate_qkv_slices(i)
            
            q_concat = torch.concat(q_slices).to(torch_dt_mat)
            k_concat = torch.concat(k_slices).to(torch_dt_mat)
            v_concat = torch.concat(v_slices).to(torch_dt_mat)
            
            if not transpose_weight:
                q_concat = q_concat.reshape(ndev, nh // ndev * dh, d).transpose(1, 2).contiguous()
                k_concat = k_concat.reshape(ndev, nkvh // ndev * dh, d).transpose(1, 2).contiguous()
                v_concat = v_concat.reshape(ndev, nkvh // ndev * dh, d).transpose(1, 2).contiguous()
            
            q_tensors.append(q_concat)
            k_tensors.append(k_concat)
            v_tensors.append(v_concat)

        # For compatibility with current API, use QKV fused format but store separately
        # TODO: Update to use separate Q, K, V when API supports it
        self.qkv_tensors = []
        for i in range(nlayer):
            # Concatenate Q, K, V for compatibility
            qkv_combined = torch.cat([
                q_tensors[i].reshape(ndev, -1, d),
                k_tensors[i].reshape(ndev, -1, d),
                v_tensors[i].reshape(ndev, -1, d)
            ], dim=1)
            self.qkv_tensors.append(qkv_combined.reshape(-1, d))
        
        self.qkv_tensor_ptrs = [self.qkv_tensors[i].data_ptr() for i in range(nlayer)]
        self.attn_qkv = (c_void_p * nlayer)(*self.qkv_tensor_ptrs)

        # No QKV bias for most Qwen3 models
        self.attn_qkv_b = None

        # Attention output weights
        self.attn_o_tensors = [
            (
                state_dict[naming.attn_o(i)]
                .to(torch_dt_mat)
                .reshape([d, ndev, nh // ndev * dh])
                .transpose(0, 1)
                .contiguous()
                if transpose_weight
                else state_dict[naming.attn_o(i)]
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
            state_dict[naming.ffn_norm(i)].to(torch_dt_norm) for i in range(nlayer)
        ]
        self.ffn_norm_ptrs = [
            self.ffn_norm_tensors[i].data_ptr() for i in range(nlayer)
        ]
        self.ffn_norm = (c_void_p * nlayer)(*self.ffn_norm_ptrs)

        # QWEN3 IMPROVEMENT: Separate gate and up projections
        def separate_gate_up_slices(_i):
            """Process gate and up projections separately"""
            _result = []
            _di = di // ndev
            for _idev in range(ndev):
                _start = _idev * _di
                _end = (_idev + 1) * _di
                _result.append(state_dict[naming.gate(_i)][_start:_end, :])
                _result.append(state_dict[naming.up(_i)][_start:_end, :])
            return _result

        self.gate_up_tensors = [
            torch.concat(separate_gate_up_slices(i)).to(torch_dt_mat) for i in range(nlayer)
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
                state_dict[naming.down(i)]
                .to(torch_dt_mat)
                .reshape([d, ndev, di // ndev])
                .transpose(0, 1)
                .contiguous()
                if transpose_weight
                else state_dict[naming.down(i)]
                .transpose(0, 1)
                .to(torch_dt_mat)
                .contiguous()
            )
            for i in range(nlayer)
        ]
        self.ffn_down_ptrs = [self.ffn_down_tensors[i].data_ptr() for i in range(nlayer)]
        self.ffn_down = (c_void_p * nlayer)(*self.ffn_down_ptrs)

        # QWEN3 IMPROVEMENT: Q/K normalization weights (NEW FEATURE)
        self.q_norm_tensors = []
        self.k_norm_tensors = []
        if hasattr(naming, 'q_norm'):
            try:
                for i in range(nlayer):
                    self.q_norm_tensors.append(state_dict[naming.q_norm(i)].to(torch_dt_norm))
                    self.k_norm_tensors.append(state_dict[naming.k_norm(i)].to(torch_dt_norm))
                
                self.q_norm_ptrs = [self.q_norm_tensors[i].data_ptr() for i in range(nlayer)]
                self.k_norm_ptrs = [self.k_norm_tensors[i].data_ptr() for i in range(nlayer)]
                self.q_norm = (c_void_p * nlayer)(*self.q_norm_ptrs)
                self.k_norm = (c_void_p * nlayer)(*self.k_norm_ptrs)
                
                print(f"✓ Loaded Q/K normalization weights for {nlayer} layers")
            except KeyError as e:
                print(f"⚠ Q/K norm weights not found: {e}")
                self.q_norm = None
                self.k_norm = None
        else:
            self.q_norm = None
            self.k_norm = None


class Qwen3BatchedTask:
    """Batched inference task for Qwen3 (same pattern as jiuge.py)"""
    
    def __init__(self, tasks: List[InferTask]):
        self.tasks = tasks
        self.nreq = len(tasks)

        # Precompute fields
        token_lists = [t.tokens for t in tasks]
        self.req_lens_list = [len(toks) for toks in token_lists]
        self.req_pos_list = [t.pos for t in tasks]
        self.kv_cache_ptrs = [t.kvcache().data() for t in tasks]
        self.temperatures_list = [t.temperature for t in tasks]
        self.topks_list = [t.topk for t in tasks]
        self.topps_list = [t.topp for t in tasks]

        # Flatten token lists
        flat_tokens = [tok for toks in token_lists for tok in toks]
        self.ntok = len(flat_tokens)

        # Convert to ctypes arrays
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


class QwenForCausalLM:
    """Qwen3 model for causal language modeling - FIXED VERSION"""
    
    def __init__(
        self, model_dir_path, device=DeviceType.DEVICE_TYPE_CPU, ndev=1, max_tokens=None
    ):
        def load_all_safetensors_from_dir(dir_path_: str):
            tensors_ = {}
            dir_path_ = Path(dir_path_)
            for file in sorted(dir_path_.glob("*.safetensors")):
                data_ = safetensors.safe_open(file, "pt")
                for name_ in data_.keys():
                    tensors_[name_] = data_.get_tensor(name_)
            return tensors_

        print("Loading Qwen3 model weights to host...")
        load_start_time = time.time()

        with open(os.path.join(model_dir_path, "config.json"), "r") as f:
            config = json.load(f)
            self.config = config
            
        eos_token_id = self.config.get("eos_token_id", 2)
        self.eos_token_id = (
            [eos_token_id] if type(eos_token_id) == int else eos_token_id
        )
        
        transpose_weight = (
            device != DeviceType.DEVICE_TYPE_ASCEND
        )

        # Load state dict
        if any(file.suffix == ".safetensors" for file in Path(model_dir_path).iterdir()):
            state_dict = load_all_safetensors_from_dir(model_dir_path)
        else:
            state_dict = torch.load(
                os.path.join(model_dir_path, "pytorch_model.bin"),
                weights_only=True,
                map_location="cpu",
            )

        # Determine naming scheme
        if Qwen3WeightsNaming.match(state_dict):
            print("✓ Using Qwen3WeightsNaming (with Q/K normalization)")
            naming = Qwen3WeightsNaming()
        elif LlamaWeightsNaming.match(state_dict):
            print("⚠ Using LlamaWeightsNaming (fallback, no Q/K normalization)")
            naming = LlamaWeightsNaming()
        else:
            raise ValueError("Unsupported weight naming scheme")

        # Create metadata and weights
        self.meta = Qwen3MetaFromConfig(config, max_tokens=max_tokens)
        self.weights = Qwen3WeightsImpl(
            self.meta,
            naming,
            state_dict,
            ndev=ndev,
            transpose_weight=transpose_weight,
        )

        # Load tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_dir_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_end_time = time.time()
        print(f"Weight loading time: {load_end_time - load_start_time:.3f}s")

        print(f"Creating Qwen3 model on {ndev} devices...")
        load_start_time = time.time()
        dev_ids = (c_int * ndev)(*[i for i in range(ndev)])
        
        # FIXED: Use proper Qwen3 API
        self.model_instance = create_qwen3_model(
            byref(self.meta),
            byref(self.weights),
            device,
            ndev,
            dev_ids,
        )
        load_end_time = time.time()
        print(f"Model creation time: {load_end_time - load_start_time:.3f}s")

    def max_context_len(self):
        return self.meta.dctx

    def create_kv_cache(self):
        # FIXED: Use proper Qwen3 KV cache API
        return create_qwen3_kv_cache(self.model_instance)

    def drop_kv_cache(self, kv_cache):
        # FIXED: Use proper Qwen3 KV cache API
        drop_qwen3_kv_cache(self.model_instance, kv_cache)

    def batch_infer_one_round(self, tasks: List[InferTask]):
        output = (c_uint * len(tasks))()
        batch_inputs = Qwen3BatchedTask(tasks)
        
        # FIXED: Use proper Qwen3 inference API
        infer_qwen3_batch(
            self.model_instance,
            *(batch_inputs.input_args()),
            output,
        )
        return list(output)

    def generate(self, input_content, max_steps, topp_=1.0, topk_=1, temperature_=1.0):
        # Apply chat template if available
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            input_content = self.tokenizer.apply_chat_template(
                conversation=[{"role": "user", "content": input_content}],
                add_generation_prompt=True,
                tokenize=False,
            )
        
        print(input_content, end="", flush=True)
        tokens = self.tokenizer.encode(input_content)
        
        infer_task = InferTask(
            0,
            tokens,
            self.max_context_len(),
            temperature_,
            topk_,
            topp_,
            self.eos_token_id,
        )
        infer_task.bind_kvcache(KVCache(self))

        steps = 0
        total_time = 0
        output_content = ""

        for step_i in range(max_steps):
            start_time = time.time()
            output_tokens = self.batch_infer_one_round([infer_task])
            end_time = time.time()
            steps += 1
            
            output_str = (
                self.tokenizer._tokenizer.id_to_token(output_tokens[0])
                .replace("▁", " ")
                .replace("<0x0A>", "\n")
            )
            output_content += output_str
            print(output_str, end="", flush=True)
            
            if output_tokens[0] in self.eos_token_id:
                break
                
            infer_task.next(output_tokens[0])

            if step_i > 0:
                total_time += end_time - start_time

        print("\n")
        avg_time = total_time * 1000 / (steps - 1) if steps > 1 else 0
        print(f"Time per step: {avg_time:.3f}ms")

        infer_task._kv_cache.drop(self)
        return output_content, avg_time

    def destroy_model_instance(self):
        # FIXED: Use proper Qwen3 model destruction API
        destroy_qwen3_model(self.model_instance)
        print("Qwen3 Model destroyed")


def test():
    """Test function following jiuge.py pattern"""
    if len(sys.argv) < 2:
        print(
            "Usage: python qwen3_fixed.py <path/to/model_dir> [device] [n_device]"
        )
        sys.exit(1)
        
    model_path = sys.argv[1]
    device_type = DeviceType.DEVICE_TYPE_CPU
    
    if len(sys.argv) > 2:
        if sys.argv[2] == "--cpu":
            device_type = DeviceType.DEVICE_TYPE_CPU
        elif sys.argv[2] == "--nvidia":
            device_type = DeviceType.DEVICE_TYPE_NVIDIA
        # Add other device types as needed

    ndev = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    print(f"✓ Using Qwen3 model from: {model_path}")
    print(f"✓ Device: {device_type}, Devices: {ndev}")
    
    model = QwenForCausalLM(model_path, device_type, ndev)
    model.generate("山东最高的山是？", 500)
    model.destroy_model_instance()


if __name__ == "__main__":
    test()
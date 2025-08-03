from typing import List
from libinfinicore_infer import (
    Qwen3MetaCStruct,
    Qwen3WeightsCStruct,
    KVCacheCStruct,
    DataType,
    DeviceType,
    create_qwen3_model,
    destroy_qwen3_model,
    create_kv_cache,
    drop_kv_cache,
    infer_batch,
)
from infer_task import InferTask, KVCache

from ctypes import POINTER, c_float, c_int, c_uint, c_void_p, byref
import os
from pathlib import Path
import safetensors
import sys
import time
import json
import math
import torch
import transformers

torch.set_default_device("cpu")


class Qwen3WeightsNaming:
    """定义 Qwen3 模型的权重命名规则"""
    
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

    def attn_q_b(self, i):
        return f"model.layers.{i}.self_attn.q_proj.bias"

    def attn_k_b(self, i):
        return f"model.layers.{i}.self_attn.k_proj.bias"

    def attn_v_b(self, i):
        return f"model.layers.{i}.self_attn.v_proj.bias"

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
        """检查状态字典是否匹配 Qwen3 模型"""
        return (
            "model.norm.weight" in state_dict
            and "model.layers.0.self_attn.q_proj.weight" in state_dict
            and "model.layers.0.mlp.gate_proj.weight" in state_dict
        )


class Qwen3MetaFromConfig(Qwen3MetaCStruct):
    """从配置文件创建 Qwen3 元数据"""
    
    def __init__(self, config, dtype=torch.float16, max_tokens=None):
        if dtype == torch.float16:
            dt_ = DataType.INFINI_DTYPE_F16
        elif dtype == torch.float32:
            dt_ = DataType.INFINI_DTYPE_F32
        elif dtype == torch.bfloat16:
            dt_ = DataType.INFINI_DTYPE_BF16
        else:
            dt_ = DataType.INFINI_DTYPE_F16

        # Qwen3 1.7B specific parameters 
        # Note: The README mentions larger dimensions (5120 hidden, 25600 intermediate)
        # but these result in ~12B parameters, not 1.7B. Using realistic 1.7B dimensions:
        qwen3_1_7b_config = {
            "num_hidden_layers": 26,  # Calculated for ~1.7B parameters
            "hidden_size": 2048,      # Adjusted for 1.7B model size
            "num_attention_heads": 32, # Adjusted accordingly  
            "num_key_value_heads": 32, # Assume same as attention heads for now
            "intermediate_size": 6144, # Adjusted for 1.7B model size
            "vocab_size": 151936,     # From README - Qwen3 specific
            "max_position_embeddings": 32768,  # Default Qwen3
            "rms_norm_eps": 1e-6,     # Default Qwen3
            "rope_theta": 10000.0,    # Default Qwen3
        }

        # Override with provided config values
        for key, value in qwen3_1_7b_config.items():
            if key not in config:
                config[key] = value

        super().__init__(
            dt_logits=dt_,
            nlayer=config["num_hidden_layers"],
            d=config["hidden_size"],
            nh=config["num_attention_heads"],
            nkvh=config.get("num_key_value_heads", config["num_attention_heads"]),
            dh=config["hidden_size"] // config["num_attention_heads"],
            di=config["intermediate_size"],
            dctx=(
                config["max_position_embeddings"] if max_tokens is None else max_tokens
            ),
            dvoc=config["vocab_size"],
            epsilon=config["rms_norm_eps"],
            theta=config.get("rope_theta", 10000.0),
            end_token=151643,  # Qwen3 specific end token
            # Qwen3 specific scaling factors
            scale_input=1.0,
            scale_output=1.0,
            scale_o=1.0,
            scale_down=1.0,
        )
        self.torch_dtype_logits = dtype


class Qwen3WeightsImpl(Qwen3WeightsCStruct):
    """权重实现，处理权重加载和格式转换"""
    
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
        scale_input = meta.scale_input
        scale_output = meta.scale_output
        scale_o = meta.scale_o
        scale_down = meta.scale_down
        
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

        # Handle embedding weights
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
        
        # Input embedding
        self.input_embd_tensor = (
            state_dict[input_embd_naming].to(torch_dt_logits) * scale_input
        )
        self.input_embd = self.input_embd_tensor.data_ptr()
        
        # Output norm
        self.output_norm_tensor = (
            state_dict[naming.output_norm()].to(torch_dt_norm) * scale_output
        )
        self.output_norm = self.output_norm_tensor.data_ptr()
        
        # Output embedding
        self.output_embd_tensor = state_dict[output_embd_naming].to(torch_dt_mat)
        if not transpose_weight:
            self.output_embd_tensor = self.output_embd_tensor.T.contiguous()
        self.output_embd = self.output_embd_tensor.data_ptr()

        # Layer-specific weights
        self.attn_norm_tensors = []
        self.attn_qkv_tensors = []
        self.attn_qkv_b_tensors = []
        self.attn_o_tensors = []
        self.ffn_norm_tensors = []
        self.ffn_gate_up_tensors = []
        self.ffn_down_tensors = []

        self.attn_norm_ptrs = []
        self.attn_qkv_ptrs = []
        self.attn_qkv_b_ptrs = []
        self.attn_o_ptrs = []
        self.ffn_norm_ptrs = []
        self.ffn_gate_up_ptrs = []
        self.ffn_down_ptrs = []

        for layer in range(nlayer):
            # Attention norm
            attn_norm_tensor = state_dict[naming.attn_norm(layer)].to(torch_dt_norm)
            self.attn_norm_tensors.append(attn_norm_tensor)
            self.attn_norm_ptrs.append(attn_norm_tensor.data_ptr())

            # QKV projection (concatenated)
            q_tensor = state_dict[naming.attn_q(layer)].to(torch_dt_mat)
            k_tensor = state_dict[naming.attn_k(layer)].to(torch_dt_mat)
            v_tensor = state_dict[naming.attn_v(layer)].to(torch_dt_mat)
            
            # Concatenate Q, K, V weights
            qkv_tensor = torch.cat([q_tensor, k_tensor, v_tensor], dim=0)
            if not transpose_weight:
                qkv_tensor = qkv_tensor.T.contiguous()
            self.attn_qkv_tensors.append(qkv_tensor)
            self.attn_qkv_ptrs.append(qkv_tensor.data_ptr())

            # QKV bias (if exists)
            has_bias = naming.attn_q_b(layer) in state_dict
            if has_bias:
                q_bias = state_dict[naming.attn_q_b(layer)].to(torch_dt_mat)
                k_bias = state_dict[naming.attn_k_b(layer)].to(torch_dt_mat)
                v_bias = state_dict[naming.attn_v_b(layer)].to(torch_dt_mat)
                qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
                self.attn_qkv_b_tensors.append(qkv_bias)
                self.attn_qkv_b_ptrs.append(qkv_bias.data_ptr())
            else:
                self.attn_qkv_b_tensors.append(None)
                self.attn_qkv_b_ptrs.append(None)

            # Output projection
            o_tensor = state_dict[naming.attn_o(layer)].to(torch_dt_mat) * scale_o
            if not transpose_weight:
                o_tensor = o_tensor.T.contiguous()
            self.attn_o_tensors.append(o_tensor)
            self.attn_o_ptrs.append(o_tensor.data_ptr())

            # FFN norm
            ffn_norm_tensor = state_dict[naming.ffn_norm(layer)].to(torch_dt_norm)
            self.ffn_norm_tensors.append(ffn_norm_tensor)
            self.ffn_norm_ptrs.append(ffn_norm_tensor.data_ptr())

            # Gate and Up projections (concatenated)
            gate_tensor = state_dict[naming.gate(layer)].to(torch_dt_mat)
            up_tensor = state_dict[naming.up(layer)].to(torch_dt_mat)
            gate_up_tensor = torch.cat([gate_tensor, up_tensor], dim=0)
            if not transpose_weight:
                gate_up_tensor = gate_up_tensor.T.contiguous()
            self.ffn_gate_up_tensors.append(gate_up_tensor)
            self.ffn_gate_up_ptrs.append(gate_up_tensor.data_ptr())

            # Down projection
            down_tensor = state_dict[naming.down(layer)].to(torch_dt_mat) * scale_down
            if not transpose_weight:
                down_tensor = down_tensor.T.contiguous()
            self.ffn_down_tensors.append(down_tensor)
            self.ffn_down_ptrs.append(down_tensor.data_ptr())

        # Initialize parent struct
        super().__init__(
            nlayer=nlayer,
            dt_norm=self.dt_norm,
            dt_mat=self.dt_mat,
            transpose_linear_weights=self.transpose_linear_weights,
            input_embd=self.input_embd,
            output_norm=self.output_norm,
            output_embd=self.output_embd,
            attn_norm=self.attn_norm_ptrs,
            attn_qkv=self.attn_qkv_ptrs,
            attn_qkv_b=self.attn_qkv_b_ptrs if has_bias else None,
            attn_o=self.attn_o_ptrs,
            ffn_norm=self.ffn_norm_ptrs,
            ffn_gate_up=self.ffn_gate_up_ptrs,
            ffn_down=self.ffn_down_ptrs,
        )


class Qwen3ForCausalLM:
    """主要的 Qwen3 模型接口类"""
    
    def __init__(self, model_dir_path, device=DeviceType.DEVICE_TYPE_CPU, ndev=1):
        self.model_dir_path = Path(model_dir_path)
        self.device = device
        self.ndev = ndev
        
        # Load config
        config_path = self.model_dir_path / "config.json"
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        # Load model state dict
        self.state_dict = self._load_state_dict()
        
        # Create meta and weights
        self.naming = Qwen3WeightsNaming()
        self.meta = Qwen3MetaFromConfig(self.config)
        self.weights = Qwen3WeightsImpl(
            self.meta, self.naming, self.state_dict, ndev=ndev
        )
        
        # Create model
        dev_ids = list(range(ndev))
        self.model = create_qwen3_model(
            byref(self.meta), byref(self.weights), device, ndev, 
            (c_int * ndev)(*dev_ids)
        )
        
        if not self.model:
            raise RuntimeError("Failed to create Qwen3 model")

    def _load_state_dict(self):
        """Load model weights from safetensors or pytorch files"""
        state_dict = {}
        
        # Try to load from safetensors first
        safetensors_files = list(self.model_dir_path.glob("*.safetensors"))
        if safetensors_files:
            for file in safetensors_files:
                with safetensors.safe_open(file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
        else:
            # Fallback to pytorch files
            pytorch_files = list(self.model_dir_path.glob("*.bin"))
            for file in pytorch_files:
                checkpoint = torch.load(file, map_location="cpu")
                state_dict.update(checkpoint)
        
        return state_dict

    def create_kv_cache(self):
        """创建 KV Cache"""
        cache_ptr = create_kv_cache(self.model)
        return KVCache(cache_ptr, self.model)

    def batch_infer_one_round(self, tasks: List[InferTask]):
        """批量推理一轮"""
        if not tasks:
            return
        
        # Prepare input data
        tokens = []
        req_lens = []
        req_pos = []
        kv_caches = []
        temperatures = []
        topks = []
        topps = []
        
        for task in tasks:
            tokens.extend(task.tokens)
            req_lens.append(len(task.tokens))
            req_pos.append(task.pos)
            kv_caches.append(task.kv_cache.ptr)
            temperatures.append(task.temperature)
            topks.append(task.top_k)
            topps.append(task.top_p)
        
        ntok = len(tokens)
        nreq = len(tasks)
        
        # Convert to C types
        tokens_array = (c_uint * ntok)(*tokens)
        req_lens_array = (c_uint * nreq)(*req_lens)
        req_pos_array = (c_uint * nreq)(*req_pos)
        kv_caches_array = (c_void_p * nreq)(*kv_caches)
        temperatures_array = (c_float * nreq)(*temperatures)
        topks_array = (c_uint * nreq)(*topks)
        topps_array = (c_float * nreq)(*topps)
        output_array = (c_uint * nreq)()
        
        # Run inference
        infer_batch(
            self.model,
            tokens_array, ntok,
            req_lens_array, nreq, req_pos_array,
            kv_caches_array,
            temperatures_array, topks_array, topps_array,
            output_array
        )
        
        # Update tasks with results
        for i, task in enumerate(tasks):
            task.output_token = output_array[i]
            task.pos += len(task.tokens)

    def generate(self, input_content, max_steps=50, temperature=0.0, top_k=1, top_p=1.0):
        """生成文本"""
        # This would need a tokenizer - simplified for now
        # In practice, you'd load the tokenizer from the model directory
        tokens = [1]  # Simplified: assume input is already tokenized
        
        kv_cache = self.create_kv_cache()
        task = InferTask(
            tokens=tokens,
            pos=0,
            kv_cache=kv_cache,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        generated_tokens = []
        for _ in range(max_steps):
            self.batch_infer_one_round([task])
            if task.output_token == self.meta.end_token:
                break
            generated_tokens.append(task.output_token)
            task.tokens = [task.output_token]
        
        return generated_tokens

    def __del__(self):
        """析构函数"""
        if hasattr(self, 'model') and self.model:
            destroy_qwen3_model(self.model)


# 兼容性函数
def create_qwen3_for_causal_lm(model_path, device_type=DeviceType.DEVICE_TYPE_CPU, ndev=1):
    """创建 Qwen3 因果语言模型"""
    return Qwen3ForCausalLM(model_path, device_type, ndev)
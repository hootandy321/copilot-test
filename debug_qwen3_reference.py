#!/usr/bin/env python3
"""
Debug version of Qwen3 reference implementation
Based on qwen3/modeling_qwen3.py but modified to save intermediate layer outputs for debugging C++ vs Python differences.

This script serves as the "golden reference" implementation that will be compared against the C++ version.
"""

import torch
import numpy as np
import json
import os
from pathlib import Path
from typing import Optional, Tuple, Union
import safetensors
from transformers import AutoTokenizer, AutoConfig

# Set device and precision
torch.set_default_device("cpu")
torch.set_default_dtype(torch.float32)


def save_tensor_debug(tensor, name, step=None):
    """Save tensor data for debugging comparison"""
    if step is not None:
        filename = f"py_layer_{step}_{name}.txt"
    else:
        filename = f"py_{name}.txt"
        
    # Convert to numpy and save as text
    if isinstance(tensor, torch.Tensor):
        data = tensor.detach().cpu().numpy()
    else:
        data = np.array(tensor)
    
    # Save both the raw data and some statistics
    with open(filename, 'w') as f:
        f.write(f"# Tensor: {name}\n")
        f.write(f"# Shape: {data.shape}\n")
        f.write(f"# Dtype: {data.dtype}\n")
        f.write(f"# Mean: {data.mean()}\n")
        f.write(f"# Std: {data.std()}\n")
        f.write(f"# Min: {data.min()}\n")
        f.write(f"# Max: {data.max()}\n")
        f.write("# Data:\n")
        
        # For large tensors, save a flattened version for easier comparison
        if data.size > 1000:
            # Save first and last elements, plus some statistics
            flat_data = data.flatten()
            f.write("# First 50 elements:\n")
            for i in range(min(50, len(flat_data))):
                f.write(f"{flat_data[i]:.6e}\n")
            f.write("# Last 50 elements:\n")
            for i in range(max(0, len(flat_data)-50), len(flat_data)):
                f.write(f"{flat_data[i]:.6e}\n")
        else:
            # Save all elements for small tensors
            flat_data = data.flatten()
            for val in flat_data:
                f.write(f"{val:.6e}\n")
    
    print(f"  Saved debug tensor: {filename} (shape: {data.shape})")


class DebugQwen3RMSNorm(torch.nn.Module):
    """RMSNorm implementation matching the modeling_qwen3.py version"""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class DebugQwen3MLP(torch.nn.Module):
    """MLP implementation matching modeling_qwen3.py"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = torch.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # Use SiLU activation (same as Swish)
        self.act_fn = torch.nn.SiLU()

    def forward(self, x, debug_layer=None):
        # Gate and Up projections
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        if debug_layer is not None:
            save_tensor_debug(gate, "mlp_gate_proj", debug_layer)
            save_tensor_debug(up, "mlp_up_proj", debug_layer)
        
        # SwiGLU: gate * SiLU(up)
        activated_up = self.act_fn(up)
        intermediate = gate * activated_up
        
        if debug_layer is not None:
            save_tensor_debug(activated_up, "mlp_activated_up", debug_layer)
            save_tensor_debug(intermediate, "mlp_intermediate", debug_layer)
        
        # Down projection
        output = self.down_proj(intermediate)
        
        if debug_layer is not None:
            save_tensor_debug(output, "mlp_output", debug_layer)
        
        return output


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value tensors for grouped query attention"""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class DebugQwen3Attention(torch.nn.Module):
    """Attention implementation matching modeling_qwen3.py with debugging"""
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = True

        self.q_proj = torch.nn.Linear(config.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = torch.nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = torch.nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = torch.nn.Linear(self.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        
        # Q/K normalization (Qwen3-specific)
        self.q_norm = DebugQwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = DebugQwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        debug_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Q, K, V projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        if debug_layer is not None:
            save_tensor_debug(query_states, "attn_q_proj_raw", debug_layer)
            save_tensor_debug(key_states, "attn_k_proj_raw", debug_layer)
            save_tensor_debug(value_states, "attn_v_proj_raw", debug_layer)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply Q/K normalization (Qwen3-specific)
        # We need to apply this per-head
        batch_size, num_heads, seq_len, head_dim = query_states.shape
        query_states_flat = query_states.reshape(-1, head_dim)  # [batch*heads*seq, head_dim]
        query_states_normed = self.q_norm(query_states_flat)
        query_states = query_states_normed.reshape(batch_size, num_heads, seq_len, head_dim)
        
        batch_size, num_kv_heads, seq_len, head_dim = key_states.shape
        key_states_flat = key_states.reshape(-1, head_dim)
        key_states_normed = self.k_norm(key_states_flat)
        key_states = key_states_normed.reshape(batch_size, num_kv_heads, seq_len, head_dim)
        
        if debug_layer is not None:
            save_tensor_debug(query_states, "attn_q_normed", debug_layer)
            save_tensor_debug(key_states, "attn_k_normed", debug_layer)
        
        # Apply rotary position embedding
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        if debug_layer is not None:
            save_tensor_debug(query_states, "attn_q_rope", debug_layer)
            save_tensor_debug(key_states, "attn_k_rope", debug_layer)
        
        # Repeat KV for grouped query attention
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        
        if debug_layer is not None:
            save_tensor_debug(attn_weights, "attn_scores", debug_layer)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Apply softmax
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        if debug_layer is not None:
            save_tensor_debug(attn_weights, "attn_weights", debug_layer)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_attention_heads * self.head_dim)
        
        if debug_layer is not None:
            save_tensor_debug(attn_output, "attn_values_output", debug_layer)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        if debug_layer is not None:
            save_tensor_debug(attn_output, "attn_o_proj", debug_layer)
        
        return attn_output, attn_weights


class DebugQwen3DecoderLayer(torch.nn.Module):
    """Decoder layer with debugging output"""
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        
        self.self_attn = DebugQwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = DebugQwen3MLP(config)
        self.input_layernorm = DebugQwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DebugQwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        debug_outputs: bool = False,
    ) -> torch.Tensor:
        
        residual = hidden_states
        
        if debug_outputs:
            save_tensor_debug(hidden_states, "input_hidden_states", self.layer_idx)
        
        # Pre-attention normalization
        hidden_states = self.input_layernorm(hidden_states)
        
        if debug_outputs:
            save_tensor_debug(hidden_states, "attn_norm_output", self.layer_idx)
        
        # Self attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            debug_layer=self.layer_idx if debug_outputs else None,
        )
        
        # Residual connection
        hidden_states = residual + hidden_states
        
        if debug_outputs:
            save_tensor_debug(hidden_states, "attn_residual_output", self.layer_idx)
        
        # MLP block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        if debug_outputs:
            save_tensor_debug(hidden_states, "mlp_norm_output", self.layer_idx)
        
        hidden_states = self.mlp(hidden_states, debug_layer=self.layer_idx if debug_outputs else None)
        hidden_states = residual + hidden_states
        
        if debug_outputs:
            save_tensor_debug(hidden_states, "layer_output", self.layer_idx)
        
        return hidden_states


class DebugQwen3RotaryEmbedding(torch.nn.Module):
    """Rotary position embedding"""
    
    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size // config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta if hasattr(config, 'rope_theta') else 10000.0

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class DebugQwen3Model(torch.nn.Module):
    """Debug version of Qwen3 model that saves intermediate layer outputs"""
    
    def __init__(self, config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config

        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = torch.nn.ModuleList([
            DebugQwen3DecoderLayer(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = DebugQwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DebugQwen3RotaryEmbedding(config=config)

    def _create_causal_mask(self, seq_len, device, dtype):
        """Create causal attention mask"""
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

    def forward(
        self,
        input_ids: torch.LongTensor,
        debug_outputs: bool = False,
    ) -> torch.Tensor:
        
        # Input embeddings
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if debug_outputs:
            save_tensor_debug(input_ids, "input_ids", "input")
            save_tensor_debug(hidden_states, "input_embeddings", "input")
        
        # Position IDs
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Create attention mask
        attention_mask = self._create_causal_mask(seq_len, device, hidden_states.dtype)
        
        # Create position embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        if debug_outputs:
            save_tensor_debug(position_embeddings[0], "position_cos", "input")
            save_tensor_debug(position_embeddings[1], "position_sin", "input")
        
        # Pass through decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                debug_outputs=debug_outputs,
            )
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        if debug_outputs:
            save_tensor_debug(hidden_states, "final_norm_output", "output")
        
        return hidden_states


def run_debug_inference(model_path: str, input_text: str = "Hello world"):
    """Run debug inference with intermediate output saving"""
    
    print(f"Loading model from: {model_path}")
    
    # Load tokenizer and config
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer/config: {e}")
        return
    
    # Create debug model
    model = DebugQwen3Model(config)
    
    # Load weights
    print("Loading model weights...")
    model_files = list(Path(model_path).glob("*.safetensors"))
    if not model_files:
        model_files = list(Path(model_path).glob("pytorch_model*.bin"))
    
    if not model_files:
        print("No model weight files found!")
        return
    
    # Load state dict
    state_dict = {}
    for file in model_files:
        if file.suffix == ".safetensors":
            with safetensors.safe_open(file, framework="pt") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        else:
            checkpoint = torch.load(file, map_location="cpu", weights_only=True)
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            state_dict.update(checkpoint)
    
    # Load weights into model
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print(f"Model loaded successfully. Config: {config.num_hidden_layers} layers, {config.hidden_size} dim")
    
    # Tokenize input
    if input_text is None:
        # Use a simple fixed input for reproducibility
        input_ids = torch.tensor([[1, 15339, 1079]], dtype=torch.long)  # Simple token sequence
        print("Using fixed input tokens: [1, 15339, 1079]")
    else:
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        print(f"Input text: '{input_text}'")
        print(f"Input tokens: {input_ids.tolist()}")
    
    # Clear any existing debug files
    for f in Path(".").glob("py_*.txt"):
        f.unlink()
    
    print("\nRunning debug inference...")
    with torch.no_grad():
        hidden_states = model(input_ids, debug_outputs=True)
    
    print(f"Inference completed. Final hidden states shape: {hidden_states.shape}")
    print("Debug output files saved with prefix 'py_'")
    
    return hidden_states


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python debug_qwen3_reference.py <model_path> [input_text]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    input_text = sys.argv[2] if len(sys.argv) > 2 else None
    
    run_debug_inference(model_path, input_text)
#!/usr/bin/env python3
"""
Qwen3 C++ Implementation Verification Tool

This script tests the computational accuracy of the C++ Qwen3 implementation
by comparing it against a PyTorch reference implementation using the same weights.
It focuses on verifying that the C++ inference produces correct results.
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

# Try to import our implementations
sys.path.insert(0, '/home/runner/work/copilot-test/copilot-test/qwen3')

try:
    from qwen3_simplified import Qwen3ForCausalLM as CppQwen3Model
    CPP_MODEL_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"⚠ C++ Qwen3 model not available: {e}")
    CPP_MODEL_AVAILABLE = False

try:
    import transformers
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠ Transformers not available")
    TRANSFORMERS_AVAILABLE = False


@dataclass
class TestResult:
    """Single test result"""
    input_text: str
    cpp_output: Optional[str]
    py_output: Optional[str]
    cpp_tokens: Optional[List[int]]
    py_tokens: Optional[List[int]]
    cpp_time: float
    py_time: float
    token_accuracy: float
    text_similarity: float
    error: Optional[str] = None


class SimplePyTorchQwen3:
    """Simple PyTorch reference implementation for comparison"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.config = None
        self.weights = None
        self.tokenizer = None
        
        self._load_model()
    
    def _load_model(self):
        """Load model configuration, weights, and tokenizer"""
        # Load config
        with open(os.path.join(self.model_path, "config.json"), "r") as f:
            self.config = json.load(f)
        
        # Load tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load weights
        self.weights = self._load_weights()
        print(f"✓ PyTorch reference model loaded")
    
    def _load_weights(self) -> Dict[str, torch.Tensor]:
        """Load model weights from safetensors files"""
        import safetensors
        
        weights = {}
        model_dir = Path(self.model_path)
        
        for file in sorted(model_dir.glob("*.safetensors")):
            data = safetensors.safe_open(file, "pt")
            for name in data.keys():
                weights[name] = data.get_tensor(name).to(dtype=torch.float16)
        
        return weights
    
    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """RMS Normalization"""
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return x * weight
    
    def _apply_rotary_emb(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary positional embedding"""
        # Simplified RoPE implementation
        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)
        
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
    
    def _attention_layer(self, x: torch.Tensor, layer_idx: int, position_cos: torch.Tensor, position_sin: torch.Tensor) -> torch.Tensor:
        """Single attention layer computation"""
        batch_size, seq_len, hidden_size = x.shape
        
        nh = self.config["num_attention_heads"]
        nkvh = self.config.get("num_key_value_heads", nh)
        head_dim = hidden_size // nh
        
        # Input layernorm
        attn_norm_weight = self.weights[f"model.layers.{layer_idx}.input_layernorm.weight"]
        x_norm = self._rms_norm(x, attn_norm_weight)
        
        # QKV projection
        q_weight = self.weights[f"model.layers.{layer_idx}.self_attn.q_proj.weight"]
        k_weight = self.weights[f"model.layers.{layer_idx}.self_attn.k_proj.weight"]
        v_weight = self.weights[f"model.layers.{layer_idx}.self_attn.v_proj.weight"]
        
        q = F.linear(x_norm, q_weight).view(batch_size, seq_len, nh, head_dim).transpose(1, 2)
        k = F.linear(x_norm, k_weight).view(batch_size, seq_len, nkvh, head_dim).transpose(1, 2)
        v = F.linear(x_norm, v_weight).view(batch_size, seq_len, nkvh, head_dim).transpose(1, 2)
        
        # Qwen3-specific: Q/K normalization (if available)
        q_norm_key = f"model.layers.{layer_idx}.self_attn.q_norm.weight"
        k_norm_key = f"model.layers.{layer_idx}.self_attn.k_norm.weight"
        
        if q_norm_key in self.weights and k_norm_key in self.weights:
            q_norm_weight = self.weights[q_norm_key]
            k_norm_weight = self.weights[k_norm_key]
            q = self._rms_norm(q, q_norm_weight)
            k = self._rms_norm(k, k_norm_weight)
        
        # Apply rotary embedding
        q, k = self._apply_rotary_emb(q, k, position_cos, position_sin)
        
        # Attention computation
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        
        # Causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        attn_weights = attn_weights.masked_fill(causal_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)
        o_weight = self.weights[f"model.layers.{layer_idx}.self_attn.o_proj.weight"]
        attn_output = F.linear(attn_output, o_weight)
        
        # Residual connection
        x = x + attn_output
        
        return x
    
    def _mlp_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """MLP layer computation"""
        # Post-attention normalization
        ffn_norm_weight = self.weights[f"model.layers.{layer_idx}.post_attention_layernorm.weight"]
        x_norm = self._rms_norm(x, ffn_norm_weight)
        
        # Gate and up projections
        gate_weight = self.weights[f"model.layers.{layer_idx}.mlp.gate_proj.weight"]
        up_weight = self.weights[f"model.layers.{layer_idx}.mlp.up_proj.weight"]
        down_weight = self.weights[f"model.layers.{layer_idx}.mlp.down_proj.weight"]
        
        gate = F.linear(x_norm, gate_weight)
        up = F.linear(x_norm, up_weight)
        
        # SwiGLU activation
        mlp_output = F.linear(F.silu(gate) * up, down_weight)
        
        # Residual connection
        x = x + mlp_output
        
        return x
    
    def _create_position_embeddings(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create rotary position embeddings"""
        head_dim = self.config["hidden_size"] // self.config["num_attention_heads"]
        
        # Create position indices
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        
        # Create frequency tensor
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        
        # Compute position embeddings
        freqs = torch.outer(position_ids.float().squeeze(), inv_freq)
        cos = torch.cos(freqs).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
        sin = torch.sin(freqs).unsqueeze(0).unsqueeze(0)
        
        # Expand to full head dimension
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
        
        return cos, sin
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model"""
        batch_size, seq_len = input_ids.shape
        
        # Embedding
        embed_weight = self.weights["model.embed_tokens.weight"]
        x = F.embedding(input_ids, embed_weight)
        
        # Position embeddings
        position_cos, position_sin = self._create_position_embeddings(seq_len)
        
        # Transformer layers
        num_layers = self.config["num_hidden_layers"]
        for layer_idx in range(num_layers):
            x = self._attention_layer(x, layer_idx, position_cos, position_sin)
            x = self._mlp_layer(x, layer_idx)
        
        # Final normalization
        final_norm_weight = self.weights["model.norm.weight"]
        x = self._rms_norm(x, final_norm_weight)
        
        # LM head
        lm_head_weight = self.weights["lm_head.weight"]
        logits = F.linear(x, lm_head_weight)
        
        return logits
    
    def generate(self, prompt: str, max_new_tokens: int = 50) -> Tuple[str, float]:
        """Generate text from prompt"""
        start_time = time.time()
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        generated_tokens = []
        current_ids = input_ids
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                logits = self.forward(current_ids)
                
                # Get next token
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Check for EOS
                if next_token.item() in [self.tokenizer.eos_token_id]:
                    break
                
                generated_tokens.append(next_token.item())
                current_ids = torch.cat([current_ids, next_token], dim=1)
        
        end_time = time.time()
        
        # Decode
        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return output_text, end_time - start_time


class Qwen3VerificationTool:
    """Main verification tool"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        
        # Load models
        self.cpp_model = None
        self.py_model = None
        self.tokenizer = None
        
        self._load_models()
    
    def _load_models(self):
        """Load both C++ and PyTorch models"""
        print("Loading models for verification...")
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print("✓ Tokenizer loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {e}")
        
        # Load C++ model
        if CPP_MODEL_AVAILABLE:
            try:
                self.cpp_model = CppQwen3Model(self.model_path, device_type=self.device)
                print("✓ C++ model loaded")
            except Exception as e:
                print(f"⚠ Failed to load C++ model: {e}")
        
        # Load PyTorch reference
        try:
            self.py_model = SimplePyTorchQwen3(self.model_path, device=self.device)
            print("✓ PyTorch reference model loaded")
        except Exception as e:
            print(f"⚠ Failed to load PyTorch reference model: {e}")
        
        if self.cpp_model is None and self.py_model is None:
            raise RuntimeError("No models could be loaded for verification")
    
    def compare_single_inference(self, input_text: str, max_tokens: int = 20) -> TestResult:
        """Compare single inference between C++ and PyTorch implementations"""
        print(f"Testing: '{input_text}'")
        
        cpp_output = None
        py_output = None
        cpp_tokens = None
        py_tokens = None
        cpp_time = 0.0
        py_time = 0.0
        error = None
        
        # Test C++ model
        if self.cpp_model:
            try:
                cpp_output, cpp_time = self.cpp_model.generate(input_text, max_steps=max_tokens)
                cpp_tokens = self.tokenizer.encode(cpp_output, add_special_tokens=False)
                print(f"  C++ output: '{cpp_output}' ({cpp_time:.3f}s)")
            except Exception as e:
                error = f"C++ inference failed: {e}"
                print(f"  C++ error: {error}")
        
        # Test PyTorch model
        if self.py_model:
            try:
                py_output, py_time = self.py_model.generate(input_text, max_new_tokens=max_tokens)
                py_tokens = self.tokenizer.encode(py_output, add_special_tokens=False)
                print(f"  PyTorch output: '{py_output}' ({py_time:.3f}s)")
            except Exception as e:
                py_error = f"PyTorch inference failed: {e}"
                error = error + "; " + py_error if error else py_error
                print(f"  PyTorch error: {py_error}")
        
        # Compute metrics
        token_accuracy = 0.0
        text_similarity = 0.0
        
        if cpp_tokens and py_tokens:
            # Token-level accuracy
            min_len = min(len(cpp_tokens), len(py_tokens))
            if min_len > 0:
                matches = sum(1 for i in range(min_len) if cpp_tokens[i] == py_tokens[i])
                token_accuracy = matches / min_len
        
        if cpp_output and py_output:
            # Simple text similarity (character-level)
            from difflib import SequenceMatcher
            text_similarity = SequenceMatcher(None, cpp_output, py_output).ratio()
        
        return TestResult(
            input_text=input_text,
            cpp_output=cpp_output,
            py_output=py_output,
            cpp_tokens=cpp_tokens,
            py_tokens=py_tokens,
            cpp_time=cpp_time,
            py_time=py_time,
            token_accuracy=token_accuracy,
            text_similarity=text_similarity,
            error=error
        )
    
    def run_verification_suite(self, test_inputs: List[str]) -> List[TestResult]:
        """Run comprehensive verification test suite"""
        print("="*80)
        print("QWEN3 C++ IMPLEMENTATION VERIFICATION")
        print("="*80)
        print(f"Model path: {self.model_path}")
        print(f"Device: {self.device}")
        print(f"Test cases: {len(test_inputs)}")
        print()
        
        results = []
        
        for i, input_text in enumerate(test_inputs):
            print(f"\n[{i+1}/{len(test_inputs)}] ", end="")
            result = self.compare_single_inference(input_text)
            results.append(result)
        
        self._generate_report(results)
        
        return results
    
    def _generate_report(self, results: List[TestResult]):
        """Generate verification report"""
        print("\n" + "="*80)
        print("VERIFICATION REPORT")
        print("="*80)
        
        successful_tests = [r for r in results if r.error is None]
        failed_tests = [r for r in results if r.error is not None]
        
        print(f"Total tests: {len(results)}")
        print(f"Successful: {len(successful_tests)}")
        print(f"Failed: {len(failed_tests)}")
        
        if successful_tests:
            avg_token_accuracy = np.mean([r.token_accuracy for r in successful_tests])
            avg_text_similarity = np.mean([r.text_similarity for r in successful_tests])
            avg_speedup = np.mean([r.py_time / r.cpp_time for r in successful_tests if r.cpp_time > 0])
            
            print(f"\nACCURACY METRICS:")
            print(f"  Average token accuracy: {avg_token_accuracy:.4f}")
            print(f"  Average text similarity: {avg_text_similarity:.4f}")
            print(f"  Average speedup (PyTorch/C++): {avg_speedup:.2f}x")
            
            # Detailed results
            print(f"\nDETAILED RESULTS:")
            for result in successful_tests:
                print(f"  Input: '{result.input_text}'")
                print(f"    Token accuracy: {result.token_accuracy:.4f}")
                print(f"    Text similarity: {result.text_similarity:.4f}")
                print(f"    C++ time: {result.cpp_time:.3f}s")
                print(f"    PyTorch time: {result.py_time:.3f}s")
                print()
        
        if failed_tests:
            print(f"FAILED TESTS:")
            for result in failed_tests:
                print(f"  '{result.input_text}': {result.error}")
        
        # Create visualization
        self._create_accuracy_plot(successful_tests)
    
    def _create_accuracy_plot(self, results: List[TestResult]):
        """Create accuracy visualization"""
        if not results:
            return
        
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Token accuracy plot
            token_accuracies = [r.token_accuracy for r in results]
            text_similarities = [r.text_similarity for r in results]
            test_names = [f"Test {i+1}" for i in range(len(results))]
            
            axes[0].bar(test_names, token_accuracies, alpha=0.7, color='blue')
            axes[0].set_title('Token-Level Accuracy')
            axes[0].set_ylabel('Accuracy')
            axes[0].set_ylim(0, 1)
            axes[0].tick_params(axis='x', rotation=45)
            
            # Text similarity plot
            axes[1].bar(test_names, text_similarities, alpha=0.7, color='green')
            axes[1].set_title('Text Similarity')
            axes[1].set_ylabel('Similarity')
            axes[1].set_ylim(0, 1)
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plot_file = '/tmp/qwen3_verification_results.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Verification plot saved to: {plot_file}")
            plt.close()
            
        except Exception as e:
            print(f"Failed to create verification plot: {e}")


def main():
    parser = argparse.ArgumentParser(description='Qwen3 C++ Implementation Verification Tool')
    parser.add_argument('model_path', help='Path to Qwen3 model directory')
    parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('--test-inputs', nargs='+', 
                       default=[
                           'Hello',
                           '你好',
                           'What is',
                           '山东最高的山',
                           '1 + 1 =',
                           'The capital of France',
                       ],
                       help='Test input strings')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model path does not exist: {args.model_path}")
        sys.exit(1)
    
    try:
        # Create verification tool
        verifier = Qwen3VerificationTool(args.model_path, args.device)
        
        # Run verification suite
        results = verifier.run_verification_suite(args.test_inputs)
        
        print(f"\n✓ Verification completed!")
        print("Results plot saved to /tmp/qwen3_verification_results.png")
        
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
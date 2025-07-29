#!/usr/bin/env python3
"""
Qwen3 Model Validation Script
This script compares the C++ implementation with the original PyTorch implementation
to ensure correctness of the inference results.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import transformers
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import traceback

# Import the custom Qwen3 implementation
from qwen3 import Qwen3ForCausalLM, Qwen3Meta, Qwen3Weights, Qwen3WeightsNaming, LlamaWeightsNaming
import safetensors

# Set default device
torch.set_default_device("cpu")

class PyTorchQwen3Reference:
    """Reference PyTorch implementation for comparison"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device("cpu")
        self._load_model()
    
    def _load_model(self):
        """Load the original PyTorch model"""
        print("Loading PyTorch reference model...")
        
        try:
            # Try to load with transformers AutoModel
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                torch_dtype=torch.float16,
                device_map="cpu",
                trust_remote_code=True
            )
            self.model.eval()
            print("✓ PyTorch model loaded successfully")
            
        except Exception as e:
            print(f"⚠ Failed to load with AutoModel: {e}")
            print("Attempting manual model construction...")
            
            # Fallback: manual model construction
            self._load_manual()
    
    def _load_manual(self):
        """Manually construct model from weights"""
        # Load config and weights manually
        with open(os.path.join(self.model_path, "config.json"), "r") as f:
            self.config = json.load(f)
        
        # Create tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        
        # For now, set model to None - this would need custom model implementation
        self.model = None
        print("⚠ Manual model construction not implemented - using tokenizer only")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to tokens"""
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": text}]
            inputs = self.tokenizer.apply_chat_template(
                messages, 
                return_tensors="pt", 
                add_generation_prompt=True
            )
            return inputs[0].tolist()
        else:
            return self.tokenizer.encode(text, return_tensors="pt")[0].tolist()
    
    def decode(self, tokens: List[int]) -> str:
        """Decode tokens to text"""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def generate_tokens(self, input_tokens: List[int], max_new_tokens: int = 50, 
                       temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9) -> List[int]:
        """Generate tokens using PyTorch model"""
        if self.model is None:
            print("⚠ PyTorch model not available, returning dummy tokens")
            return [self.tokenizer.eos_token_id] * max_new_tokens
        
        with torch.no_grad():
            input_tensor = torch.tensor([input_tokens], dtype=torch.long, device=self.device)
            
            # Generate with specified parameters
            outputs = self.model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract new tokens
            new_tokens = outputs[0][len(input_tokens):].tolist()
            return new_tokens
    
    def get_embeddings(self, input_tokens: List[int]) -> np.ndarray:
        """Get embedding layer outputs"""
        if self.model is None:
            return None
        
        with torch.no_grad():
            input_tensor = torch.tensor([input_tokens], dtype=torch.long, device=self.device)
            embeddings = self.model.model.embed_tokens(input_tensor)
            return embeddings.cpu().numpy()
    
    def get_layer_outputs(self, input_tokens: List[int], layer_idx: int) -> np.ndarray:
        """Get specific transformer layer outputs"""
        if self.model is None:
            return None
        
        # This would require hooking into the model's forward pass
        # For now, return None
        print(f"⚠ Layer output extraction not implemented for layer {layer_idx}")
        return None


class ValidationMetrics:
    """Compute various metrics for comparing model outputs"""
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two arrays"""
        a_flat = a.flatten()
        b_flat = b.flatten()
        return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat))
    
    @staticmethod
    def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Compute L2 distance between two arrays"""
        return np.linalg.norm(a.flatten() - b.flatten())
    
    @staticmethod
    def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
        """Compute maximum absolute difference"""
        return np.max(np.abs(a - b))
    
    @staticmethod
    def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
        """Compute mean absolute difference"""
        return np.mean(np.abs(a - b))
    
    @staticmethod
    def relative_error(a: np.ndarray, b: np.ndarray) -> float:
        """Compute relative error"""
        return np.mean(np.abs(a - b) / (np.abs(a) + 1e-8))
    
    @classmethod
    def compare_arrays(cls, a: np.ndarray, b: np.ndarray, name: str = "") -> Dict[str, float]:
        """Comprehensive comparison of two arrays"""
        if a is None or b is None:
            return {"error": "One or both arrays are None"}
        
        if a.shape != b.shape:
            return {"error": f"Shape mismatch: {a.shape} vs {b.shape}"}
        
        return {
            "cosine_similarity": cls.cosine_similarity(a, b),
            "l2_distance": cls.l2_distance(a, b),
            "max_abs_diff": cls.max_abs_diff(a, b),
            "mean_abs_diff": cls.mean_abs_diff(a, b),
            "relative_error": cls.relative_error(a, b)
        }


class Qwen3Validator:
    """Main validation class comparing C++ and PyTorch implementations"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        
        print("Initializing Qwen3 Validator...")
        
        # Load both implementations
        self.cpp_model = Qwen3ForCausalLM(model_path, device_type=device)
        self.pytorch_model = PyTorchQwen3Reference(model_path)
        
        print("✓ Both models loaded")
    
    def validate_tokenization(self, test_texts: List[str]) -> bool:
        """Validate that tokenization is consistent"""
        print("\n" + "="*60)
        print("TOKENIZATION VALIDATION")
        print("="*60)
        
        all_passed = True
        
        for text in test_texts:
            print(f"\nText: '{text}'")
            
            # Get tokens from both implementations
            cpp_tokens = self.cpp_model.tokenizer.encode(text)
            pytorch_tokens = self.pytorch_model.encode(text)
            
            print(f"C++:     {cpp_tokens}")
            print(f"PyTorch: {pytorch_tokens}")
            
            if cpp_tokens == pytorch_tokens:
                print("✓ Tokenization matches")
            else:
                print("✗ Tokenization differs!")
                all_passed = False
        
        return all_passed
    
    def validate_single_step_inference(self, input_tokens: List[int], 
                                     temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9) -> Dict[str, Any]:
        """Validate single step inference"""
        print(f"\n" + "="*60)
        print("SINGLE STEP INFERENCE VALIDATION")
        print("="*60)
        print(f"Input tokens: {input_tokens[:10]}{'...' if len(input_tokens) > 10 else ''}")
        print(f"Parameters: temperature={temperature}, top_k={top_k}, top_p={top_p}")
        
        results = {}
        
        # C++ implementation
        print("\nRunning C++ inference...")
        cpp_start = time.time()
        
        try:
            from infer_task import InferTask
            
            # Create task
            task = InferTask(
                id=0,
                tokens=input_tokens,
                max_tokens=self.cpp_model.max_context_len(),
                temperature=temperature,
                topk=top_k,
                topp=top_p,
                end_tokens=self.cpp_model.eos_token_id
            )
            
            # Bind KV cache
            kv_cache = self.cpp_model.create_kv_cache()
            task.bind_kvcache(kv_cache, 0)
            
            # Get one token
            outputs = self.cpp_model.batch_infer_one_round([task])
            cpp_token = outputs[0]
            cpp_time = time.time() - cpp_start
            
            # Clean up
            task.kvcache().drop(self.cpp_model)
            
            results['cpp_token'] = cpp_token
            results['cpp_time'] = cpp_time
            print(f"C++ output token: {cpp_token}")
            print(f"C++ time: {cpp_time*1000:.3f}ms")
            
        except Exception as e:
            print(f"✗ C++ inference failed: {e}")
            results['cpp_error'] = str(e)
            return results
        
        # PyTorch implementation
        print("\nRunning PyTorch inference...")
        pytorch_start = time.time()
        
        try:
            pytorch_tokens = self.pytorch_model.generate_tokens(
                input_tokens, max_new_tokens=1, 
                temperature=temperature, top_k=top_k, top_p=top_p
            )
            pytorch_token = pytorch_tokens[0] if pytorch_tokens else None
            pytorch_time = time.time() - pytorch_start
            
            results['pytorch_token'] = pytorch_token
            results['pytorch_time'] = pytorch_time
            print(f"PyTorch output token: {pytorch_token}")
            print(f"PyTorch time: {pytorch_time*1000:.3f}ms")
            
        except Exception as e:
            print(f"✗ PyTorch inference failed: {e}")
            results['pytorch_error'] = str(e)
            return results
        
        # Compare results
        if 'cpp_token' in results and 'pytorch_token' in results:
            if results['cpp_token'] == results['pytorch_token']:
                print("✓ Tokens match!")
                results['tokens_match'] = True
            else:
                print("✗ Tokens differ!")
                results['tokens_match'] = False
                
                # Decode both tokens for human readability
                cpp_text = self.cpp_model.tokenizer.decode([cpp_token])
                pytorch_text = self.pytorch_model.decode([pytorch_token])
                print(f"C++ decoded: '{cpp_text}'")
                print(f"PyTorch decoded: '{pytorch_text}'")
        
        return results
    
    def validate_generation(self, text: str, max_steps: int = 10) -> Dict[str, Any]:
        """Validate full generation sequence"""
        print(f"\n" + "="*60)
        print("GENERATION VALIDATION")
        print("="*60)
        print(f"Input text: '{text}'")
        print(f"Max steps: {max_steps}")
        
        results = {}
        
        # Get input tokens
        input_tokens = self.cpp_model.tokenizer.encode(text)
        print(f"Input tokens: {input_tokens}")
        
        # C++ generation
        print("\nC++ Generation:")
        cpp_start = time.time()
        try:
            cpp_output, cpp_avg_time = self.cpp_model.generate(text, max_steps)
            cpp_total_time = time.time() - cpp_start
            results['cpp_output'] = cpp_output
            results['cpp_time'] = cpp_total_time
            results['cpp_avg_step_time'] = cpp_avg_time
            print(f"Output: {cpp_output}")
            print(f"Total time: {cpp_total_time:.3f}s")
            print(f"Avg step time: {cpp_avg_time*1000:.3f}ms")
        except Exception as e:
            print(f"✗ C++ generation failed: {e}")
            results['cpp_error'] = str(e)
        
        # PyTorch generation
        print("\nPyTorch Generation:")
        pytorch_start = time.time()
        try:
            pytorch_tokens = self.pytorch_model.generate_tokens(
                input_tokens, max_new_tokens=max_steps
            )
            pytorch_output = self.pytorch_model.decode(pytorch_tokens)
            pytorch_total_time = time.time() - pytorch_start
            results['pytorch_output'] = pytorch_output
            results['pytorch_time'] = pytorch_total_time
            print(f"Output: {pytorch_output}")
            print(f"Total time: {pytorch_total_time:.3f}s")
        except Exception as e:
            print(f"✗ PyTorch generation failed: {e}")
            results['pytorch_error'] = str(e)
        
        # Compare outputs
        if 'cpp_output' in results and 'pytorch_output' in results:
            if results['cpp_output'] == results['pytorch_output']:
                print("✓ Outputs match exactly!")
                results['outputs_match'] = True
            else:
                print("✗ Outputs differ")
                results['outputs_match'] = False
                
                # Compute similarity metrics
                cpp_words = results['cpp_output'].split()
                pytorch_words = results['pytorch_output'].split()
                
                # Word-level overlap
                common_words = set(cpp_words) & set(pytorch_words)
                word_overlap = len(common_words) / max(len(cpp_words), len(pytorch_words), 1)
                results['word_overlap'] = word_overlap
                print(f"Word overlap: {word_overlap:.2%}")
        
        return results
    
    def validate_embeddings(self, input_tokens: List[int]) -> Dict[str, Any]:
        """Validate embedding layer outputs"""
        print(f"\n" + "="*60)
        print("EMBEDDING VALIDATION")
        print("="*60)
        
        results = {}
        
        # This would require extending the C++ API to expose embeddings
        print("⚠ Embedding validation requires additional C++ API exposure")
        print("  This feature is not yet implemented")
        
        # Placeholder for future implementation
        results['status'] = 'not_implemented'
        
        return results
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation tests"""
        print("="*80)
        print("COMPREHENSIVE QWEN3 VALIDATION")
        print("="*80)
        
        all_results = {}
        
        # Test cases
        test_texts = [
            "Hello",
            "What is the capital of France?",
            "山东最高的山是什么？",
            "请用Python写一个快速排序算法。"
        ]
        
        # 1. Tokenization validation
        try:
            tokenization_passed = self.validate_tokenization(test_texts)
            all_results['tokenization'] = {'passed': tokenization_passed}
        except Exception as e:
            print(f"✗ Tokenization validation failed: {e}")
            all_results['tokenization'] = {'error': str(e)}
        
        # 2. Single step inference validation
        for i, text in enumerate(test_texts[:2]):  # Test first 2 texts
            try:
                input_tokens = self.cpp_model.tokenizer.encode(text)
                step_results = self.validate_single_step_inference(input_tokens)
                all_results[f'single_step_{i}'] = step_results
            except Exception as e:
                print(f"✗ Single step validation {i} failed: {e}")
                all_results[f'single_step_{i}'] = {'error': str(e)}
        
        # 3. Generation validation
        for i, text in enumerate(test_texts[:2]):  # Test first 2 texts
            try:
                gen_results = self.validate_generation(text, max_steps=5)
                all_results[f'generation_{i}'] = gen_results
            except Exception as e:
                print(f"✗ Generation validation {i} failed: {e}")
                all_results[f'generation_{i}'] = {'error': str(e)}
        
        # 4. Embedding validation (placeholder)
        try:
            input_tokens = self.cpp_model.tokenizer.encode(test_texts[0])
            emb_results = self.validate_embeddings(input_tokens)
            all_results['embeddings'] = emb_results
        except Exception as e:
            print(f"✗ Embedding validation failed: {e}")
            all_results['embeddings'] = {'error': str(e)}
        
        # Summary
        self._print_validation_summary(all_results)
        
        return all_results
    
    def _print_validation_summary(self, results: Dict[str, Any]):
        """Print validation summary"""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        total_tests = 0
        passed_tests = 0
        
        for test_name, test_results in results.items():
            total_tests += 1
            
            if 'error' in test_results:
                print(f"✗ {test_name}: ERROR - {test_results['error']}")
            elif test_name == 'tokenization':
                if test_results.get('passed', False):
                    print(f"✓ {test_name}: PASSED")
                    passed_tests += 1
                else:
                    print(f"✗ {test_name}: FAILED")
            elif 'tokens_match' in test_results:
                if test_results['tokens_match']:
                    print(f"✓ {test_name}: TOKENS MATCH")
                    passed_tests += 1
                else:
                    print(f"✗ {test_name}: TOKENS DIFFER")
            elif 'outputs_match' in test_results:
                if test_results['outputs_match']:
                    print(f"✓ {test_name}: OUTPUTS MATCH")
                    passed_tests += 1
                else:
                    overlap = test_results.get('word_overlap', 0)
                    print(f"✗ {test_name}: OUTPUTS DIFFER (overlap: {overlap:.2%})")
            elif test_results.get('status') == 'not_implemented':
                print(f"⚠ {test_name}: NOT IMPLEMENTED")
                total_tests -= 1  # Don't count as failed
            else:
                print(f"? {test_name}: UNKNOWN STATUS")
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed ({passed_tests/max(total_tests,1)*100:.1f}%)")
        
        if passed_tests == total_tests:
            print("✓ All tests passed! The implementation appears to be correct.")
        else:
            print("✗ Some tests failed. Please review the differences.")


def main():
    """Main validation script"""
    if len(sys.argv) < 2:
        print("Usage: python qwen3_validation.py <model_path> [device]")
        print("\nExample:")
        print("  python qwen3_validation.py ./models/Qwen3-1.7B-Instruct")
        print("  python qwen3_validation.py ./models/Qwen3-1.7B-Instruct cpu")
        sys.exit(1)
    
    model_path = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "cpu"
    
    print(f"Qwen3 Model Validation")
    print(f"Model path: {model_path}")
    print(f"Device: {device}")
    print()
    
    if not os.path.exists(model_path):
        print(f"✗ Model path does not exist: {model_path}")
        sys.exit(1)
    
    try:
        # Create validator
        validator = Qwen3Validator(model_path, device)
        
        # Run comprehensive validation
        results = validator.run_comprehensive_validation()
        
        # Save results to file
        import json
        results_file = "qwen3_validation_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    json_results[key] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                                       for k, v in value.items()}
                else:
                    json_results[key] = value
            json.dump(json_results, f, indent=2)
        
        print(f"\n✓ Validation results saved to: {results_file}")
        
    except Exception as e:
        print(f"✗ Validation failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
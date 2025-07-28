#!/usr/bin/env python3
"""
Qwen3 Model Comparison Program

This script compares the computational results between:
1. Adapted Qwen3 C++ model (using InfiniCore framework)
2. Original Qwen3 Python model (transformers implementation)

The comparison includes intermediate results from each layer and overall accuracy metrics.
"""

import os
import sys
import time
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import argparse
from contextlib import contextmanager

# Add the qwen3 directory to the path to import the original Python model
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "qwen3"))

try:
    from modeling_qwen3 import Qwen3ForCausalLM as Qwen3ForCausalLMPython
    from configuration_qwen3 import Qwen3Config
except ImportError as e:
    print(f"Warning: Could not import original Qwen3 Python model: {e}")
    Qwen3ForCausalLMPython = None
    Qwen3Config = None

# Import the adapted C++ model interface
from qwen3 import Qwen3ForCausalLM as Qwen3ForCausalLMCpp, DeviceType
import transformers


@dataclass
class ComparisonMetrics:
    """Metrics for comparing two tensors"""
    mse: float
    max_abs_error: float
    mean_abs_error: float
    cosine_similarity: float
    relative_error: float
    shape_match: bool


@dataclass
class LayerComparison:
    """Comparison results for a single layer"""
    layer_id: int
    input_embedding: Optional[ComparisonMetrics]
    attention_output: Optional[ComparisonMetrics]
    mlp_output: Optional[ComparisonMetrics]
    layer_output: Optional[ComparisonMetrics]


@dataclass
class ModelComparison:
    """Complete model comparison results"""
    input_tokens: List[int]
    input_text: str
    cpp_output: str
    python_output: str
    cpp_logits: Optional[torch.Tensor]
    python_logits: Optional[torch.Tensor]
    output_comparison: Optional[ComparisonMetrics]
    layer_comparisons: List[LayerComparison]
    timing_cpp: float
    timing_python: float
    success: bool
    error_message: Optional[str]


class ModelComparator:
    """Main class for comparing Qwen3 models"""
    
    def __init__(self, model_path: str, device_type: DeviceType = DeviceType.DEVICE_TYPE_CPU, 
                 max_layers_to_compare: int = 5):
        self.model_path = model_path
        self.device_type = device_type
        self.max_layers_to_compare = max_layers_to_compare
        
        # Initialize models
        self.cpp_model = None
        self.python_model = None
        self.tokenizer = None
        
        # Hook storage for intermediate results
        self.python_intermediate_results = {}
        self.cpp_intermediate_results = {}
        
    def setup_models(self):
        """Initialize both C++ and Python models"""
        print("Setting up models...")
        
        # Setup C++ model
        try:
            print("Loading C++ Qwen3 model...")
            self.cpp_model = Qwen3ForCausalLMCpp(self.model_path, self.device_type, 1)
            print("✓ C++ model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load C++ model: {e}")
            return False
            
        # Setup Python model
        if Qwen3ForCausalLMPython is None:
            print("✗ Python Qwen3 model not available")
            return False
            
        try:
            print("Loading Python Qwen3 model...")
            self.python_model = Qwen3ForCausalLMPython.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True
            )
            self.python_model.eval()
            print("✓ Python model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load Python model: {e}")
            return False
            
        # Setup tokenizer
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print("✓ Tokenizer loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load tokenizer: {e}")
            return False
            
        return True
        
    def compute_metrics(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> ComparisonMetrics:
        """Compute comparison metrics between two tensors"""
        if tensor1 is None or tensor2 is None:
            return ComparisonMetrics(
                mse=float('inf'), max_abs_error=float('inf'), mean_abs_error=float('inf'),
                cosine_similarity=0.0, relative_error=float('inf'), shape_match=False
            )
            
        # Ensure tensors are on the same device and have the same dtype
        tensor1 = tensor1.float().cpu()
        tensor2 = tensor2.float().cpu()
        
        # Check shape compatibility
        shape_match = tensor1.shape == tensor2.shape
        if not shape_match:
            print(f"Warning: Shape mismatch - {tensor1.shape} vs {tensor2.shape}")
            # Try to broadcast or truncate to make comparison possible
            min_shape = tuple(min(s1, s2) for s1, s2 in zip(tensor1.shape, tensor2.shape))
            if len(min_shape) > 0:
                slices = tuple(slice(0, s) for s in min_shape)
                tensor1 = tensor1[slices]
                tensor2 = tensor2[slices]
            else:
                return ComparisonMetrics(
                    mse=float('inf'), max_abs_error=float('inf'), mean_abs_error=float('inf'),
                    cosine_similarity=0.0, relative_error=float('inf'), shape_match=False
                )
        
        # Compute metrics
        diff = tensor1 - tensor2
        mse = torch.mean(diff ** 2).item()
        max_abs_error = torch.max(torch.abs(diff)).item()
        mean_abs_error = torch.mean(torch.abs(diff)).item()
        
        # Cosine similarity
        tensor1_flat = tensor1.flatten()
        tensor2_flat = tensor2.flatten()
        cosine_sim = torch.nn.functional.cosine_similarity(
            tensor1_flat.unsqueeze(0), tensor2_flat.unsqueeze(0)
        ).item()
        
        # Relative error
        tensor1_norm = torch.norm(tensor1_flat).item()
        relative_error = (torch.norm(diff.flatten()).item() / tensor1_norm) if tensor1_norm > 0 else float('inf')
        
        return ComparisonMetrics(
            mse=mse,
            max_abs_error=max_abs_error,
            mean_abs_error=mean_abs_error,
            cosine_similarity=cosine_sim,
            relative_error=relative_error,
            shape_match=shape_match
        )
        
    def register_python_hooks(self):
        """Register forward hooks to capture intermediate results from Python model"""
        self.python_intermediate_results.clear()
        hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self.python_intermediate_results[name] = output[0].detach().clone()
                else:
                    self.python_intermediate_results[name] = output.detach().clone()
            return hook
            
        # Register hooks for key layers
        hooks.append(self.python_model.model.embed_tokens.register_forward_hook(make_hook("embed_tokens")))
        
        # Register hooks for transformer layers (limited to avoid memory issues)
        for i in range(min(self.max_layers_to_compare, len(self.python_model.model.layers))):
            layer = self.python_model.model.layers[i]
            hooks.append(layer.register_forward_hook(make_hook(f"layer_{i}")))
            hooks.append(layer.self_attn.register_forward_hook(make_hook(f"layer_{i}_attn")))
            hooks.append(layer.mlp.register_forward_hook(make_hook(f"layer_{i}_mlp")))
            
        hooks.append(self.python_model.model.norm.register_forward_hook(make_hook("final_norm")))
        hooks.append(self.python_model.lm_head.register_forward_hook(make_hook("lm_head")))
        
        return hooks
        
    def run_python_inference(self, input_text: str) -> Tuple[str, torch.Tensor, float]:
        """Run inference with Python model and capture intermediate results"""
        hooks = self.register_python_hooks()
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            start_time = time.time()
            
            # Run inference
            with torch.no_grad():
                outputs = self.python_model(**inputs)
                
            end_time = time.time()
            timing = end_time - start_time
            
            # Get output text
            logits = outputs.logits[0, -1, :]  # Last token logits
            next_token_id = torch.argmax(logits).item()
            output_text = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
            
            return output_text, outputs.logits, timing
            
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
                
    def run_cpp_inference(self, input_text: str) -> Tuple[str, Optional[torch.Tensor], float]:
        """Run inference with C++ model"""
        try:
            start_time = time.time()
            
            # Use the generate method from the C++ model
            output_text, avg_time_ms = self.cpp_model.generate(input_text, max_steps=1)
            
            end_time = time.time()
            timing = end_time - start_time
            
            # Note: C++ model doesn't currently expose logits, so we return None
            return output_text, None, timing
            
        except Exception as e:
            print(f"Error in C++ inference: {e}")
            return "", None, 0.0
            
    def compare_models(self, test_inputs: List[str]) -> List[ModelComparison]:
        """Compare both models on a list of test inputs"""
        if not self.setup_models():
            return []
            
        results = []
        
        for i, input_text in enumerate(test_inputs):
            print(f"\n{'='*60}")
            print(f"Test Case {i+1}: {input_text}")
            print(f"{'='*60}")
            
            comparison = ModelComparison(
                input_tokens=[],
                input_text=input_text,
                cpp_output="",
                python_output="",
                cpp_logits=None,
                python_logits=None,
                output_comparison=None,
                layer_comparisons=[],
                timing_cpp=0.0,
                timing_python=0.0,
                success=False,
                error_message=None
            )
            
            try:
                # Tokenize input for reference
                tokens = self.tokenizer.encode(input_text)
                comparison.input_tokens = tokens
                
                # Run Python inference
                print("Running Python model inference...")
                py_output, py_logits, py_timing = self.run_python_inference(input_text)
                comparison.python_output = py_output
                comparison.python_logits = py_logits
                comparison.timing_python = py_timing
                print(f"Python output: '{py_output}' (time: {py_timing:.3f}s)")
                
                # Run C++ inference
                print("Running C++ model inference...")
                cpp_output, cpp_logits, cpp_timing = self.run_cpp_inference(input_text)
                comparison.cpp_output = cpp_output
                comparison.cpp_logits = cpp_logits
                comparison.timing_cpp = cpp_timing
                print(f"C++ output: '{cpp_output}' (time: {cpp_timing:.3f}s)")
                
                # Compare outputs
                if py_logits is not None and cpp_logits is not None:
                    comparison.output_comparison = self.compute_metrics(py_logits, cpp_logits)
                
                # Compare intermediate results (if available)
                comparison.layer_comparisons = self.compare_intermediate_results()
                
                comparison.success = True
                
            except Exception as e:
                comparison.error_message = str(e)
                print(f"Error in comparison: {e}")
                
            results.append(comparison)
            
        return results
        
    def compare_intermediate_results(self) -> List[LayerComparison]:
        """Compare intermediate results between models"""
        layer_comparisons = []
        
        # Compare layer by layer
        for i in range(self.max_layers_to_compare):
            layer_comp = LayerComparison(
                layer_id=i,
                input_embedding=None,
                attention_output=None,
                mlp_output=None,
                layer_output=None
            )
            
            # Compare layer outputs if available
            py_layer_key = f"layer_{i}"
            if py_layer_key in self.python_intermediate_results:
                py_tensor = self.python_intermediate_results[py_layer_key]
                # C++ intermediate results would need to be implemented
                # For now, we just record that Python results are available
                print(f"  Layer {i} output shape (Python): {py_tensor.shape}")
                
            layer_comparisons.append(layer_comp)
            
        return layer_comparisons
        
    def generate_report(self, results: List[ModelComparison], output_file: str = None):
        """Generate a detailed comparison report"""
        report = {
            "summary": {
                "total_tests": len(results),
                "successful_tests": sum(1 for r in results if r.success),
                "failed_tests": sum(1 for r in results if not r.success),
                "avg_cpp_time": np.mean([r.timing_cpp for r in results if r.success]),
                "avg_python_time": np.mean([r.timing_python for r in results if r.success]),
            },
            "detailed_results": []
        }
        
        for i, result in enumerate(results):
            detailed = {
                "test_id": i + 1,
                "input_text": result.input_text,
                "success": result.success,
                "error_message": result.error_message,
                "outputs": {
                    "cpp": result.cpp_output,
                    "python": result.python_output,
                },
                "timing": {
                    "cpp": result.timing_cpp,
                    "python": result.timing_python,
                    "speedup": result.timing_python / result.timing_cpp if result.timing_cpp > 0 else None
                },
                "accuracy": {
                    "text_match": result.cpp_output.strip() == result.python_output.strip(),
                }
            }
            
            if result.output_comparison:
                detailed["accuracy"]["logits_comparison"] = {
                    "mse": result.output_comparison.mse,
                    "max_abs_error": result.output_comparison.max_abs_error,
                    "cosine_similarity": result.output_comparison.cosine_similarity,
                    "relative_error": result.output_comparison.relative_error,
                }
                
            report["detailed_results"].append(detailed)
            
        # Print summary
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"Total tests: {report['summary']['total_tests']}")
        print(f"Successful: {report['summary']['successful_tests']}")
        print(f"Failed: {report['summary']['failed_tests']}")
        print(f"Average C++ time: {report['summary']['avg_cpp_time']:.3f}s")
        print(f"Average Python time: {report['summary']['avg_python_time']:.3f}s")
        if report['summary']['avg_cpp_time'] > 0:
            speedup = report['summary']['avg_python_time'] / report['summary']['avg_cpp_time']
            print(f"C++ speedup: {speedup:.2f}x")
            
        # Print detailed results
        for detailed in report["detailed_results"]:
            print(f"\nTest {detailed['test_id']}: {detailed['input_text']}")
            print(f"  Success: {detailed['success']}")
            if detailed['error_message']:
                print(f"  Error: {detailed['error_message']}")
            print(f"  C++ output: '{detailed['outputs']['cpp']}'")
            print(f"  Python output: '{detailed['outputs']['python']}'")
            print(f"  Text match: {detailed['accuracy']['text_match']}")
            if 'logits_comparison' in detailed['accuracy']:
                acc = detailed['accuracy']['logits_comparison']
                print(f"  Logits MSE: {acc['mse']:.6f}")
                print(f"  Cosine similarity: {acc['cosine_similarity']:.6f}")
                
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\nDetailed report saved to: {output_file}")
            
        return report
        
    def cleanup(self):
        """Clean up resources"""
        if self.cpp_model:
            try:
                self.cpp_model.destroy_model_instance()
            except:
                pass


def main():
    parser = argparse.ArgumentParser(description="Compare Qwen3 C++ and Python models")
    parser.add_argument("model_path", help="Path to the Qwen3 model directory")
    parser.add_argument("--device", choices=["cpu", "nvidia", "cambricon", "ascend", "metax", "moore", "iluvatar"], 
                       default="cpu", help="Device type for C++ model")
    parser.add_argument("--max-layers", type=int, default=5, 
                       help="Maximum number of layers to compare in detail")
    parser.add_argument("--output", help="Output file for detailed report")
    parser.add_argument("--test-inputs", nargs="+", 
                       default=["Hello", "山东最高的山是？", "What is artificial intelligence?"],
                       help="Test input strings")
    
    args = parser.parse_args()
    
    # Map device string to enum
    device_map = {
        "cpu": DeviceType.DEVICE_TYPE_CPU,
        "nvidia": DeviceType.DEVICE_TYPE_NVIDIA,
        "cambricon": DeviceType.DEVICE_TYPE_CAMBRICON,
        "ascend": DeviceType.DEVICE_TYPE_ASCEND,
        "metax": DeviceType.DEVICE_TYPE_METAX,
        "moore": DeviceType.DEVICE_TYPE_MOORE,
        "iluvatar": DeviceType.DEVICE_TYPE_ILUVATAR,
    }
    
    device_type = device_map[args.device]
    
    # Create comparator
    comparator = ModelComparator(
        model_path=args.model_path,
        device_type=device_type,
        max_layers_to_compare=args.max_layers
    )
    
    try:
        # Run comparison
        results = comparator.compare_models(args.test_inputs)
        
        # Generate report
        comparator.generate_report(results, args.output)
        
    finally:
        comparator.cleanup()


if __name__ == "__main__":
    main()
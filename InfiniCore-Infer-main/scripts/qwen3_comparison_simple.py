#!/usr/bin/env python3
"""
Simple Qwen3 Model Comparison Program

Compares the text generation outputs and performance between:
1. Adapted Qwen3 C++ model (using InfiniCore framework)  
2. Original Qwen3 Python model (transformers implementation)

This version focuses on end-to-end comparison that can run immediately.
"""

import os
import sys
import time
import json
import torch
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Add paths for imports
current_dir = Path(__file__).parent
repo_root = current_dir.parent.parent
sys.path.insert(0, str(repo_root / "qwen3"))
sys.path.insert(0, str(current_dir))

try:
    from modeling_qwen3 import Qwen3ForCausalLM as Qwen3Python
    from configuration_qwen3 import Qwen3Config
    PYTHON_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import original Qwen3 Python model: {e}")
    PYTHON_MODEL_AVAILABLE = False

from qwen3 import Qwen3ForCausalLM as Qwen3Cpp, DeviceType
import transformers


@dataclass 
class ComparisonResult:
    """Results from comparing both models"""
    test_input: str
    cpp_output: str
    python_output: str
    cpp_time: float
    python_time: float
    text_similarity: float  # Simple text similarity score
    success: bool
    error_msg: str = ""


class Qwen3Comparator:
    """Compare Qwen3 C++ and Python implementations"""
    
    def __init__(self, model_path: str, device_type: str = "cpu"):
        self.model_path = model_path
        self.device_type = self._parse_device_type(device_type)
        self.cpp_model = None
        self.python_model = None
        self.tokenizer = None
        
    def _parse_device_type(self, device_str: str) -> DeviceType:
        """Convert device string to DeviceType enum"""
        device_map = {
            "cpu": DeviceType.DEVICE_TYPE_CPU,
            "nvidia": DeviceType.DEVICE_TYPE_NVIDIA,
            "cambricon": DeviceType.DEVICE_TYPE_CAMBRICON,
            "ascend": DeviceType.DEVICE_TYPE_ASCEND,
            "metax": DeviceType.DEVICE_TYPE_METAX,
            "moore": DeviceType.DEVICE_TYPE_MOORE,
            "iluvatar": DeviceType.DEVICE_TYPE_ILUVATAR,
        }
        return device_map.get(device_str, DeviceType.DEVICE_TYPE_CPU)
        
    def setup_models(self) -> bool:
        """Initialize both models and tokenizer"""
        success = True
        
        # Setup tokenizer first
        try:
            print("Loading tokenizer...")
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print("✓ Tokenizer loaded")
        except Exception as e:
            print(f"✗ Failed to load tokenizer: {e}")
            return False
            
        # Setup C++ model
        try:
            print("Loading C++ Qwen3 model...")
            self.cpp_model = Qwen3Cpp(self.model_path, self.device_type, 1)
            print("✓ C++ model loaded")
        except Exception as e:
            print(f"✗ Failed to load C++ model: {e}")
            success = False
            
        # Setup Python model  
        if PYTHON_MODEL_AVAILABLE:
            try:
                print("Loading Python Qwen3 model...")
                self.python_model = Qwen3Python.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                self.python_model.eval()
                print("✓ Python model loaded")
            except Exception as e:
                print(f"✗ Failed to load Python model: {e}")
                self.python_model = None
        else:
            print("✗ Python model not available")
            self.python_model = None
            
        return success and (self.cpp_model is not None)
        
    def run_cpp_inference(self, prompt: str, max_tokens: int = 50) -> Tuple[str, float]:
        """Run inference with C++ model"""
        if self.cpp_model is None:
            return "", 0.0
            
        try:
            start_time = time.time()
            
            # Format prompt for chat
            formatted_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False
            )
            
            # Generate response
            output, avg_step_time = self.cpp_model.generate(formatted_prompt, max_tokens)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            return output, total_time
            
        except Exception as e:
            print(f"C++ inference error: {e}")
            return f"Error: {e}", 0.0
            
    def run_python_inference(self, prompt: str, max_tokens: int = 50) -> Tuple[str, float]:
        """Run inference with Python model"""
        if self.python_model is None:
            return "Python model not available", 0.0
            
        try:
            start_time = time.time()
            
            # Format prompt for chat
            formatted_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False
            )
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            )
            
            # Generate
            with torch.no_grad():
                outputs = self.python_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract generated part only
            if formatted_prompt in response:
                generated = response[len(formatted_prompt):].strip()
            else:
                generated = response
                
            end_time = time.time()
            total_time = end_time - start_time
            
            return generated, total_time
            
        except Exception as e:
            print(f"Python inference error: {e}")
            return f"Error: {e}", 0.0
            
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity score"""
        if not text1 or not text2:
            return 0.0
            
        # Simple token-based similarity
        tokens1 = text1.lower().split()
        tokens2 = text2.lower().split()
        
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
            
        # Calculate Jaccard similarity
        set1, set2 = set(tokens1), set(tokens2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
        
    def compare_single_input(self, test_input: str, max_tokens: int = 50) -> ComparisonResult:
        """Compare both models on a single input"""
        print(f"\nTesting: '{test_input}'")
        print("-" * 60)
        
        result = ComparisonResult(
            test_input=test_input,
            cpp_output="",
            python_output="", 
            cpp_time=0.0,
            python_time=0.0,
            text_similarity=0.0,
            success=False
        )
        
        try:
            # Run C++ inference
            print("Running C++ model...")
            cpp_output, cpp_time = self.run_cpp_inference(test_input, max_tokens)
            result.cpp_output = cpp_output
            result.cpp_time = cpp_time
            print(f"C++ output: '{cpp_output}' ({cpp_time:.3f}s)")
            
            # Run Python inference  
            print("Running Python model...")
            python_output, python_time = self.run_python_inference(test_input, max_tokens)
            result.python_output = python_output
            result.python_time = python_time
            print(f"Python output: '{python_output}' ({python_time:.3f}s)")
            
            # Calculate similarity
            similarity = self.calculate_text_similarity(cpp_output, python_output)
            result.text_similarity = similarity
            print(f"Text similarity: {similarity:.3f}")
            
            result.success = True
            
        except Exception as e:
            result.error_msg = str(e)
            print(f"Comparison error: {e}")
            
        return result
        
    def run_comparison_suite(self, test_inputs: List[str], max_tokens: int = 50) -> List[ComparisonResult]:
        """Run comparison on multiple test inputs"""
        if not self.setup_models():
            print("Failed to setup models, aborting comparison")
            return []
            
        results = []
        
        print(f"\n{'='*80}")
        print("QWEN3 MODEL COMPARISON")
        print(f"{'='*80}")
        print(f"Model path: {self.model_path}")
        print(f"Device: {self.device_type}")
        print(f"Test cases: {len(test_inputs)}")
        print(f"Max tokens per test: {max_tokens}")
        
        for i, test_input in enumerate(test_inputs, 1):
            print(f"\n[{i}/{len(test_inputs)}]", end="")
            result = self.compare_single_input(test_input, max_tokens)
            results.append(result)
            
        return results
        
    def generate_report(self, results: List[ComparisonResult], output_file: str = None):
        """Generate and display comparison report"""
        if not results:
            print("No results to report")
            return
            
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        successful_results = [r for r in results if r.success]
        
        # Summary statistics
        total_tests = len(results)
        successful_tests = len(successful_results)
        failed_tests = total_tests - successful_tests
        
        print(f"Total tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {failed_tests}")
        
        if successful_results:
            avg_cpp_time = sum(r.cpp_time for r in successful_results) / len(successful_results)
            avg_python_time = sum(r.python_time for r in successful_results) / len(successful_results)
            avg_similarity = sum(r.text_similarity for r in successful_results) / len(successful_results)
            
            print(f"\nPerformance:")
            print(f"  Average C++ time: {avg_cpp_time:.3f}s")
            print(f"  Average Python time: {avg_python_time:.3f}s")
            if avg_cpp_time > 0:
                speedup = avg_python_time / avg_cpp_time
                print(f"  C++ speedup: {speedup:.2f}x")
                
            print(f"\nAccuracy:")
            print(f"  Average text similarity: {avg_similarity:.3f}")
            
        # Detailed results
        print(f"\n{'='*80}")
        print("DETAILED RESULTS")
        print(f"{'='*80}")
        
        for i, result in enumerate(results, 1):
            print(f"\nTest {i}: {result.test_input}")
            print(f"  Success: {result.success}")
            if result.error_msg:
                print(f"  Error: {result.error_msg}")
            else:
                print(f"  C++ output: '{result.cpp_output}'")
                print(f"  Python output: '{result.python_output}'")
                print(f"  Similarity: {result.text_similarity:.3f}")
                print(f"  Timing - C++: {result.cpp_time:.3f}s, Python: {result.python_time:.3f}s")
                
        # Save to file if requested
        if output_file:
            report_data = {
                "summary": {
                    "total_tests": total_tests,
                    "successful_tests": successful_tests,
                    "failed_tests": failed_tests,
                    "avg_cpp_time": avg_cpp_time if successful_results else 0,
                    "avg_python_time": avg_python_time if successful_results else 0,
                    "avg_similarity": avg_similarity if successful_results else 0,
                },
                "results": [
                    {
                        "test_input": r.test_input,
                        "cpp_output": r.cpp_output,
                        "python_output": r.python_output,
                        "cpp_time": r.cpp_time,
                        "python_time": r.python_time,
                        "text_similarity": r.text_similarity,
                        "success": r.success,
                        "error_msg": r.error_msg
                    }
                    for r in results
                ]
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            print(f"\nDetailed report saved to: {output_file}")
            
    def cleanup(self):
        """Clean up model resources"""
        if self.cpp_model:
            try:
                self.cpp_model.destroy_model_instance()
                print("C++ model cleaned up")
            except:
                pass


def main():
    parser = argparse.ArgumentParser(description="Compare Qwen3 C++ and Python models")
    parser.add_argument("model_path", help="Path to Qwen3 model directory")
    parser.add_argument("--device", choices=["cpu", "nvidia", "cambricon", "ascend", "metax", "moore", "iluvatar"], 
                       default="cpu", help="Device type for C++ model")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate per test")
    parser.add_argument("--output", help="Output JSON file for detailed results")
    parser.add_argument("--test-inputs", nargs="+", 
                       default=["Hello", "山东最高的山是？", "What is AI?", "请介绍一下自己", "1+1=?"],
                       help="Test input strings")
    
    args = parser.parse_args()
    
    # Create comparator
    comparator = Qwen3Comparator(args.model_path, args.device)
    
    try:
        # Run comparison
        results = comparator.run_comparison_suite(args.test_inputs, args.max_tokens)
        
        # Generate report
        comparator.generate_report(results, args.output)
        
    except KeyboardInterrupt:
        print("\nComparison interrupted by user")
    except Exception as e:
        print(f"Comparison failed: {e}")
    finally:
        comparator.cleanup()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Demo script for Qwen3-1.7B model usage with InfiniCore-Infer

This script demonstrates how to use the adapted Qwen3 model.
Note: Requires compiled InfiniCore libraries and actual Qwen3 model files.
"""

import sys
import os
import argparse

def demo_qwen3_inference():
    """Demonstrate Qwen3 model inference"""
    print("=== Qwen3-1.7B InfiniCore-Infer Demo ===\n")
    
    # Add the scripts directory to path
    scripts_path = "/home/runner/work/copilot-test/copilot-test/InfiniCore-Infer-main/scripts"
    sys.path.append(scripts_path)
    
    try:
        from jiuge import JiugeForCauslLM
        from libinfinicore_infer import DeviceType
        print("✓ Successfully imported InfiniCore-Infer modules")
    except Exception as e:
        print(f"✗ Failed to import modules: {e}")
        print("Note: This requires compiled InfiniCore libraries")
        return False
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Demo Qwen3 inference")
    parser.add_argument("--model-path", required=True, help="Path to Qwen3 model directory")
    parser.add_argument("--device", choices=["cpu", "nvidia", "ascend"], default="cpu")
    parser.add_argument("--ndev", type=int, default=1, help="Number of devices")
    parser.add_argument("--prompt", default="山东最高的山是？", help="Input prompt")
    
    args = parser.parse_args()
    
    # Map device type
    device_map = {
        "cpu": DeviceType.DEVICE_TYPE_CPU,
        "nvidia": DeviceType.DEVICE_TYPE_NVIDIA, 
        "ascend": DeviceType.DEVICE_TYPE_ASCEND,
    }
    device_type = device_map[args.device]
    
    try:
        print(f"Loading Qwen3 model from: {args.model_path}")
        print(f"Device: {args.device}, Number of devices: {args.ndev}")
        
        # Load model
        model = JiugeForCauslLM(args.model_path, device_type, args.ndev)
        print("✓ Model loaded successfully")
        
        # Test inference
        print(f"\nInput: {args.prompt}")
        print("Output: ", end="", flush=True)
        
        output, avg_time = model.generate(args.prompt, max_steps=100)
        
        print(f"\n\nInference completed:")
        print(f"Average time per token: {avg_time:.2f}ms")
        print(f"Total output length: {len(output)} characters")
        
        # Cleanup
        model.destroy_model_instance()
        print("✓ Model cleaned up successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during inference: {e}")
        return False

def demo_server_launch():
    """Demonstrate launching the OpenAI-compatible server"""
    print("\n=== Server Demo ===")
    
    print("To launch the Qwen3 inference server:")
    print("python launch_server.py --model-path /path/to/qwen3-model --dev cpu --ndev 1")
    
    print("\nTest the server with curl:")
    print("""curl -N -H "Content-Type: application/json" \\
     -X POST http://127.0.0.1:8000/chat/completions \\
     -d '{
       "model": "qwen3",
       "messages": [{"role": "user", "content": "Hello, how are you?"}],
       "temperature": 1.0,
       "top_k": 50,
       "top_p": 0.8,
       "max_tokens": 512,
       "stream": true
     }'""")

def show_requirements():
    """Show setup requirements"""
    print("=== Setup Requirements ===\n")
    
    print("1. Environment Setup:")
    print("   export INFINI_ROOT=$HOME/.infini")
    print("   # Ensure InfiniCore is built and installed")
    
    print("\n2. Python Dependencies:")
    print("   pip install torch transformers safetensors fastapi uvicorn")
    
    print("\n3. Model Files:")
    print("   Download Qwen3-1.7B model with:")
    print("   - config.json (with model_type: 'qwen3')")
    print("   - model weights in safetensors format")
    print("   - tokenizer files")
    
    print("\n4. Build InfiniCore-Infer:")
    print("   cd InfiniCore-main && python scripts/install.py")
    print("   cd InfiniCore-Infer-main && xmake && xmake install")

def main():
    if len(sys.argv) == 1:
        print("Qwen3-1.7B InfiniCore-Infer Integration Demo")
        print("\nUsage options:")
        print("  --help: Show this help")
        print("  --requirements: Show setup requirements")
        print("  --server-demo: Show server launch instructions")
        print("  --model-path <path>: Run inference demo")
        return
    
    if "--help" in sys.argv:
        print(__doc__)
        return
        
    if "--requirements" in sys.argv:
        show_requirements()
        return
        
    if "--server-demo" in sys.argv:
        demo_server_launch()
        return
    
    # If model-path provided, run inference demo
    if "--model-path" in sys.argv:
        demo_qwen3_inference()
    else:
        print("Please specify --model-path or see --help for options")

if __name__ == "__main__":
    main()
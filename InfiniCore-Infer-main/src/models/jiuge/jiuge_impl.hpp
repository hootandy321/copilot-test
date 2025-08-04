/*
 * Jiuge Model Implementation Header
 * 
 * This header defines the core data structures and interfaces for the Jiuge
 * transformer model implementation using InfiniCore. It provides:
 * 
 * - Device resource management for distributed inference
 * - Thread synchronization primitives for asynchronous execution
 * - Request handling structures for batch processing
 * - KV cache management for efficient autoregressive generation
 * 
 * The design supports tensor parallelism across multiple devices with
 * efficient memory management and thread-safe operation.
 */

#ifndef JIUGE_IMPL_H
#define JIUGE_IMPL_H

#include "infinicore_infer.h"

#include "../../allocator.hpp"
#include "../../tensor.hpp"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

/*
 * Device Resource Structure
 * 
 * Contains all resources needed for inference on a single device in a distributed setup.
 * Each device in the cluster has its own DeviceResource instance containing:
 * 
 * 1. Device Context: InfiniCore device handle and operation context
 * 2. Model Weights: Device-specific weight tensor slices for tensor parallelism
 * 3. Execution Context: Streams and communicators for async operations
 * 4. Memory Management: Pools for efficient buffer allocation
 * 
 * Memory Layout for Distributed Inference:
 * - Global tensors (embeddings, norms): replicated on all devices
 * - Attention weights: partitioned by attention heads across devices
 * - FFN weights: partitioned by intermediate dimension across devices
 * - KV caches: partitioned by attention heads across devices
 */
struct DeviceResource {
    // ================================
    // Device Context and Handles
    // ================================
    infiniDevice_t device;           // InfiniCore device type (GPU/CPU)
    int device_id;                   // Physical device ID for this resource
    infiniopHandle_t handle;         // InfiniCore operation handle for compute ops
    
    // ================================
    // Model Weight Tensors
    // ================================
    // Global model tensors (replicated across all devices)
    std::shared_ptr<Tensor> w_in_embd;   // Input embedding table [dvoc, d]
    std::shared_ptr<Tensor> w_out_norm;  // Final layer normalization [d]
    std::shared_ptr<Tensor> w_out_embd;  // Output embedding/LM head [d, dvoc]
    std::shared_ptr<Tensor> sin_table;   // RoPE sine table [dctx, dh/2]
    std::shared_ptr<Tensor> cos_table;   // RoPE cosine table [dctx, dh/2]
    
    // Per-layer weight tensors (distributed across devices)
    std::vector<std::shared_ptr<Tensor>> w_attn_norm;   // Attention norm [d] per layer
    std::vector<std::shared_ptr<Tensor>> w_attn_qkv;    // QKV weights [d, (nh+2*nkvh)/ndev*dh] per layer
    std::vector<std::shared_ptr<Tensor>> b_attn_qkv;    // QKV bias [(nh+2*nkvh)/ndev*dh] per layer (optional)
    std::vector<std::shared_ptr<Tensor>> w_attn_out;    // Attention out [nh/ndev*dh, d] per layer
    std::vector<std::shared_ptr<Tensor>> w_ffn_norm;    // FFN norm [d] per layer
    std::vector<std::shared_ptr<Tensor>> w_ffn_gate_up; // FFN gate&up [d, 2*di/ndev] per layer
    std::vector<std::shared_ptr<Tensor>> w_ffn_down;    // FFN down [di/ndev, d] per layer
    
    // ================================
    // Execution and Communication
    // ================================
    infinirtStream_t stream;         // Execution stream for asynchronous operations
    infinicclComm_t comm;           // Inter-device communicator for distributed ops
    
    // ================================
    // Memory Management
    // ================================
    std::shared_ptr<MemoryPool> memory_pool;  // Pool for temporary buffer allocation
};

/*
 * Inference State Structure
 * 
 * Manages thread synchronization and lifecycle for device worker threads.
 * Each device has an associated InferState for coordinating asynchronous inference.
 * 
 * Thread Synchronization Pattern:
 * 1. Main thread waits on cv_load until loaded=true (device initialization complete)
 * 2. Main thread sets proceed=true and notifies cv_start to trigger inference
 * 3. Worker thread processes inference and sets proceed=false, notifies cv_done
 * 4. Main thread waits on cv_done until proceed=false (inference complete)
 * 5. For shutdown: main thread sets exit_flag=true and notifies cv_start
 * 
 * This design enables efficient pipeline parallelism and device coordination.
 */
struct InferState {
    // ================================
    // Thread Synchronization Primitives
    // ================================
    std::mutex mtx;                          // Mutex for protecting shared state
    std::condition_variable cv_load;         // Signals when device initialization is complete
    std::condition_variable cv_start;        // Signals when new inference request is available  
    std::condition_variable cv_done;         // Signals when inference processing is complete
    
    // ================================
    // Thread State Flags
    // ================================
    bool loaded = false;                     // True when device resources are initialized
    bool proceed = false;                    // True when inference should start/is in progress
    bool exit_flag = false;                  // True when worker thread should exit
};

/*
 * Inference Request Structure
 * 
 * Contains all data needed for a batch inference request across multiple devices.
 * This structure is shared between the main thread and all worker threads,
 * allowing efficient batch processing without data duplication.
 * 
 * Batch Organization:
 * - tokens: Concatenated token sequences from all requests [ntok total]
 * - req_lens: Length of each individual request [nreq]
 * - req_pos: Starting position in KV cache for each request [nreq]
 * - kv_caches: Separate KV cache for each request [nreq]
 * 
 * Sampling Parameters:
 * - temperature: Controls randomness (higher = more random) [nreq]
 * - topk: Keep only top-k tokens for sampling [nreq]
 * - topp: Nucleus sampling threshold (cumulative probability) [nreq]
 */
struct InferRequest {
    // ================================
    // Input Token Data
    // ================================
    const uint32_t *tokens;        // Concatenated input token IDs [ntok]
    uint32_t ntok;                // Total number of tokens across all requests
    
    // ================================
    // Request Batch Information  
    // ================================
    const uint32_t *req_lens;     // Length of each request [nreq]
    uint32_t nreq;               // Number of requests in this batch
    const uint32_t *req_pos;     // Starting position for each request in KV cache [nreq]
    
    // ================================
    // KV Cache Management
    // ================================
    struct KVCache **kv_caches;  // KV cache storage for each request [nreq]
    
    // ================================
    // Sampling Configuration
    // ================================
    const float *temperature;    // Temperature scaling for each request [nreq]
    const uint32_t *topk;       // Top-k filtering for each request [nreq]
    const float *topp;          // Top-p (nucleus) sampling for each request [nreq]
    
    // ================================
    // Output Data
    // ================================
    uint32_t *output;           // Generated token IDs for each request [nreq]
};

/*
 * JiugeModel Main Structure
 * 
 * Top-level model class that orchestrates distributed transformer inference.
 * Manages multiple devices, worker threads, and coordinates batch processing.
 * 
 * Architecture:
 * - One worker thread per device for parallel execution
 * - Shared request structure for efficient batch processing  
 * - Device resources partitioned for tensor parallelism
 * - Thread-safe coordination via InferState synchronization
 * 
 * Distributed Inference Strategy:
 * - Model weights are partitioned across devices by attention heads and FFN dimensions
 * - Each device processes the full sequence but only a slice of the model parameters
 * - All-reduce operations synchronize results across devices after attention and FFN
 * - Only device 0 performs final output generation and token sampling
 */
struct JiugeModel {
    // ================================
    // Model Configuration
    // ================================
    JiugeMeta meta;                           // Model architecture metadata
    infiniDevice_t device;                    // Device type (GPU/CPU)
    std::vector<int> dev_ids;                 // Physical device IDs for distributed inference
    
    // ================================
    // Distributed Resources
    // ================================
    std::vector<DeviceResource> dev_resources; // Per-device resources and weights
    std::vector<InferState> states;           // Per-device synchronization state
    std::vector<std::thread> threads;         // Worker threads for async processing
    
    // ================================
    // Shared Request Processing
    // ================================
    InferRequest req;                         // Shared request data across all devices

    // ================================
    // Constructor
    // ================================
    /*
     * Initialize distributed model with tensor parallelism
     * 
     * Parameters:
     * - meta: Model architecture (layers, dimensions, data types)
     * - weights: Model weight tensors 
     * - device: InfiniCore device type
     * - device_ids: Physical device IDs for distributed inference
     * 
     * Initialization Process:
     * 1. Create device resources and partition weights across devices
     * 2. Initialize InfiniCCL communication for multi-device coordination
     * 3. Launch worker threads and wait for device initialization
     * 4. Set up synchronization primitives for inference coordination
     */
    JiugeModel(const JiugeMeta *, const JiugeWeights *, infiniDevice_t device, std::vector<int> device_ids);
};

/*
 * KV Cache Structure
 * 
 * Manages key-value cache storage for efficient autoregressive generation.
 * The cache stores past key and value vectors to avoid recomputing attention
 * for previously generated tokens.
 * 
 * Cache Organization:
 * - Separate storage for keys (k) and values (v)
 * - Partitioned across devices for distributed inference [ndev]
 * - Separate cache per transformer layer [nlayer]
 * - Each cache tensor has shape [max_seq_len, nkvh/ndev, dh]
 * 
 * Memory Layout:
 * kv_caches[request][device][layer] -> Tensor[max_seq_len, nkvh/ndev, dh]
 * 
 * Usage Pattern:
 * 1. During prefill: store keys/values for all input tokens
 * 2. During generation: append new keys/values for each generated token
 * 3. Attention computation uses cached keys/values plus current tokens
 * 
 * This design enables O(1) token generation after initial prefill.
 */
struct KVCache {
    // ================================
    // Cache Storage Tensors
    // ================================
    std::vector<std::vector<std::shared_ptr<Tensor>>> k;  // Key cache [ndev][nlayer]
    std::vector<std::vector<std::shared_ptr<Tensor>>> v;  // Value cache [ndev][nlayer]
    
    /*
     * Cache Tensor Dimensions per Device per Layer:
     * Shape: [max_seq_len, nkvh/ndev, dh]
     * - max_seq_len: Maximum context length (dctx)
     * - nkvh/ndev: Key-value heads allocated to this device
     * - dh: Head dimension
     * 
     * Storage Layout in Memory:
     * - Contiguous storage allows efficient slicing for different sequence lengths
     * - Device partitioning enables parallel KV cache updates
     * - Layer separation allows independent cache management per transformer layer
     */
};

#endif

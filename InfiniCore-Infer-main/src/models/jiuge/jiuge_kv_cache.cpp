/*
 * KV Cache Management Implementation
 * 
 * This file implements efficient key-value cache operations for autoregressive
 * transformer inference. The KV cache stores past attention keys and values
 * to avoid recomputation during sequential token generation.
 * 
 * Key Features:
 * - Multi-device distributed cache storage
 * - Efficient memory allocation and copying
 * - Cache sharing and duplication for beam search
 * - Proper resource cleanup and memory management
 * 
 * Cache Organization:
 * - Partitioned across devices for tensor parallelism
 * - Separate storage per transformer layer
 * - Contiguous memory layout for efficient access patterns
 */

#include "jiuge_impl.hpp"

/*
 * Create KV Cache for New Inference Request
 * 
 * Allocates fresh key-value cache storage for a new inference request.
 * Creates separate cache tensors for each device and transformer layer
 * to support distributed inference with tensor parallelism.
 * 
 * Cache Layout:
 * - cache->k[idev][layer]: Key cache for device idev, layer layer
 * - cache->v[idev][layer]: Value cache for device idev, layer layer  
 * - Each tensor shape: [max_len, nkvh/ndev, dh]
 *   - max_len: Maximum sequence length (model context limit)
 *   - nkvh/ndev: Key-value heads per device (distributed across devices)
 *   - dh: Head dimension
 * 
 * Memory Allocation:
 * - Uses device-specific memory allocation for optimal performance
 * - Allocates maximum capacity upfront to avoid reallocation during generation
 * - Zero-initialized memory ensures clean cache state
 * 
 * Parameters:
 * - model: JiugeModel instance containing device and architecture info
 * 
 * Returns: Pointer to newly allocated KVCache structure
 */
__C struct KVCache *createKVCache(const JiugeModel *model) {
    // Create new KVCache instance
    KVCache *cache = new KVCache();
    
    // Extract model dimensions for cache sizing
    auto ndev = model->dev_resources.size();          // Number of devices
    auto nkvh = model->meta.nkvh / ndev;             // KV heads per device  
    auto max_len = model->meta.dctx;                 // Maximum context length
    auto dh = model->meta.dh;                        // Head dimension
    
    // Define cache tensor shape: [max_len, nkvh_per_device, dh]
    auto shape = std::vector<size_t>{max_len, nkvh, dh};
    
    /*
     * Allocate Cache Tensors for Each Device
     * 
     * For distributed inference, each device stores a partition of the KV cache
     * corresponding to its assigned attention heads. This enables parallel
     * cache updates and attention computation across devices.
     */
    for (unsigned int idev = 0; idev < ndev; idev++) {
        // Set device context for memory allocation
        RUN_INFINI(infinirtSetDevice(model->device, model->dev_ids[idev]));
        
        // Create per-layer cache storage for this device
        auto kcache = std::vector<std::shared_ptr<Tensor>>();  // Key cache per layer
        auto vcache = std::vector<std::shared_ptr<Tensor>>();  // Value cache per layer
        
        /*
         * Allocate Cache Tensors for Each Transformer Layer
         * 
         * Each layer needs separate key and value cache storage.
         * Cache tensors are allocated with maximum capacity to avoid
         * reallocation during autoregressive generation.
         */
        for (unsigned int layer = 0; layer < model->meta.nlayer; layer++) {
            // Allocate key cache: [max_len, nkvh/ndev, dh]
            kcache.push_back(std::move(Tensor::buffer(model->meta.dt_logits, shape)));
            
            // Allocate value cache: [max_len, nkvh/ndev, dh]  
            vcache.push_back(std::move(Tensor::buffer(model->meta.dt_logits, shape)));
        }
        
        // Store device-specific cache arrays
        cache->k.push_back(kcache);
        cache->v.push_back(vcache);
    }

    return cache;
}

/*
 * Duplicate KV Cache for Beam Search or Branching
 * 
 * Creates a copy of an existing KV cache with the first seq_len tokens.
 * This is useful for beam search where multiple generation paths share
 * a common prefix but diverge at some point.
 * 
 * Use Cases:
 * - Beam search: Copy cache at branching points
 * - Speculative decoding: Create candidate generation branches
 * - Multi-turn conversations: Copy base context for new turns
 * 
 * Copying Process:
 * 1. Create new cache with same structure as original
 * 2. Copy first seq_len tokens from each cache tensor
 * 3. Leave remaining capacity for future token generation
 * 
 * Memory Efficiency:
 * - Only copies the active portion of the cache (seq_len tokens)
 * - Preserves cache structure and device distribution
 * - Uses efficient device-to-device memory copy operations
 * 
 * Parameters:
 * - model: JiugeModel instance for cache structure
 * - kv_cache: Source cache to duplicate
 * - seq_len: Number of tokens to copy from source cache
 * 
 * Returns: Pointer to newly created cache with copied data
 */
__C struct KVCache *duplicateKVCache(const JiugeModel *model,
                                     const KVCache *kv_cache,
                                     unsigned int seq_len) {
    // Create new cache with same structure as original
    auto new_kv_cache = createKVCache(model);
    
    // Extract model dimensions for copy size calculation
    auto ndev = model->dev_resources.size();          // Number of devices
    auto nkvh = model->meta.nkvh / ndev;             // KV heads per device
    auto dh = model->meta.dh;                        // Head dimension  
    auto dt_size = dsize(model->meta.dt_logits);     // Data type size in bytes
    
    /*
     * Copy Cache Data Across All Devices and Layers
     * 
     * Iterate through each device and layer to copy the active portion
     * of the cache (first seq_len tokens). This preserves the distributed
     * structure while only copying the relevant data.
     */
    for (unsigned int idev = 0; idev < ndev; idev++) {
        // Set device context for memory operations
        RUN_INFINI(infinirtSetDevice(model->device, model->dev_ids[idev]));
        
        /*
         * Copy Per-Layer Cache Data
         * 
         * For each transformer layer, copy both key and value caches.
         * Memory copy size: seq_len * nkvh * dh * data_type_size
         * 
         * Cache memory layout: [seq_len, nkvh, dh] contiguous storage
         * Copy operation: efficient device-to-device memory transfer
         */
        for (unsigned int layer = 0; layer < model->meta.nlayer; layer++) {
            /*
             * Copy Key Cache: [seq_len, nkvh, dh] elements
             * 
             * Source: kv_cache->k[idev][layer] (original cache)
             * Destination: new_kv_cache->k[idev][layer] (new cache)
             * Size: seq_len * nkvh * dh * sizeof(data_type)
             */
            RUN_INFINI(infinirtMemcpy(new_kv_cache->k[idev][layer]->data(),   // Destination
                                      kv_cache->k[idev][layer]->data(),      // Source
                                      seq_len * nkvh * dh * dt_size,         // Copy size in bytes
                                      INFINIRT_MEMCPY_D2D));                 // Device-to-device copy
            
            /*
             * Copy Value Cache: [seq_len, nkvh, dh] elements
             * 
             * Same structure and size as key cache copy.
             */                          
            RUN_INFINI(infinirtMemcpy(new_kv_cache->v[idev][layer]->data(),   // Destination
                                      kv_cache->v[idev][layer]->data(),      // Source  
                                      seq_len * nkvh * dh * dt_size,         // Copy size in bytes
                                      INFINIRT_MEMCPY_D2D));                 // Device-to-device copy
        }
    }
    
    return new_kv_cache;
}

/*
 * Destroy KV Cache and Release Memory
 * 
 * Properly deallocates a KV cache and releases all associated device memory.
 * This function ensures clean memory management and prevents memory leaks
 * in long-running inference applications.
 * 
 * Cleanup Process:
 * 1. Set device context for each device
 * 2. Release tensor memory using shared_ptr reset (automatic deallocation)
 * 3. Deallocate the KVCache structure itself
 * 
 * Memory Safety:
 * - Uses RAII principles via shared_ptr for automatic memory management
 * - Ensures device context is properly set before memory operations
 * - Prevents double-free errors through proper reference counting
 * 
 * Parameters:
 * - model: JiugeModel instance for device context
 * - kv_cache: Cache to destroy and deallocate
 */
__C void dropKVCache(JiugeModel const *model, KVCache *kv_cache) {
    auto ndev = model->dev_resources.size();
    
    /*
     * Release Cache Tensors for Each Device and Layer
     * 
     * Iterate through all devices and layers to properly release
     * the tensor memory using shared_ptr reference counting.
     */
    for (unsigned int idev = 0; idev < ndev; idev++) {
        // Set device context for memory operations
        RUN_INFINI(infinirtSetDevice(model->device, model->dev_ids[idev]));
        
        /*
         * Release Per-Layer Cache Tensors
         * 
         * Call reset() on shared_ptr to decrement reference count.
         * When reference count reaches zero, tensor memory is automatically freed.
         */
        for (unsigned int layer = 0; layer < model->meta.nlayer; layer++) {
            kv_cache->k[idev][layer].reset();  // Release key cache tensor
            kv_cache->v[idev][layer].reset();  // Release value cache tensor
        }
    }
    
    // Deallocate the KVCache structure itself
    delete kv_cache;
}

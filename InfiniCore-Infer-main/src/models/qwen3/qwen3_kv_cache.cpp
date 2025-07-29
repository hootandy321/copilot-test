#include "qwen3_impl.hpp"
#include "qwen3_weight.hpp"

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "infinicore_infer.h"

#include <vector>

// KV Cache implementation for Qwen3 (based on jiuge implementation)
__C struct KVCache *
createQwen3KVCache(const struct Qwen3Model *model) {
    auto meta = &model->meta;
    auto nlayer = meta->nlayer;
    auto nkvh = meta->nkvh;
    auto dh = meta->dh;
    auto dctx = meta->dctx;
    auto ndev = model->dev_ids.size();
    auto dt_logits = meta->dt_logits;

    auto cache = new KVCache;
    cache->k.resize(ndev);
    cache->v.resize(ndev);
    
    for (size_t idev = 0; idev < ndev; idev++) {
        cache->k[idev].resize(nlayer);
        cache->v[idev].resize(nlayer);
        
        RUN_INFINI(infinirtSetDevice(model->device, model->dev_ids[idev]));
        
        for (size_t layer = 0; layer < nlayer; layer++) {
            // K cache: [dctx, nkvh / ndev, dh]
            auto k_shape = std::vector<size_t>({dctx, nkvh / ndev, dh});
            cache->k[idev][layer] = Tensor::buffer(dt_logits, k_shape, model->dev_resources[idev].memory_pool);
            
            // V cache: [dctx, nkvh / ndev, dh]  
            auto v_shape = std::vector<size_t>({dctx, nkvh / ndev, dh});
            cache->v[idev][layer] = Tensor::buffer(dt_logits, v_shape, model->dev_resources[idev].memory_pool);
        }
    }
    
    return cache;
}

__C struct KVCache *
duplicateQwen3KVCache(const struct Qwen3Model *model,
                      const struct KVCache *src, uint32_t seq_len) {
    auto meta = &model->meta;
    auto nlayer = meta->nlayer;
    auto nkvh = meta->nkvh;
    auto dh = meta->dh;
    auto ndev = model->dev_ids.size();
    auto dt_logits = meta->dt_logits;

    auto cache = new KVCache;
    cache->k.resize(ndev);
    cache->v.resize(ndev);
    
    for (size_t idev = 0; idev < ndev; idev++) {
        cache->k[idev].resize(nlayer);
        cache->v[idev].resize(nlayer);
        
        RUN_INFINI(infinirtSetDevice(model->device, model->dev_ids[idev]));
        
        for (size_t layer = 0; layer < nlayer; layer++) {
            // K cache: [dctx, nkvh / ndev, dh]
            auto k_shape = std::vector<size_t>({meta->dctx, nkvh / ndev, dh});
            cache->k[idev][layer] = Tensor::buffer(dt_logits, k_shape, model->dev_resources[idev].memory_pool);
            
            // V cache: [dctx, nkvh / ndev, dh]
            auto v_shape = std::vector<size_t>({meta->dctx, nkvh / ndev, dh});
            cache->v[idev][layer] = Tensor::buffer(dt_logits, v_shape, model->dev_resources[idev].memory_pool);
            
            // Copy from source cache if seq_len > 0
            if (seq_len > 0 && src != nullptr) {
                size_t copy_size = seq_len * (nkvh / ndev) * dh * dsize(dt_logits);
                RUN_INFINI(infinirtMemcpyAsync(
                    cache->k[idev][layer]->data(),
                    src->k[idev][layer]->data(),
                    copy_size,
                    INFINIRT_MEMCPY_D2D,
                    model->dev_resources[idev].stream));
                RUN_INFINI(infinirtMemcpyAsync(
                    cache->v[idev][layer]->data(),
                    src->v[idev][layer]->data(),
                    copy_size,
                    INFINIRT_MEMCPY_D2D,
                    model->dev_resources[idev].stream));
            }
        }
        RUN_INFINI(infinirtStreamSynchronize(model->dev_resources[idev].stream));
    }
    
    return cache;
}

__C void
dropQwen3KVCache(const struct Qwen3Model *model,
                 struct KVCache *cache) {
    if (cache == nullptr) return;
    
    auto ndev = model->dev_ids.size();
    auto nlayer = model->meta.nlayer;
    
    for (size_t idev = 0; idev < ndev; idev++) {
        for (size_t layer = 0; layer < nlayer; layer++) {
            cache->k[idev][layer].reset();
            cache->v[idev][layer].reset();
        }
        cache->k[idev].clear();
        cache->v[idev].clear();
    }
    cache->k.clear();
    cache->v.clear();
    
    delete cache;
}
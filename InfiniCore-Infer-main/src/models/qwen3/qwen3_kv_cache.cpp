#include "qwen3_impl.hpp"

__C struct KVCache *createQwen3KVCache(const Qwen3Model *model) {
    KVCache *cache = new KVCache();
    auto ndev = model->dev_resources.size();
    auto nkvh = model->meta.nkvh / ndev;
    auto max_len = model->meta.dctx;
    auto dh = model->meta.dh;
    auto shape = std::vector<size_t>{max_len, nkvh, dh};
    for (unsigned int idev = 0; idev < ndev; idev++) {
        RUN_INFINI(infinirtSetDevice(model->device, model->dev_ids[idev]));
        auto kcache = std::vector<std::shared_ptr<Tensor>>();
        auto vcache = std::vector<std::shared_ptr<Tensor>>();
        for (unsigned int layer = 0; layer < model->meta.nlayer; layer++) {
            kcache.push_back(std::move(Tensor::buffer(model->meta.dt_logits, shape)));
            vcache.push_back(std::move(Tensor::buffer(model->meta.dt_logits, shape)));
        }
        cache->k.push_back(kcache);
        cache->v.push_back(vcache);
    }

    return cache;
}

__C struct KVCache *duplicateQwen3KVCache(const Qwen3Model *model,
                                          const KVCache *kv_cache,
                                          unsigned int seq_len) {
    auto new_kv_cache = createQwen3KVCache(model);
    auto ndev = model->dev_resources.size();
    auto nkvh = model->meta.nkvh / ndev;
    auto dh = model->meta.dh;
    auto dt_size = dsize(model->meta.dt_logits);
    for (unsigned int idev = 0; idev < ndev; idev++) {
        RUN_INFINI(infinirtSetDevice(model->device, model->dev_ids[idev]));
        for (unsigned int layer = 0; layer < model->meta.nlayer; layer++) {
            RUN_INFINI(infinirtMemcpy(new_kv_cache->k[idev][layer]->data(),
                                      kv_cache->k[idev][layer]->data(),
                                      seq_len * nkvh * dh * dt_size,
                                      INFINIRT_MEMCPY_D2D));
            RUN_INFINI(infinirtMemcpy(new_kv_cache->v[idev][layer]->data(),
                                      kv_cache->v[idev][layer]->data(),
                                      seq_len * nkvh * dh * dt_size,
                                      INFINIRT_MEMCPY_D2D));
        }
    }
    return new_kv_cache;
}

__C void dropQwen3KVCache(Qwen3Model const *model, KVCache *kv_cache) {
    auto ndev = model->dev_resources.size();
    for (unsigned int idev = 0; idev < ndev; idev++) {
        RUN_INFINI(infinirtSetDevice(model->device, model->dev_ids[idev]));
        for (unsigned int layer = 0; layer < model->meta.nlayer; layer++) {
            kv_cache->k[idev][layer].reset();
            kv_cache->v[idev][layer].reset();
        }
        kv_cache->k[idev].clear();
        kv_cache->v[idev].clear();
    }
    kv_cache->k.clear();
    kv_cache->v.clear();
    delete kv_cache;
}
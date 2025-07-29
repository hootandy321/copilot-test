{
    files = {
        "build/.objs/infinicore_infer/linux/x86_64/release/src/models/jiuge/jiuge.cpp.o",
        "build/.objs/infinicore_infer/linux/x86_64/release/src/models/jiuge/jiuge_kv_cache.cpp.o",
        "build/.objs/infinicore_infer/linux/x86_64/release/src/models/qwen3/qwen3_kv_cache.cpp.o",
        "build/.objs/infinicore_infer/linux/x86_64/release/src/models/qwen3/qwen3.cpp.o",
        "build/.objs/infinicore_infer/linux/x86_64/release/src/tensor/tensor.cpp.o",
        "build/.objs/infinicore_infer/linux/x86_64/release/src/tensor/strorage.cpp.o",
        "build/.objs/infinicore_infer/linux/x86_64/release/src/tensor/transform.cpp.o",
        "build/.objs/infinicore_infer/linux/x86_64/release/src/allocator/memory_allocator.cpp.o"
    },
    values = {
        "/usr/bin/g++",
        {
            "-shared",
            "-m64",
            "-fPIC",
            "-L/home/hootandy/.infini/lib",
            "-linfiniop",
            "-linfinirt",
            "-linfiniccl"
        }
    }
}
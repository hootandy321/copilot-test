#include "qwen3_debug.hpp"
#include "../../tensor.hpp"
#include "../../allocator.hpp"
#include <iostream>
#include <vector>
#include <random>

int main() {
    std::cout << "开始测试调试功能..." << std::endl;
    
    // 创建一个内存池
    auto memory_pool = std::make_shared<MemoryPool>(1024 * 1024); // 1MB内存池
    
    // 创建一个float32类型的张量
    std::vector<size_t> shape = {2, 3, 4}; // 24个元素
    auto tensor_f32 = Tensor::buffer(INFINI_DTYPE_F32, shape, memory_pool);
    
    // 填充一些测试数据
    float* data_f32 = reinterpret_cast<float*>(tensor_f32->data());
    for (int i = 0; i < 24; i++) {
        data_f32[i] = static_cast<float>(i) * 0.5f;
    }
    
    // 收集float32张量数据
    collectDebugData(tensor_f32, "test_tensor_f32");
    
    // 创建一个较小的float16类型的张量
    std::vector<size_t> shape2 = {2, 5};
    auto tensor_f16 = Tensor::buffer(INFINI_DTYPE_F16, shape2, memory_pool);
    
    // 填充一些测试数据 (float16格式)
    uint16_t* data_f16 = reinterpret_cast<uint16_t*>(tensor_f16->data());
    for (int i = 0; i < 10; i++) {
        // 简单模拟float16数据 (实际转换比较复杂，这里只是测试)
        data_f16[i] = static_cast<uint16_t>(i + 1000);
    }
    
    // 收集float16张量数据
    collectDebugData(tensor_f16, "test_tensor_f16");
    
    // 保存到JSON文件
    saveToJson("debug_test_output.json");
    
    std::cout << "测试完成，数据已保存到 debug_test_output.json" << std::endl;
    
    return 0;
}
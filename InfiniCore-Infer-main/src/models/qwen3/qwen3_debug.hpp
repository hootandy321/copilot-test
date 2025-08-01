#include "../../tensor.hpp"
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <fstream>
#include "/home/hootandy/InfiniCore-Infer-main/third_party/nlohmann/json.hpp"
// 你需要先有 half2float 的定义
static inline float half2float(uint16_t h) {
    uint32_t sign     = (h & 0x8000) << 16;
    uint32_t exp_mant = (h & 0x7FFF) << 13;
    uint32_t result   = sign | exp_mant;
    return *reinterpret_cast<float*>(&result);
}

// 全局调试数据存储
std::map<std::string, std::vector<float>> debug_data;

// 收集张量数据
void collectDebugData(std::shared_ptr<Tensor> tensor, const std::string &name) {
    if (!tensor) {
        std::cerr << "[ERROR] Tensor " << name << " is null!" << std::endl;
        return;
    }
    const std::vector<size_t>& shape = tensor->shape();
    size_t total = 1;
    for (auto dim : shape) total *= dim;

    // 只支持 float32 和 float16 (half) 两种类型
    std::vector<float> buffer;
    if (tensor->dtype() == INFINI_DTYPE_F32) {
        const float* ptr = reinterpret_cast<const float*>(tensor->data());
        buffer.assign(ptr, ptr + total);
    } else if (tensor->dtype() == INFINI_DTYPE_F16) {
        const uint16_t* ptr = reinterpret_cast<const uint16_t*>(tensor->data());
        buffer.reserve(total);
        for (size_t i = 0; i < total; ++i) {
            buffer.push_back(half2float(ptr[i]));
        }
    } else {
        std::cerr << "[WARN] collectDebugData: Unsupported dtype for " << name << std::endl;
        return;
    }
    debug_data[name] = std::move(buffer);

    // std::cout << "[DEBUG] Collected data for " << name << " (elements=" << total << ")" << std::endl;
}

// 保存所有收集到的数据到 JSON 文件
void saveToJson(const std::string &filename) {
    nlohmann::json json_output;
    for (const auto &entry : debug_data) {
        json_output[entry.first] = entry.second;
    }
    std::ofstream file(filename);
    if (file.is_open()) {
        file << json_output.dump(4); // pretty print
        file.close();
        std::cout << "[DEBUG] Data saved to " << filename << std::endl;
    } else {
        std::cerr << "[ERROR] Failed to open " << filename << " for writing." << std::endl;
    }
}
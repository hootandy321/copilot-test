#include <iostream>
#include <vector>
#include <memory>
#include <cstring>
#include <iomanip>

// Include the qw headers
#include "infinicore_infer.h"
#include "src/models/qw/qwen3_impl.hpp"
#include "src/tensor.hpp"

using namespace std;

// Simple test class for small matrix verification
class QwMatrixTest {
private:
    // Minimal model configuration for testing
    Qwen3Meta meta;
    Qwen3Weights weights;
    
    // Small test dimensions - visible in terminal
    static constexpr size_t NLAYER = 2;    // 2 layers
    static constexpr size_t D = 4;         // Hidden dimension 4
    static constexpr size_t NH = 2;        // 2 attention heads  
    static constexpr size_t NKVH = 2;      // 2 key-value heads
    static constexpr size_t DH = 2;        // Head dimension 2 (D/NH)
    static constexpr size_t DI = 8;        // Intermediate dimension 8
    static constexpr size_t DVOC = 10;     // Vocabulary size 10
    static constexpr size_t DCTX = 16;     // Context length 16
    
    // Test weights storage
    vector<float> input_embd_data;
    vector<float> output_embd_data;
    vector<float> output_norm_data;
    vector<vector<float>> attn_norm_data;
    vector<vector<float>> attn_q_norm_data;
    vector<vector<float>> attn_k_norm_data;
    vector<vector<float>> attn_q_proj_data;
    vector<vector<float>> attn_k_proj_data;
    vector<vector<float>> attn_v_proj_data;
    vector<vector<float>> attn_o_proj_data;
    vector<vector<float>> mlp_norm_data;
    vector<vector<float>> mlp_gate_proj_data;
    vector<vector<float>> mlp_up_proj_data;
    vector<vector<float>> mlp_down_proj_data;
    
    // Weight pointers for C API
    vector<const void*> attn_norm_ptrs;
    vector<const void*> attn_q_norm_ptrs;
    vector<const void*> attn_k_norm_ptrs;
    vector<const void*> attn_q_proj_ptrs;
    vector<const void*> attn_k_proj_ptrs;
    vector<const void*> attn_v_proj_ptrs;
    vector<const void*> attn_o_proj_ptrs;
    vector<const void*> mlp_norm_ptrs;
    vector<const void*> mlp_gate_proj_ptrs;
    vector<const void*> mlp_up_proj_ptrs;
    vector<const void*> mlp_down_proj_ptrs;

public:
    QwMatrixTest() {
        initializeMetadata();
        initializeWeights();
        setupWeightPointers();
    }
    
    void initializeMetadata() {
        meta.dt_logits = INFINI_DTYPE_F32;
        meta.nlayer = NLAYER;
        meta.d = D;
        meta.nh = NH;
        meta.nkvh = NKVH;
        meta.dh = DH;
        meta.di = DI;
        meta.dctx = DCTX;
        meta.dvoc = DVOC;
        meta.epsilon = 1e-6f;
        meta.theta = 10000.0f;
        meta.bos_token = 1;
        meta.end_token = 2;
        meta.attn_dropout = 0.0f;
        meta.tie_embd = false;
        
        cout << "\n=== QW Matrix Test Configuration ===" << endl;
        cout << "Layers: " << NLAYER << ", Hidden: " << D << ", Heads: " << NH << "/" << NKVH << endl;
        cout << "Head dim: " << DH << ", Intermediate: " << DI << ", Vocab: " << DVOC << endl;
    }
    
    void initializeWeights() {
        // Initialize input embedding [DVOC, D] with small identifiable values
        input_embd_data.resize(DVOC * D);
        for (size_t i = 0; i < DVOC; ++i) {
            for (size_t j = 0; j < D; ++j) {
                input_embd_data[i * D + j] = 0.1f + 0.01f * i + 0.001f * j;
            }
        }
        
        // Initialize output embedding [D, DVOC] 
        output_embd_data.resize(D * DVOC);
        for (size_t i = 0; i < D; ++i) {
            for (size_t j = 0; j < DVOC; ++j) {
                output_embd_data[i * DVOC + j] = 0.2f + 0.01f * i + 0.001f * j;
            }
        }
        
        // Initialize output norm [D]
        output_norm_data.resize(D);
        for (size_t i = 0; i < D; ++i) {
            output_norm_data[i] = 1.0f + 0.01f * i;
        }
        
        // Initialize layer weights
        attn_norm_data.resize(NLAYER);
        attn_q_norm_data.resize(NLAYER);
        attn_k_norm_data.resize(NLAYER);
        attn_q_proj_data.resize(NLAYER);
        attn_k_proj_data.resize(NLAYER);
        attn_v_proj_data.resize(NLAYER);
        attn_o_proj_data.resize(NLAYER);
        mlp_norm_data.resize(NLAYER);
        mlp_gate_proj_data.resize(NLAYER);
        mlp_up_proj_data.resize(NLAYER);
        mlp_down_proj_data.resize(NLAYER);
        
        for (size_t layer = 0; layer < NLAYER; ++layer) {
            // Attention norm [D]
            attn_norm_data[layer].resize(D);
            for (size_t i = 0; i < D; ++i) {
                attn_norm_data[layer][i] = 1.0f + 0.1f * layer + 0.01f * i;
            }
            
            // Q/K norm [DH]
            attn_q_norm_data[layer].resize(DH);
            attn_k_norm_data[layer].resize(DH);
            for (size_t i = 0; i < DH; ++i) {
                attn_q_norm_data[layer][i] = 1.0f + 0.1f * layer + 0.02f * i;
                attn_k_norm_data[layer][i] = 1.0f + 0.1f * layer + 0.03f * i;
            }
            
            // Q projection [D, NH*DH]
            attn_q_proj_data[layer].resize(D * NH * DH);
            for (size_t i = 0; i < D * NH * DH; ++i) {
                attn_q_proj_data[layer][i] = 0.1f + 0.05f * layer + 0.001f * i;
            }
            
            // K projection [D, NKVH*DH]  
            attn_k_proj_data[layer].resize(D * NKVH * DH);
            for (size_t i = 0; i < D * NKVH * DH; ++i) {
                attn_k_proj_data[layer][i] = 0.2f + 0.05f * layer + 0.001f * i;
            }
            
            // V projection [D, NKVH*DH]
            attn_v_proj_data[layer].resize(D * NKVH * DH);
            for (size_t i = 0; i < D * NKVH * DH; ++i) {
                attn_v_proj_data[layer][i] = 0.3f + 0.05f * layer + 0.001f * i;
            }
            
            // O projection [NH*DH, D]
            attn_o_proj_data[layer].resize(NH * DH * D);
            for (size_t i = 0; i < NH * DH * D; ++i) {
                attn_o_proj_data[layer][i] = 0.4f + 0.05f * layer + 0.001f * i;
            }
            
            // MLP norm [D]
            mlp_norm_data[layer].resize(D);
            for (size_t i = 0; i < D; ++i) {
                mlp_norm_data[layer][i] = 1.0f + 0.2f * layer + 0.01f * i;
            }
            
            // MLP projections
            mlp_gate_proj_data[layer].resize(D * DI);
            mlp_up_proj_data[layer].resize(D * DI);
            mlp_down_proj_data[layer].resize(DI * D);
            
            for (size_t i = 0; i < D * DI; ++i) {
                mlp_gate_proj_data[layer][i] = 0.5f + 0.05f * layer + 0.001f * i;
                mlp_up_proj_data[layer][i] = 0.6f + 0.05f * layer + 0.001f * i;
            }
            
            for (size_t i = 0; i < DI * D; ++i) {
                mlp_down_proj_data[layer][i] = 0.7f + 0.05f * layer + 0.001f * i;
            }
        }
        
        cout << "\n=== Weight Initialization Complete ===" << endl;
        cout << "Input embedding: " << input_embd_data.size() << " elements" << endl;
        cout << "Layer weights initialized for " << NLAYER << " layers" << endl;
    }
    
    void setupWeightPointers() {
        // Resize pointer vectors
        attn_norm_ptrs.resize(NLAYER);
        attn_q_norm_ptrs.resize(NLAYER);
        attn_k_norm_ptrs.resize(NLAYER);
        attn_q_proj_ptrs.resize(NLAYER);
        attn_k_proj_ptrs.resize(NLAYER);
        attn_v_proj_ptrs.resize(NLAYER);
        attn_o_proj_ptrs.resize(NLAYER);
        mlp_norm_ptrs.resize(NLAYER);
        mlp_gate_proj_ptrs.resize(NLAYER);
        mlp_up_proj_ptrs.resize(NLAYER);
        mlp_down_proj_ptrs.resize(NLAYER);
        
        // Setup pointers
        for (size_t i = 0; i < NLAYER; ++i) {
            attn_norm_ptrs[i] = attn_norm_data[i].data();
            attn_q_norm_ptrs[i] = attn_q_norm_data[i].data();
            attn_k_norm_ptrs[i] = attn_k_norm_data[i].data();
            attn_q_proj_ptrs[i] = attn_q_proj_data[i].data();
            attn_k_proj_ptrs[i] = attn_k_proj_data[i].data();
            attn_v_proj_ptrs[i] = attn_v_proj_data[i].data();
            attn_o_proj_ptrs[i] = attn_o_proj_data[i].data();
            mlp_norm_ptrs[i] = mlp_norm_data[i].data();
            mlp_gate_proj_ptrs[i] = mlp_gate_proj_data[i].data();
            mlp_up_proj_ptrs[i] = mlp_up_proj_data[i].data();
            mlp_down_proj_ptrs[i] = mlp_down_proj_data[i].data();
        }
        
        // Setup weights structure
        weights.nlayer = NLAYER;
        weights.dt_norm = INFINI_DTYPE_F32;
        weights.dt_mat = INFINI_DTYPE_F32;
        weights.transpose_linear_weights = 0;  // [in, out] format
        
        weights.input_embd = input_embd_data.data();
        weights.output_embd = output_embd_data.data();
        weights.output_norm = output_norm_data.data();
        
        weights.attn_norm = attn_norm_ptrs.data();
        weights.attn_q_norm = attn_q_norm_ptrs.data();
        weights.attn_k_norm = attn_k_norm_ptrs.data();
        weights.attn_q_proj = attn_q_proj_ptrs.data();
        weights.attn_k_proj = attn_k_proj_ptrs.data();
        weights.attn_v_proj = attn_v_proj_ptrs.data();
        weights.attn_o_proj = attn_o_proj_ptrs.data();
        weights.mlp_norm = mlp_norm_ptrs.data();
        weights.mlp_gate_proj = mlp_gate_proj_ptrs.data();
        weights.mlp_up_proj = mlp_up_proj_ptrs.data();
        weights.mlp_down_proj = mlp_down_proj_ptrs.data();
        
        cout << "\n=== Weight Pointers Setup Complete ===" << endl;
    }
    
    void displayMatrix(const string& name, const float* data, size_t rows, size_t cols, size_t max_display = 0) {
        cout << "\n--- " << name << " [" << rows << "x" << cols << "] ---" << endl;
        
        size_t display_rows = (max_display > 0) ? min(max_display, rows) : rows;
        size_t display_cols = (max_display > 0) ? min(max_display, cols) : cols;
        
        for (size_t i = 0; i < display_rows; ++i) {
            cout << "  ";
            for (size_t j = 0; j < display_cols; ++j) {
                cout << fixed << setprecision(3) << setw(7) << data[i * cols + j] << " ";
            }
            if (display_cols < cols) cout << " ... (" << (cols - display_cols) << " more)";
            cout << endl;
        }
        if (display_rows < rows) {
            cout << "  ... (" << (rows - display_rows) << " more rows)" << endl;
        }
    }
    
    void displayWeights() {
        cout << "\n=================== WEIGHT MATRICES ===================" << endl;
        
        displayMatrix("Input Embedding", input_embd_data.data(), DVOC, D);
        displayMatrix("Output Norm", output_norm_data.data(), 1, D);
        
        for (size_t layer = 0; layer < NLAYER; ++layer) {
            cout << "\n--- Layer " << layer << " ---" << endl;
            displayMatrix("Attention Norm", attn_norm_data[layer].data(), 1, D);
            displayMatrix("Q Norm", attn_q_norm_data[layer].data(), 1, DH);
            displayMatrix("K Norm", attn_k_norm_data[layer].data(), 1, DH);
            displayMatrix("Q Projection", attn_q_proj_data[layer].data(), D, NH * DH, 4);
            displayMatrix("K Projection", attn_k_proj_data[layer].data(), D, NKVH * DH, 4);
            displayMatrix("V Projection", attn_v_proj_data[layer].data(), D, NKVH * DH, 4);
            displayMatrix("O Projection", attn_o_proj_data[layer].data(), NH * DH, D, 4);
            displayMatrix("MLP Norm", mlp_norm_data[layer].data(), 1, D);
            displayMatrix("MLP Gate", mlp_gate_proj_data[layer].data(), D, DI, 4);
            displayMatrix("MLP Up", mlp_up_proj_data[layer].data(), D, DI, 4);
            displayMatrix("MLP Down", mlp_down_proj_data[layer].data(), DI, D, 4);
        }
    }
    
    void testSimpleInference() {
        cout << "\n=================== INFERENCE TEST ===================" << endl;
        
        try {
            // Enable debug mode to see intermediate calculations
            setQwen3DebugMode(1);
            
            // Create model with CPU device
            int dev_ids[] = {0};
            struct Qwen3Model* model = createQwen3Model(&meta, &weights, INFINI_DEVICE_CPU, 1, dev_ids);
            
            if (!model) {
                cout << "❌ Failed to create model!" << endl;
                return;
            }
            cout << "✅ Model created successfully!" << endl;
            
            // Create KV cache
            struct Qwen3KVCache* kv_cache = createQwen3KVCache(model);
            if (!kv_cache) {
                cout << "❌ Failed to create KV cache!" << endl;
                destroyQwen3Model(model);
                return;
            }
            cout << "✅ KV cache created successfully!" << endl;
            
            // Simple test input: token sequence [1, 3, 5]
            vector<uint32_t> tokens = {1, 3, 5};
            uint32_t ntok = tokens.size();
            
            cout << "\nInput tokens: ";
            for (uint32_t token : tokens) {
                cout << token << " ";
            }
            cout << endl;
            
            // Display corresponding embeddings
            cout << "\nCorresponding embeddings:" << endl;
            for (size_t i = 0; i < tokens.size(); ++i) {
                uint32_t token = tokens[i];
                cout << "Token " << token << " -> [";
                for (size_t j = 0; j < D; ++j) {
                    cout << fixed << setprecision(3) << input_embd_data[token * D + j];
                    if (j < D - 1) cout << ", ";
                }
                cout << "]" << endl;
            }
            
            // Setup inference parameters
            uint32_t nreq = 1;
            uint32_t req_lens[] = {ntok};
            uint32_t req_pos[] = {0};
            struct Qwen3KVCache* kv_caches[] = {kv_cache};
            float temperature[] = {0.0f};  // Deterministic
            uint32_t topk[] = {1};
            float topp[] = {1.0f};
            uint32_t output[1];
            
            cout << "\nRunning inference..." << endl;
            cout << "Request: " << nreq << " sequences, " << ntok << " total tokens" << endl;
            cout << "Sampling: temperature=" << temperature[0] << ", topk=" << topk[0] << ", topp=" << topp[0] << endl;
            
            // Run inference
            inferQwen3Batch(model, tokens.data(), ntok, req_lens, nreq, req_pos, 
                           kv_caches, temperature, topk, topp, output);
                           
            cout << "\n✅ Inference completed!" << endl;
            cout << "Output token: " << output[0] << endl;
            
            // Verify output is in valid range
            if (output[0] < DVOC) {
                cout << "✅ Output token is valid (< vocab_size=" << DVOC << ")" << endl;
            } else {
                cout << "❌ Output token " << output[0] << " exceeds vocab_size " << DVOC << endl;
            }
            
            // Cleanup
            dropQwen3KVCache(model, kv_cache);
            destroyQwen3Model(model);
            
            cout << "✅ Test completed successfully!" << endl;
            
        } catch (const exception& e) {
            cout << "❌ Exception during test: " << e.what() << endl;
        }
    }
};

int main() {
    cout << "\n╔══════════════════════════════════════╗" << endl;
    cout << "║     QW Matrix Test - Small Scale     ║" << endl;
    cout << "╚══════════════════════════════════════╝" << endl;
    
    try {
        QwMatrixTest test;
        
        // Display all weight matrices for verification
        test.displayWeights();
        
        // Run simple inference test
        test.testSimpleInference();
        
        cout << "\n╔══════════════════════════════════════╗" << endl;
        cout << "║        Test Completed                ║" << endl;
        cout << "╚══════════════════════════════════════╝" << endl;
        
    } catch (const exception& e) {
        cout << "❌ Fatal error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
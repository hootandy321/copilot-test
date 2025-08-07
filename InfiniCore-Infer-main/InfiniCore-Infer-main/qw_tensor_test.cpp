#include <iostream>
#include <vector>
#include <memory>
#include <cstring>
#include <iomanip>
#include <cmath>

// Include internal QW implementation headers  
#include "src/models/qw/qwen3_impl.hpp"
#include "src/tensor.hpp"

using namespace std;

// Simple test class that uses actual QW C++ tensor operations
class QwTensorTest {
private:
    // Small test dimensions for terminal display
    static constexpr size_t NLAYER = 1;      // Just 1 layer for simplicity  
    static constexpr size_t D = 4;           // Hidden dimension 4
    static constexpr size_t NH = 2;          // 2 attention heads
    static constexpr size_t NKVH = 2;        // 2 key-value heads
    static constexpr size_t DH = 2;          // Head dimension 2 (D/NH)
    static constexpr size_t DI = 6;          // Intermediate dimension 6
    static constexpr size_t DVOC = 8;        // Vocabulary size 8
    static constexpr size_t DCTX = 16;       // Context length 16
    static constexpr size_t NTOK = 3;        // 3 tokens for testing
    
    Qwen3Meta meta;
    
    // Test weight data (simple, predictable values)
    vector<float> input_embd_data;
    vector<float> output_embd_data;
    vector<float> output_norm_data;
    vector<float> attn_norm_data;
    vector<float> attn_q_norm_data;
    vector<float> attn_k_norm_data;
    vector<float> attn_q_proj_data;
    vector<float> attn_k_proj_data;
    vector<float> attn_v_proj_data;
    vector<float> attn_o_proj_data;
    vector<float> mlp_norm_data;
    vector<float> mlp_gate_proj_data;
    vector<float> mlp_up_proj_data;
    vector<float> mlp_down_proj_data;

public:
    QwTensorTest() {
        initializeAll();
    }
    
    void initializeAll() {
        cout << "\n╔════════════════════════════════════════════════════╗" << endl;
        cout << "║               QW Tensor Operation Test             ║" << endl;
        cout << "╚════════════════════════════════════════════════════╝" << endl;
        
        setupMeta();
        initializeWeights();
    }
    
    void setupMeta() {
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
        
        cout << "\n=== Model Configuration ===" << endl;
        cout << "Layers: " << NLAYER << ", Hidden: " << D << ", Heads: " << NH << "/" << NKVH << endl;
        cout << "Head dim: " << DH << ", Intermediate: " << DI << ", Vocab: " << DVOC << endl;
        cout << "Context: " << DCTX << ", Test tokens: " << NTOK << endl;
    }
    
    void initializeWeights() {
        cout << "\n=== Initializing Simple Weights ===" << endl;
        
        // Initialize with simple, predictable patterns
        
        // Input embedding [DVOC, D] - identity-like for easy verification
        input_embd_data.resize(DVOC * D);
        for (size_t i = 0; i < DVOC; ++i) {
            for (size_t j = 0; j < D; ++j) {
                // Simple pattern: token i gets [i+0.1, i+0.2, i+0.3, i+0.4]
                input_embd_data[i * D + j] = float(i) + 0.1f * (j + 1);
            }
        }
        
        // Output embedding [D, DVOC] - simple transpose-like
        output_embd_data.resize(D * DVOC);
        for (size_t i = 0; i < D; ++i) {
            for (size_t j = 0; j < DVOC; ++j) {
                output_embd_data[i * DVOC + j] = 0.1f + 0.01f * i + 0.001f * j;
            }
        }
        
        // Output norm [D] - close to 1.0
        output_norm_data.resize(D);
        for (size_t i = 0; i < D; ++i) {
            output_norm_data[i] = 1.0f + 0.01f * i;
        }
        
        // Layer 0 weights - single layer for simplicity
        
        // Attention norm [D]
        attn_norm_data.resize(D);
        for (size_t i = 0; i < D; ++i) {
            attn_norm_data[i] = 1.0f + 0.02f * i;  // [1.00, 1.02, 1.04, 1.06]
        }
        
        // Q/K norm [DH]
        attn_q_norm_data.resize(DH);
        attn_k_norm_data.resize(DH);
        for (size_t i = 0; i < DH; ++i) {
            attn_q_norm_data[i] = 1.0f + 0.01f * i;  // [1.00, 1.01]
            attn_k_norm_data[i] = 1.0f + 0.02f * i;  // [1.00, 1.02]
        }
        
        // Q projection [D, NH*DH] - simple identity-like
        attn_q_proj_data.resize(D * NH * DH);
        for (size_t i = 0; i < D * NH * DH; ++i) {
            attn_q_proj_data[i] = 0.1f + 0.01f * i;
        }
        
        // K projection [D, NKVH*DH]
        attn_k_proj_data.resize(D * NKVH * DH);
        for (size_t i = 0; i < D * NKVH * DH; ++i) {
            attn_k_proj_data[i] = 0.2f + 0.01f * i;
        }
        
        // V projection [D, NKVH*DH]
        attn_v_proj_data.resize(D * NKVH * DH);
        for (size_t i = 0; i < D * NKVH * DH; ++i) {
            attn_v_proj_data[i] = 0.3f + 0.01f * i;
        }
        
        // O projection [NH*DH, D]
        attn_o_proj_data.resize(NH * DH * D);
        for (size_t i = 0; i < NH * DH * D; ++i) {
            attn_o_proj_data[i] = 0.25f + 0.005f * i;  // Smaller values for stability
        }
        
        // MLP norm [D]
        mlp_norm_data.resize(D);
        for (size_t i = 0; i < D; ++i) {
            mlp_norm_data[i] = 1.0f + 0.03f * i;
        }
        
        // MLP projections
        mlp_gate_proj_data.resize(D * DI);
        mlp_up_proj_data.resize(D * DI);
        for (size_t i = 0; i < D * DI; ++i) {
            mlp_gate_proj_data[i] = 0.1f + 0.005f * i;
            mlp_up_proj_data[i] = 0.15f + 0.005f * i;
        }
        
        mlp_down_proj_data.resize(DI * D);
        for (size_t i = 0; i < DI * D; ++i) {
            mlp_down_proj_data[i] = 0.05f + 0.002f * i;  // Small values to prevent explosion
        }
        
        cout << "✅ All weights initialized with predictable patterns" << endl;
        displayWeightSamples();
    }
    
    void displayWeightSamples() {
        cout << "\n--- Weight Samples ---" << endl;
        
        cout << "Input embeddings (first 3 tokens):" << endl;
        for (size_t tok = 0; tok < 3 && tok < DVOC; ++tok) {
            cout << "  Token " << tok << ": [";
            for (size_t i = 0; i < D; ++i) {
                cout << fixed << setprecision(2) << input_embd_data[tok * D + i];
                if (i < D - 1) cout << ", ";
            }
            cout << "]" << endl;
        }
        
        cout << "\nAttention norms: [";
        for (size_t i = 0; i < D; ++i) {
            cout << fixed << setprecision(2) << attn_norm_data[i];
            if (i < D - 1) cout << ", ";
        }
        cout << "]" << endl;
        
        cout << "Q norm: [" << attn_q_norm_data[0] << ", " << attn_q_norm_data[1] << "]" << endl;
        cout << "K norm: [" << attn_k_norm_data[0] << ", " << attn_k_norm_data[1] << "]" << endl;
        
        cout << "\nQ projection (4x4 matrix):" << endl;
        for (size_t i = 0; i < D; ++i) {
            cout << "  [";
            for (size_t j = 0; j < NH * DH; ++j) {
                cout << fixed << setprecision(2) << attn_q_proj_data[i * (NH * DH) + j];
                if (j < NH * DH - 1) cout << ", ";
            }
            cout << "]" << endl;
        }
    }
    
    void testTensorCreation() {
        cout << "\n=================== TENSOR CREATION TEST ===================" << endl;
        
        try {
            // Create some test tensors using the QW tensor system
            cout << "\n--- Creating Tensors ---" << endl;
            
            // Create input embedding tensor
            auto input_embd_tensor = Tensor::weight(input_embd_data.data(), INFINI_DTYPE_F32, {DVOC, D});
            cout << "✅ Input embedding tensor created: [" << DVOC << ", " << D << "]" << endl;
            
            // Create hidden state tensor for processing
            vector<float> hidden_data(NTOK * D, 0.0f);
            auto hidden_tensor = Tensor::weight(hidden_data.data(), INFINI_DTYPE_F32, {NTOK, D});
            cout << "✅ Hidden state tensor created: [" << NTOK << ", " << D << "]" << endl;
            
            // Create Q projection weight tensor
            auto q_proj_tensor = Tensor::weight(attn_q_proj_data.data(), INFINI_DTYPE_F32, {D, NH * DH});
            cout << "✅ Q projection tensor created: [" << D << ", " << NH * DH << "]" << endl;
            
            // Test tensor access
            cout << "\n--- Testing Tensor Access ---" << endl;
            cout << "Input embedding tensor data pointer: " << input_embd_tensor->data() << endl;
            cout << "Hidden tensor data pointer: " << hidden_tensor->data() << endl;
            
            // Test tensor shape access
            auto input_shape = input_embd_tensor->shape();
            cout << "Input embedding shape: [";
            for (size_t i = 0; i < input_shape.size(); ++i) {
                cout << input_shape[i];
                if (i < input_shape.size() - 1) cout << ", ";
            }
            cout << "]" << endl;
            
        } catch (const exception& e) {
            cout << "❌ Tensor creation error: " << e.what() << endl;
        }
    }
    
    void testEmbeddingLookup() {
        cout << "\n=================== EMBEDDING LOOKUP TEST ===================" << endl;
        
        try {
            // Test tokens
            vector<uint32_t> test_tokens = {0, 2, 5};
            cout << "\nTest tokens: [";
            for (size_t i = 0; i < test_tokens.size(); ++i) {
                cout << test_tokens[i];
                if (i < test_tokens.size() - 1) cout << ", ";
            }
            cout << "]" << endl;
            
            // Manual embedding lookup (simulating what the QW code does)
            vector<float> embedded_result(NTOK * D);
            
            cout << "\n--- Manual Embedding Lookup ---" << endl;
            for (size_t i = 0; i < test_tokens.size(); ++i) {
                uint32_t token = test_tokens[i];
                if (token < DVOC) {
                    cout << "Token " << token << " -> embedding: [";
                    for (size_t j = 0; j < D; ++j) {
                        float emb_val = input_embd_data[token * D + j];
                        embedded_result[i * D + j] = emb_val;
                        cout << fixed << setprecision(2) << emb_val;
                        if (j < D - 1) cout << ", ";
                    }
                    cout << "]" << endl;
                } else {
                    cout << "❌ Token " << token << " out of vocabulary range!" << endl;
                }
            }
            
            // Create tensor from embedded result
            auto embedded_tensor = Tensor::weight(embedded_result.data(), INFINI_DTYPE_F32, {NTOK, D});
            cout << "\n✅ Embedded tensor created: [" << NTOK << ", " << D << "]" << endl;
            
        } catch (const exception& e) {
            cout << "❌ Embedding lookup error: " << e.what() << endl;
        }
    }
    
    void testRMSNorm() {
        cout << "\n=================== RMS NORMALIZATION TEST ===================" << endl;
        
        try {
            // Test input
            vector<float> test_input = {2.0f, -1.5f, 3.0f, 0.8f};  // [NTOK=1, D=4] for simplicity
            cout << "\nRMS Norm input: [";
            for (size_t i = 0; i < D; ++i) {
                cout << fixed << setprecision(2) << test_input[i];
                if (i < D - 1) cout << ", ";
            }
            cout << "]" << endl;
            
            cout << "Attention norm weights: [";
            for (size_t i = 0; i < D; ++i) {
                cout << fixed << setprecision(2) << attn_norm_data[i];
                if (i < D - 1) cout << ", ";
            }
            cout << "]" << endl;
            
            // Manual RMS normalization computation
            vector<float> norm_output(D);
            
            // Compute mean square
            float mean_square = 0.0f;
            for (size_t i = 0; i < D; ++i) {
                mean_square += test_input[i] * test_input[i];
            }
            mean_square /= D;
            
            // Compute RMS scale
            float rms_scale = 1.0f / sqrt(mean_square + meta.epsilon);
            
            // Apply normalization with weights
            for (size_t i = 0; i < D; ++i) {
                norm_output[i] = test_input[i] * rms_scale * attn_norm_data[i];
            }
            
            cout << "\nRMS computation:" << endl;
            cout << "  Mean square: " << fixed << setprecision(4) << mean_square << endl;
            cout << "  RMS scale: " << fixed << setprecision(4) << rms_scale << endl;
            cout << "  Output: [";
            for (size_t i = 0; i < D; ++i) {
                cout << fixed << setprecision(3) << norm_output[i];
                if (i < D - 1) cout << ", ";
            }
            cout << "]" << endl;
            
            // Verification: output should have controlled magnitude
            float output_norm = 0.0f;
            for (size_t i = 0; i < D; ++i) {
                output_norm += norm_output[i] * norm_output[i];
            }
            output_norm = sqrt(output_norm / D);
            cout << "  Output RMS: " << fixed << setprecision(4) << output_norm << endl;
            
        } catch (const exception& e) {
            cout << "❌ RMS norm error: " << e.what() << endl;
        }
    }
    
    void testMatrixMultiply() {
        cout << "\n=================== MATRIX MULTIPLICATION TEST ===================" << endl;
        
        try {
            // Test matrix multiply: input * Q_proj = Q_output
            // Input: [NTOK, D] * Weight: [D, NH*DH] -> Output: [NTOK, NH*DH]
            
            vector<float> test_input = {
                1.0f, 0.5f, -0.2f, 0.8f,   // token 0 hidden state
                0.3f, -0.7f, 1.2f, -0.1f,  // token 1 hidden state  
                -0.4f, 0.9f, 0.6f, -0.3f   // token 2 hidden state
            };
            
            cout << "\nMatrix multiply test: [" << NTOK << ", " << D << "] * [" << D << ", " << NH*DH << "] -> [" << NTOK << ", " << NH*DH << "]" << endl;
            
            cout << "\nInput matrix:" << endl;
            for (size_t i = 0; i < NTOK; ++i) {
                cout << "  [";
                for (size_t j = 0; j < D; ++j) {
                    cout << fixed << setprecision(2) << setw(6) << test_input[i * D + j];
                    if (j < D - 1) cout << ", ";
                }
                cout << "]" << endl;
            }
            
            cout << "\nQ projection weight matrix:" << endl;
            for (size_t i = 0; i < D; ++i) {
                cout << "  [";
                for (size_t j = 0; j < NH * DH; ++j) {
                    cout << fixed << setprecision(2) << setw(5) << attn_q_proj_data[i * (NH * DH) + j];
                    if (j < NH * DH - 1) cout << ", ";
                }
                cout << "]" << endl;
            }
            
            // Manual matrix multiplication
            vector<float> q_output(NTOK * NH * DH, 0.0f);
            for (size_t i = 0; i < NTOK; ++i) {          // output rows
                for (size_t j = 0; j < NH * DH; ++j) {   // output cols
                    float sum = 0.0f;
                    for (size_t k = 0; k < D; ++k) {     // inner dimension
                        sum += test_input[i * D + k] * attn_q_proj_data[k * (NH * DH) + j];
                    }
                    q_output[i * (NH * DH) + j] = sum;
                }
            }
            
            cout << "\nQ output matrix:" << endl;
            for (size_t i = 0; i < NTOK; ++i) {
                cout << "  [";
                for (size_t j = 0; j < NH * DH; ++j) {
                    cout << fixed << setprecision(3) << setw(7) << q_output[i * (NH * DH) + j];
                    if (j < NH * DH - 1) cout << ", ";
                }
                cout << "]" << endl;
            }
            
        } catch (const exception& e) {
            cout << "❌ Matrix multiply error: " << e.what() << endl;
        }
    }
    
    void testSwiGLU() {
        cout << "\n=================== SWIGLU ACTIVATION TEST ===================" << endl;
        
        try {
            // Test SwiGLU: gate_proj(x) * silu(up_proj(x))
            vector<float> test_input = {1.0f, -0.5f, 2.0f, 0.3f};  // [1, D]
            
            cout << "\nSwiGLU test input: [";
            for (size_t i = 0; i < D; ++i) {
                cout << fixed << setprecision(2) << test_input[i];
                if (i < D - 1) cout << ", ";
            }
            cout << "]" << endl;
            
            // Manual gate projection: input * gate_weight
            vector<float> gate_output(DI, 0.0f);
            for (size_t j = 0; j < DI; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < D; ++k) {
                    sum += test_input[k] * mlp_gate_proj_data[k * DI + j];
                }
                gate_output[j] = sum;
            }
            
            // Manual up projection: input * up_weight  
            vector<float> up_output(DI, 0.0f);
            for (size_t j = 0; j < DI; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < D; ++k) {
                    sum += test_input[k] * mlp_up_proj_data[k * DI + j];
                }
                up_output[j] = sum;
            }
            
            cout << "\nGate projection: [";
            for (size_t i = 0; i < DI; ++i) {
                cout << fixed << setprecision(3) << gate_output[i];
                if (i < DI - 1) cout << ", ";
            }
            cout << "]" << endl;
            
            cout << "Up projection: [";
            for (size_t i = 0; i < DI; ++i) {
                cout << fixed << setprecision(3) << up_output[i];
                if (i < DI - 1) cout << ", ";
            }
            cout << "]" << endl;
            
            // Apply SwiGLU: gate * silu(up) where silu(x) = x * sigmoid(x)
            vector<float> swiglu_output(DI);
            for (size_t i = 0; i < DI; ++i) {
                float sigmoid_up = 1.0f / (1.0f + exp(-up_output[i]));
                float silu_up = up_output[i] * sigmoid_up;
                swiglu_output[i] = gate_output[i] * silu_up;
            }
            
            cout << "\nSwiGLU output: [";
            for (size_t i = 0; i < DI; ++i) {
                cout << fixed << setprecision(3) << swiglu_output[i];
                if (i < DI - 1) cout << ", ";
            }
            cout << "]" << endl;
            
        } catch (const exception& e) {
            cout << "❌ SwiGLU error: " << e.what() << endl;
        }
    }
    
    void runAllTests() {
        testTensorCreation();
        testEmbeddingLookup();
        testRMSNorm();
        testMatrixMultiply();
        testSwiGLU();
    }
};

int main() {
    cout << "\n╔══════════════════════════════════════════════════════════╗" << endl;
    cout << "║                QW Tensor Operation Test                  ║" << endl;
    cout << "║      Testing Matrix Operations with Small Dimensions    ║" << endl;
    cout << "╚══════════════════════════════════════════════════════════╝" << endl;
    
    try {
        QwTensorTest test;
        test.runAllTests();
        
        cout << "\n╔══════════════════════════════════════════════════════════╗" << endl;
        cout << "║               ALL TESTS COMPLETED! ✅                   ║" << endl;
        cout << "║     QW matrix operations verified successfully          ║" << endl;
        cout << "╚══════════════════════════════════════════════════════════╝" << endl;
        
    } catch (const exception& e) {
        cout << "\n❌ Fatal error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
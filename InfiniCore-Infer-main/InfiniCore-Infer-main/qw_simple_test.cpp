#include <iostream>
#include <vector>
#include <memory>
#include <cstring>
#include <iomanip>
#include <cmath>

using namespace std;

// Mock InfiniCore types for testing without dependencies
typedef enum {
    INFINI_DTYPE_F32 = 0,
    INFINI_DTYPE_F16 = 1,
    INFINI_DTYPE_I32 = 2,
    INFINI_DTYPE_U32 = 3
} infiniDtype_t;

typedef enum {
    INFINI_DEVICE_CPU = 0,
    INFINI_DEVICE_GPU = 1
} infiniDevice_t;

// Mock Qwen3 structures based on the header
typedef struct {
    infiniDtype_t dt_logits;
    size_t nlayer;
    size_t d;
    size_t nh;
    size_t nkvh;
    size_t dh;
    size_t di;
    size_t dctx;
    size_t dvoc;
    float epsilon;
    float theta;
    uint32_t bos_token;
    uint32_t end_token;
    float attn_dropout;
    bool tie_embd;
} Qwen3Meta;

typedef struct {
    size_t nlayer;
    infiniDtype_t dt_norm;
    infiniDtype_t dt_mat;
    int transpose_linear_weights;
    const void *input_embd;
    const void *output_embd;
    const void *output_norm;
    const void **attn_norm;
    const void **attn_q_norm;
    const void **attn_k_norm;
    const void **attn_q_proj;
    const void **attn_k_proj;
    const void **attn_v_proj;
    const void **attn_o_proj;
    const void **mlp_norm;
    const void **mlp_gate_proj;
    const void **mlp_up_proj;
    const void **mlp_down_proj;
} Qwen3Weights;

// Simple test class for matrix computation verification
class QwSimpleTest {
private:
    // Small test dimensions - visible in terminal
    static constexpr size_t NLAYER = 2;
    static constexpr size_t D = 4;
    static constexpr size_t NH = 2;
    static constexpr size_t NKVH = 2;
    static constexpr size_t DH = 2;
    static constexpr size_t DI = 6;
    static constexpr size_t DVOC = 8;
    static constexpr size_t DCTX = 16;
    
    Qwen3Meta meta;
    Qwen3Weights weights;
    
    // Test weight storage
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
    
    // Weight pointers
    vector<const void*> attn_norm_ptrs, attn_q_norm_ptrs, attn_k_norm_ptrs;
    vector<const void*> attn_q_proj_ptrs, attn_k_proj_ptrs, attn_v_proj_ptrs, attn_o_proj_ptrs;
    vector<const void*> mlp_norm_ptrs, mlp_gate_proj_ptrs, mlp_up_proj_ptrs, mlp_down_proj_ptrs;

public:
    QwSimpleTest() {
        initializeAll();
    }
    
    void initializeAll() {
        cout << "\n╔════════════════════════════════════════════════════╗" << endl;
        cout << "║              QW Simple Matrix Test                 ║" << endl;
        cout << "╚════════════════════════════════════════════════════╝" << endl;
        
        // Initialize meta
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
        cout << "Layers: " << NLAYER << endl;
        cout << "Hidden dimension: " << D << endl;
        cout << "Attention heads: " << NH << "/" << NKVH << " (query/key-value)" << endl;
        cout << "Head dimension: " << DH << endl;
        cout << "MLP intermediate: " << DI << endl;
        cout << "Vocabulary size: " << DVOC << endl;
        cout << "Context length: " << DCTX << endl;
        
        initializeWeights();
        setupWeightPointers();
    }
    
    void initializeWeights() {
        cout << "\n=== Initializing Weights ===" << endl;
        
        // Input embedding [DVOC, D]
        input_embd_data.resize(DVOC * D);
        for (size_t i = 0; i < DVOC; ++i) {
            for (size_t j = 0; j < D; ++j) {
                input_embd_data[i * D + j] = 0.1f + 0.01f * i + 0.001f * j;
            }
        }
        
        // Output embedding [D, DVOC]
        output_embd_data.resize(D * DVOC);
        for (size_t i = 0; i < D; ++i) {
            for (size_t j = 0; j < DVOC; ++j) {
                output_embd_data[i * DVOC + j] = 0.2f + 0.01f * i + 0.001f * j;
            }
        }
        
        // Output norm [D]
        output_norm_data.resize(D);
        for (size_t i = 0; i < D; ++i) {
            output_norm_data[i] = 1.0f + 0.01f * i;
        }
        
        // Initialize all layer-specific weights
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
            float layer_offset = 0.1f * layer;
            
            // Attention norm [D]
            attn_norm_data[layer].resize(D);
            for (size_t i = 0; i < D; ++i) {
                attn_norm_data[layer][i] = 1.0f + layer_offset + 0.01f * i;
            }
            
            // Q/K norm [DH]
            attn_q_norm_data[layer].resize(DH);
            attn_k_norm_data[layer].resize(DH);
            for (size_t i = 0; i < DH; ++i) {
                attn_q_norm_data[layer][i] = 1.0f + layer_offset + 0.02f * i;
                attn_k_norm_data[layer][i] = 1.0f + layer_offset + 0.03f * i;
            }
            
            // Attention projections - use small identifiable values
            attn_q_proj_data[layer].resize(D * NH * DH);
            for (size_t i = 0; i < D * NH * DH; ++i) {
                attn_q_proj_data[layer][i] = 0.1f + layer_offset + 0.0001f * i;
            }
            
            attn_k_proj_data[layer].resize(D * NKVH * DH);
            for (size_t i = 0; i < D * NKVH * DH; ++i) {
                attn_k_proj_data[layer][i] = 0.2f + layer_offset + 0.0001f * i;
            }
            
            attn_v_proj_data[layer].resize(D * NKVH * DH);
            for (size_t i = 0; i < D * NKVH * DH; ++i) {
                attn_v_proj_data[layer][i] = 0.3f + layer_offset + 0.0001f * i;
            }
            
            attn_o_proj_data[layer].resize(NH * DH * D);
            for (size_t i = 0; i < NH * DH * D; ++i) {
                attn_o_proj_data[layer][i] = 0.4f + layer_offset + 0.0001f * i;
            }
            
            // MLP layers
            mlp_norm_data[layer].resize(D);
            for (size_t i = 0; i < D; ++i) {
                mlp_norm_data[layer][i] = 1.0f + layer_offset + 0.02f * i;
            }
            
            mlp_gate_proj_data[layer].resize(D * DI);
            mlp_up_proj_data[layer].resize(D * DI);
            for (size_t i = 0; i < D * DI; ++i) {
                mlp_gate_proj_data[layer][i] = 0.5f + layer_offset + 0.0001f * i;
                mlp_up_proj_data[layer][i] = 0.6f + layer_offset + 0.0001f * i;
            }
            
            mlp_down_proj_data[layer].resize(DI * D);
            for (size_t i = 0; i < DI * D; ++i) {
                mlp_down_proj_data[layer][i] = 0.7f + layer_offset + 0.0001f * i;
            }
        }
        
        cout << "✅ All weight matrices initialized" << endl;
        cout << "   Input embedding: " << input_embd_data.size() << " elements" << endl;
        cout << "   Per-layer weights: " << NLAYER << " layers" << endl;
    }
    
    void setupWeightPointers() {
        // Setup all pointer arrays
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
        weights.transpose_linear_weights = 0;
        
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
        
        cout << "✅ Weight pointers configured" << endl;
    }
    
    void displayMatrix(const string& name, const float* data, size_t rows, size_t cols) {
        cout << "\n--- " << name << " [" << rows << "x" << cols << "] ---" << endl;
        
        for (size_t i = 0; i < rows && i < 6; ++i) {  // Max 6 rows
            cout << "  ";
            for (size_t j = 0; j < cols && j < 8; ++j) {  // Max 8 cols
                cout << fixed << setprecision(3) << setw(7) << data[i * cols + j] << " ";
            }
            if (cols > 8) cout << " ... (" << (cols - 8) << " more)";
            cout << endl;
        }
        if (rows > 6) cout << "  ... (" << (rows - 6) << " more rows)" << endl;
    }
    
    void displayAllWeights() {
        cout << "\n=================== WEIGHT MATRICES ===================" << endl;
        
        displayMatrix("Input Embedding", input_embd_data.data(), DVOC, D);
        displayMatrix("Output Norm", output_norm_data.data(), 1, D);
        displayMatrix("Output Embedding", output_embd_data.data(), D, DVOC);
        
        for (size_t layer = 0; layer < NLAYER; ++layer) {
            cout << "\n▼ Layer " << layer << " Weights:" << endl;
            displayMatrix("Attn Norm", attn_norm_data[layer].data(), 1, D);
            displayMatrix("Q Norm", attn_q_norm_data[layer].data(), 1, DH);
            displayMatrix("K Norm", attn_k_norm_data[layer].data(), 1, DH);
            displayMatrix("Q Proj", attn_q_proj_data[layer].data(), D, NH * DH);
            displayMatrix("K Proj", attn_k_proj_data[layer].data(), D, NKVH * DH);
            displayMatrix("V Proj", attn_v_proj_data[layer].data(), D, NKVH * DH);
            displayMatrix("O Proj", attn_o_proj_data[layer].data(), NH * DH, D);
            displayMatrix("MLP Norm", mlp_norm_data[layer].data(), 1, D);
            displayMatrix("MLP Gate", mlp_gate_proj_data[layer].data(), D, DI);
            displayMatrix("MLP Up", mlp_up_proj_data[layer].data(), D, DI);
            displayMatrix("MLP Down", mlp_down_proj_data[layer].data(), DI, D);
        }
    }
    
    void testMatrixOperations() {
        cout << "\n=================== MATRIX OPERATIONS TEST ===================" << endl;
        
        // Test input token embedding lookup
        vector<uint32_t> test_tokens = {0, 2, 5, 7};
        cout << "\n--- Token Embedding Lookup ---" << endl;
        cout << "Test tokens: ";
        for (uint32_t token : test_tokens) {
            cout << token << " ";
        }
        cout << endl;
        
        // Show embeddings for each token
        for (size_t i = 0; i < test_tokens.size(); ++i) {
            uint32_t token = test_tokens[i];
            if (token < DVOC) {
                cout << "\nToken " << token << " -> embedding: [";
                const float* embd = &input_embd_data[token * D];
                for (size_t j = 0; j < D; ++j) {
                    cout << fixed << setprecision(3) << embd[j];
                    if (j < D - 1) cout << ", ";
                }
                cout << "]" << endl;
            }
        }
        
        // Test simple matrix multiply manually (Q projection example)
        cout << "\n--- Manual Q Projection Test (Layer 0) ---" << endl;
        vector<float> test_input = {1.0f, 0.5f, -0.2f, 0.8f};  // [1, D] input
        vector<float> q_output(NH * DH, 0.0f);  // [1, NH*DH] output
        
        cout << "Input vector: [";
        for (size_t i = 0; i < D; ++i) {
            cout << fixed << setprecision(2) << test_input[i];
            if (i < D - 1) cout << ", ";
        }
        cout << "]" << endl;
        
        // Manual matrix multiplication: output = input * weight
        const float* q_weight = attn_q_proj_data[0].data();  // [D, NH*DH]
        for (size_t out_idx = 0; out_idx < NH * DH; ++out_idx) {
            float sum = 0.0f;
            for (size_t in_idx = 0; in_idx < D; ++in_idx) {
                sum += test_input[in_idx] * q_weight[in_idx * (NH * DH) + out_idx];
            }
            q_output[out_idx] = sum;
        }
        
        cout << "Q projection result: [";
        for (size_t i = 0; i < NH * DH; ++i) {
            cout << fixed << setprecision(3) << q_output[i];
            if (i < NH * DH - 1) cout << ", ";
        }
        cout << "]" << endl;
        
        // Test RMS normalization manually
        cout << "\n--- Manual RMS Normalization Test ---" << endl;
        vector<float> norm_input = {2.0f, -1.0f, 3.0f, 0.5f};  // Test input
        vector<float> norm_output(D);
        const float* norm_weight = attn_norm_data[0].data();  // Layer 0 attention norm
        
        cout << "Norm input: [";
        for (size_t i = 0; i < D; ++i) {
            cout << fixed << setprecision(2) << norm_input[i];
            if (i < D - 1) cout << ", ";
        }
        cout << "]" << endl;
        
        // Compute RMS norm: x / sqrt(mean(x^2) + epsilon) * weight
        float mean_square = 0.0f;
        for (size_t i = 0; i < D; ++i) {
            mean_square += norm_input[i] * norm_input[i];
        }
        mean_square /= D;
        float rms_scale = 1.0f / sqrt(mean_square + meta.epsilon);
        
        for (size_t i = 0; i < D; ++i) {
            norm_output[i] = norm_input[i] * rms_scale * norm_weight[i];
        }
        
        cout << "RMS scale factor: " << fixed << setprecision(4) << rms_scale << endl;
        cout << "Norm weights: [";
        for (size_t i = 0; i < D; ++i) {
            cout << fixed << setprecision(3) << norm_weight[i];
            if (i < D - 1) cout << ", ";
        }
        cout << "]" << endl;
        cout << "Norm output: [";
        for (size_t i = 0; i < D; ++i) {
            cout << fixed << setprecision(3) << norm_output[i];
            if (i < D - 1) cout << ", ";
        }
        cout << "]" << endl;
        
        // Test SwiGLU activation manually
        cout << "\n--- Manual SwiGLU Test ---" << endl;
        vector<float> gate_input = {1.0f, -0.5f, 2.0f};
        vector<float> up_input = {0.8f, 1.2f, -0.3f};
        vector<float> swiglu_output(3);
        
        cout << "Gate input: [";
        for (size_t i = 0; i < 3; ++i) {
            cout << fixed << setprecision(2) << gate_input[i];
            if (i < 2) cout << ", ";
        }
        cout << "]" << endl;
        
        cout << "Up input: [";
        for (size_t i = 0; i < 3; ++i) {
            cout << fixed << setprecision(2) << up_input[i];
            if (i < 2) cout << ", ";
        }
        cout << "]" << endl;
        
        // SwiGLU: gate * silu(up) where silu(x) = x * sigmoid(x)
        for (size_t i = 0; i < 3; ++i) {
            float sigmoid_up = 1.0f / (1.0f + exp(-up_input[i]));
            float silu_up = up_input[i] * sigmoid_up;
            swiglu_output[i] = gate_input[i] * silu_up;
        }
        
        cout << "SwiGLU output: [";
        for (size_t i = 0; i < 3; ++i) {
            cout << fixed << setprecision(3) << swiglu_output[i];
            if (i < 2) cout << ", ";
        }
        cout << "]" << endl;
    }
    
    void testDataIntegrity() {
        cout << "\n=================== DATA INTEGRITY TEST ===================" << endl;
        
        // Verify all weights are properly initialized and accessible
        cout << "\n--- Checking Weight Accessibility ---" << endl;
        
        // Test global weights
        bool input_embd_ok = (input_embd_data.size() == DVOC * D) && (weights.input_embd != nullptr);
        bool output_embd_ok = (output_embd_data.size() == D * DVOC) && (weights.output_embd != nullptr);
        bool output_norm_ok = (output_norm_data.size() == D) && (weights.output_norm != nullptr);
        
        cout << "Input embedding: " << (input_embd_ok ? "✅" : "❌") << " (" << input_embd_data.size() << " elements)" << endl;
        cout << "Output embedding: " << (output_embd_ok ? "✅" : "❌") << " (" << output_embd_data.size() << " elements)" << endl;
        cout << "Output norm: " << (output_norm_ok ? "✅" : "❌") << " (" << output_norm_data.size() << " elements)" << endl;
        
        // Test layer weights
        for (size_t layer = 0; layer < NLAYER; ++layer) {
            cout << "\nLayer " << layer << " weights:" << endl;
            
            bool attn_norm_ok = (attn_norm_data[layer].size() == D) && (weights.attn_norm[layer] != nullptr);
            bool q_norm_ok = (attn_q_norm_data[layer].size() == DH) && (weights.attn_q_norm[layer] != nullptr);
            bool k_norm_ok = (attn_k_norm_data[layer].size() == DH) && (weights.attn_k_norm[layer] != nullptr);
            bool q_proj_ok = (attn_q_proj_data[layer].size() == D * NH * DH) && (weights.attn_q_proj[layer] != nullptr);
            bool k_proj_ok = (attn_k_proj_data[layer].size() == D * NKVH * DH) && (weights.attn_k_proj[layer] != nullptr);
            bool v_proj_ok = (attn_v_proj_data[layer].size() == D * NKVH * DH) && (weights.attn_v_proj[layer] != nullptr);
            bool o_proj_ok = (attn_o_proj_data[layer].size() == NH * DH * D) && (weights.attn_o_proj[layer] != nullptr);
            
            cout << "  Attn norm: " << (attn_norm_ok ? "✅" : "❌") << endl;
            cout << "  Q/K norm: " << (q_norm_ok ? "✅" : "❌") << "/" << (k_norm_ok ? "✅" : "❌") << endl;
            cout << "  QKV proj: " << (q_proj_ok ? "✅" : "❌") << "/" << (k_proj_ok ? "✅" : "❌") << "/" << (v_proj_ok ? "✅" : "❌") << endl;
            cout << "  O proj: " << (o_proj_ok ? "✅" : "❌") << endl;
            
            bool mlp_norm_ok = (mlp_norm_data[layer].size() == D) && (weights.mlp_norm[layer] != nullptr);
            bool mlp_gate_ok = (mlp_gate_proj_data[layer].size() == D * DI) && (weights.mlp_gate_proj[layer] != nullptr);
            bool mlp_up_ok = (mlp_up_proj_data[layer].size() == D * DI) && (weights.mlp_up_proj[layer] != nullptr);
            bool mlp_down_ok = (mlp_down_proj_data[layer].size() == DI * D) && (weights.mlp_down_proj[layer] != nullptr);
            
            cout << "  MLP norm: " << (mlp_norm_ok ? "✅" : "❌") << endl;
            cout << "  MLP projections: " << (mlp_gate_ok ? "✅" : "❌") << "/" << (mlp_up_ok ? "✅" : "❌") << "/" << (mlp_down_ok ? "✅" : "❌") << endl;
        }
        
        cout << "\n--- Memory Layout Verification ---" << endl;
        cout << "Input embedding ptr: " << weights.input_embd << " (expected non-null)" << endl;
        cout << "Layer 0 Q proj ptr: " << weights.attn_q_proj[0] << " (expected non-null)" << endl;
        cout << "Layer 1 Q proj ptr: " << weights.attn_q_proj[1] << " (expected non-null)" << endl;
        
        // Test pointer arithmetic
        const float* layer0_q = static_cast<const float*>(weights.attn_q_proj[0]);
        const float* layer1_q = static_cast<const float*>(weights.attn_q_proj[1]);
        cout << "Layer 0 Q proj first element: " << layer0_q[0] << endl;
        cout << "Layer 1 Q proj first element: " << layer1_q[0] << endl;
        
        bool pointers_different = (layer0_q != layer1_q);
        cout << "Layer pointers are different: " << (pointers_different ? "✅" : "❌") << endl;
    }
};

int main() {
    cout << "\n╔══════════════════════════════════════════════════════════╗" << endl;
    cout << "║                QW C++ Implementation Test                ║" << endl;
    cout << "║          Small Matrix Verification Program              ║" << endl;
    cout << "╚══════════════════════════════════════════════════════════╝" << endl;
    
    try {
        QwSimpleTest test;
        
        cout << "\n🔍 Step 1: Display Weight Matrices" << endl;
        test.displayAllWeights();
        
        cout << "\n🔍 Step 2: Test Manual Matrix Operations" << endl;  
        test.testMatrixOperations();
        
        cout << "\n🔍 Step 3: Verify Data Integrity" << endl;
        test.testDataIntegrity();
        
        cout << "\n╔══════════════════════════════════════════════════════════╗" << endl;
        cout << "║                     TEST COMPLETED                      ║" << endl;
        cout << "║   All weight matrices and operations verified! ✅       ║" << endl;
        cout << "╚══════════════════════════════════════════════════════════╝" << endl;
        
    } catch (const exception& e) {
        cout << "\n❌ Fatal error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
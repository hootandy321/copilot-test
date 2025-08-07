#include <iostream>
#include <vector>
#include <memory>
#include <cstring>
#include <iomanip>
#include <cmath>

using namespace std;

/**
 * QW Complete Forward Pass Test
 * 
 * This test simulates a complete forward pass through the QW model
 * with tiny dimensions that can be fully traced and verified manually.
 * All matrices and intermediate results are displayed for verification.
 */
class QwForwardPassTest {
private:
    // Minimal dimensions for complete traceability
    static constexpr size_t NLAYER = 1;      // 1 layer only
    static constexpr size_t D = 3;           // Hidden dimension 3
    static constexpr size_t NH = 1;          // 1 attention head
    static constexpr size_t NKVH = 1;        // 1 key-value head
    static constexpr size_t DH = 3;          // Head dimension 3 (D/NH)
    static constexpr size_t DI = 4;          // Intermediate dimension 4
    static constexpr size_t DVOC = 6;        // Vocabulary size 6
    static constexpr size_t SEQ_LEN = 2;     // Sequence length 2
    
    // Model weights (using simple, traceable values)
    vector<float> input_embedding;    // [DVOC, D]
    vector<float> output_embedding;   // [D, DVOC]
    vector<float> output_norm;        // [D]
    
    vector<float> attn_norm;          // [D]
    vector<float> q_norm;             // [DH] 
    vector<float> k_norm;             // [DH]
    vector<float> q_proj;             // [D, D]
    vector<float> k_proj;             // [D, D]
    vector<float> v_proj;             // [D, D]
    vector<float> o_proj;             // [D, D]
    
    vector<float> mlp_norm;           // [D]
    vector<float> gate_proj;          // [D, DI]
    vector<float> up_proj;            // [D, DI]
    vector<float> down_proj;          // [DI, D]

public:
    QwForwardPassTest() {
        initializeWeights();
    }
    
    void initializeWeights() {
        cout << "\n╔══════════════════════════════════════════════════════════╗" << endl;
        cout << "║              QW Complete Forward Pass Test               ║" << endl;
        cout << "╚══════════════════════════════════════════════════════════╝" << endl;
        cout << "\n=== Model Configuration ===" << endl;
        cout << "Layers: " << NLAYER << ", Hidden: " << D << ", Heads: " << NH << "/" << NKVH << endl;
        cout << "Vocab: " << DVOC << ", Sequence: " << SEQ_LEN << ", Intermediate: " << DI << endl;
        
        cout << "\n=== Initializing Weights ===" << endl;
        
        // Input embedding - simple patterns for easy tracking
        input_embedding.resize(DVOC * D);
        for (size_t i = 0; i < DVOC; ++i) {
            for (size_t j = 0; j < D; ++j) {
                input_embedding[i * D + j] = 0.1f * (i + 1) + 0.01f * j;
            }
        }
        
        // Output embedding
        output_embedding.resize(D * DVOC);
        for (size_t i = 0; i < D; ++i) {
            for (size_t j = 0; j < DVOC; ++j) {
                output_embedding[i * DVOC + j] = 0.05f + 0.01f * (i * DVOC + j);
            }
        }
        
        // Norms - close to 1.0 for stability
        output_norm = {1.0f, 1.1f, 1.2f};
        attn_norm = {1.0f, 1.05f, 1.1f};
        mlp_norm = {1.0f, 1.02f, 1.04f};
        q_norm = {1.0f, 1.0f, 1.0f};
        k_norm = {1.0f, 1.0f, 1.0f};
        
        // Attention projections - simple identity-like matrices
        q_proj = {
            0.8f, 0.1f, 0.1f,
            0.1f, 0.8f, 0.1f,
            0.1f, 0.1f, 0.8f
        };
        
        k_proj = {
            0.9f, 0.05f, 0.05f,
            0.05f, 0.9f, 0.05f,
            0.05f, 0.05f, 0.9f
        };
        
        v_proj = {
            0.7f, 0.15f, 0.15f,
            0.15f, 0.7f, 0.15f,
            0.15f, 0.15f, 0.7f
        };
        
        o_proj = {
            0.6f, 0.2f, 0.2f,
            0.2f, 0.6f, 0.2f,
            0.2f, 0.2f, 0.6f
        };
        
        // MLP projections - small values to prevent explosion
        gate_proj = {
            0.2f, 0.15f, 0.1f, 0.05f,
            0.05f, 0.2f, 0.15f, 0.1f,
            0.1f, 0.05f, 0.2f, 0.15f
        };
        
        up_proj = {
            0.25f, 0.2f, 0.1f, 0.05f,
            0.05f, 0.25f, 0.2f, 0.1f,
            0.1f, 0.05f, 0.25f, 0.2f
        };
        
        down_proj = {
            0.15f, 0.1f, 0.05f,
            0.05f, 0.15f, 0.1f,
            0.1f, 0.05f, 0.15f,
            0.05f, 0.1f, 0.15f
        };
        
        cout << "✅ All weights initialized with traceable values" << endl;
        displayAllWeights();
    }
    
    void displayMatrix(const string& name, const vector<float>& data, size_t rows, size_t cols) {
        cout << "\n--- " << name << " [" << rows << "x" << cols << "] ---" << endl;
        for (size_t i = 0; i < rows; ++i) {
            cout << "  [";
            for (size_t j = 0; j < cols; ++j) {
                cout << fixed << setprecision(2) << setw(5) << data[i * cols + j];
                if (j < cols - 1) cout << ", ";
            }
            cout << "]" << endl;
        }
    }
    
    void displayVector(const string& name, const vector<float>& data) {
        cout << "\n" << name << ": [";
        for (size_t i = 0; i < data.size(); ++i) {
            cout << fixed << setprecision(2) << data[i];
            if (i < data.size() - 1) cout << ", ";
        }
        cout << "]" << endl;
    }
    
    void displayAllWeights() {
        cout << "\n=================== WEIGHT MATRICES ===================" << endl;
        displayMatrix("Input Embedding", input_embedding, DVOC, D);
        displayMatrix("Output Embedding", output_embedding, D, DVOC);
        displayVector("Output Norm", output_norm);
        
        cout << "\n--- Layer 0 Weights ---" << endl;
        displayVector("Attention Norm", attn_norm);
        displayVector("Q Norm", q_norm);
        displayVector("K Norm", k_norm);
        displayMatrix("Q Projection", q_proj, D, D);
        displayMatrix("K Projection", k_proj, D, D);
        displayMatrix("V Projection", v_proj, D, D);
        displayMatrix("O Projection", o_proj, D, D);
        displayVector("MLP Norm", mlp_norm);
        displayMatrix("Gate Projection", gate_proj, D, DI);
        displayMatrix("Up Projection", up_proj, D, DI);
        displayMatrix("Down Projection", down_proj, DI, D);
    }
    
    vector<float> matmul(const vector<float>& A, const vector<float>& B, 
                        size_t A_rows, size_t A_cols, size_t B_cols) {
        vector<float> C(A_rows * B_cols, 0.0f);
        for (size_t i = 0; i < A_rows; ++i) {
            for (size_t j = 0; j < B_cols; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < A_cols; ++k) {
                    sum += A[i * A_cols + k] * B[k * B_cols + j];
                }
                C[i * B_cols + j] = sum;
            }
        }
        return C;
    }
    
    vector<float> rmsNorm(const vector<float>& input, const vector<float>& weight, 
                         size_t len, float epsilon = 1e-6f) {
        vector<float> output(len);
        
        // Compute RMS
        float mean_sq = 0.0f;
        for (size_t i = 0; i < len; ++i) {
            mean_sq += input[i] * input[i];
        }
        mean_sq /= len;
        float rms_scale = 1.0f / sqrt(mean_sq + epsilon);
        
        // Apply normalization with weights
        for (size_t i = 0; i < len; ++i) {
            output[i] = input[i] * rms_scale * weight[i];
        }
        
        return output;
    }
    
    vector<float> swiGLU(const vector<float>& gate, const vector<float>& up) {
        vector<float> output(gate.size());
        for (size_t i = 0; i < gate.size(); ++i) {
            float sigmoid_up = 1.0f / (1.0f + exp(-up[i]));
            float silu_up = up[i] * sigmoid_up;
            output[i] = gate[i] * silu_up;
        }
        return output;
    }
    
    vector<float> attention(const vector<float>& Q, const vector<float>& K, 
                           const vector<float>& V, size_t seq_len, size_t head_dim) {
        // Simplified attention: softmax(Q*K^T/sqrt(d)) * V
        // For simplicity, we'll compute it step by step
        
        cout << "\n--- Attention Computation ---" << endl;
        cout << "Q matrix [" << seq_len << "x" << head_dim << "]:" << endl;
        for (size_t i = 0; i < seq_len; ++i) {
            cout << "  [";
            for (size_t j = 0; j < head_dim; ++j) {
                cout << fixed << setprecision(2) << setw(5) << Q[i * head_dim + j];
                if (j < head_dim - 1) cout << ", ";
            }
            cout << "]" << endl;
        }
        
        cout << "K matrix [" << seq_len << "x" << head_dim << "]:" << endl;
        for (size_t i = 0; i < seq_len; ++i) {
            cout << "  [";
            for (size_t j = 0; j < head_dim; ++j) {
                cout << fixed << setprecision(2) << setw(5) << K[i * head_dim + j];
                if (j < head_dim - 1) cout << ", ";
            }
            cout << "]" << endl;
        }
        
        // Compute attention scores: Q * K^T / sqrt(head_dim)
        vector<float> scores(seq_len * seq_len, 0.0f);
        float scale = 1.0f / sqrt(float(head_dim));
        
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < seq_len; ++j) {
                float score = 0.0f;
                for (size_t k = 0; k < head_dim; ++k) {
                    score += Q[i * head_dim + k] * K[j * head_dim + k];
                }
                scores[i * seq_len + j] = score * scale;
            }
        }
        
        cout << "Attention scores [" << seq_len << "x" << seq_len << "]:" << endl;
        for (size_t i = 0; i < seq_len; ++i) {
            cout << "  [";
            for (size_t j = 0; j < seq_len; ++j) {
                cout << fixed << setprecision(3) << setw(6) << scores[i * seq_len + j];
                if (j < seq_len - 1) cout << ", ";
            }
            cout << "]" << endl;
        }
        
        // Apply softmax to each row
        for (size_t i = 0; i < seq_len; ++i) {
            float max_val = scores[i * seq_len];
            for (size_t j = 1; j < seq_len; ++j) {
                max_val = max(max_val, scores[i * seq_len + j]);
            }
            
            float sum_exp = 0.0f;
            for (size_t j = 0; j < seq_len; ++j) {
                scores[i * seq_len + j] = exp(scores[i * seq_len + j] - max_val);
                sum_exp += scores[i * seq_len + j];
            }
            
            for (size_t j = 0; j < seq_len; ++j) {
                scores[i * seq_len + j] /= sum_exp;
            }
        }
        
        cout << "Attention weights (after softmax):" << endl;
        for (size_t i = 0; i < seq_len; ++i) {
            cout << "  [";
            for (size_t j = 0; j < seq_len; ++j) {
                cout << fixed << setprecision(3) << setw(6) << scores[i * seq_len + j];
                if (j < seq_len - 1) cout << ", ";
            }
            cout << "]" << endl;
        }
        
        // Apply attention to values: scores * V
        vector<float> output(seq_len * head_dim, 0.0f);
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < head_dim; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < seq_len; ++k) {
                    sum += scores[i * seq_len + k] * V[k * head_dim + j];
                }
                output[i * head_dim + j] = sum;
            }
        }
        
        return output;
    }
    
    void runCompleteForwardPass() {
        cout << "\n=================== COMPLETE FORWARD PASS ===================" << endl;
        
        // Input tokens
        vector<uint32_t> tokens = {1, 3};  // Two test tokens
        cout << "\nInput tokens: [" << tokens[0] << ", " << tokens[1] << "]" << endl;
        
        // Step 1: Token embedding lookup
        cout << "\n🔍 Step 1: Token Embedding Lookup" << endl;
        vector<float> embedded(SEQ_LEN * D);
        for (size_t i = 0; i < SEQ_LEN; ++i) {
            uint32_t token = tokens[i];
            cout << "Token " << token << " -> embedding: [";
            for (size_t j = 0; j < D; ++j) {
                embedded[i * D + j] = input_embedding[token * D + j];
                cout << fixed << setprecision(2) << embedded[i * D + j];
                if (j < D - 1) cout << ", ";
            }
            cout << "]" << endl;
        }
        
        vector<float> hidden_states = embedded;
        
        // Step 2: Attention block
        cout << "\n🔍 Step 2: Attention Block" << endl;
        
        // Layer normalization before attention
        cout << "\n2.1: Pre-attention normalization" << endl;
        vector<float> normed_input(SEQ_LEN * D);
        for (size_t i = 0; i < SEQ_LEN; ++i) {
            vector<float> input_row(hidden_states.begin() + i * D, 
                                   hidden_states.begin() + (i + 1) * D);
            vector<float> normed = rmsNorm(input_row, attn_norm, D);
            
            cout << "Row " << i << " normalized: [";
            for (size_t j = 0; j < D; ++j) {
                normed_input[i * D + j] = normed[j];
                cout << fixed << setprecision(3) << normed[j];
                if (j < D - 1) cout << ", ";
            }
            cout << "]" << endl;
        }
        
        // QKV projections
        cout << "\n2.2: QKV projections" << endl;
        auto Q = matmul(normed_input, q_proj, SEQ_LEN, D, D);
        auto K = matmul(normed_input, k_proj, SEQ_LEN, D, D);
        auto V = matmul(normed_input, v_proj, SEQ_LEN, D, D);
        
        // Q/K normalization (Qwen3-specific)
        cout << "\n2.3: Q/K normalization" << endl;
        for (size_t i = 0; i < SEQ_LEN; ++i) {
            vector<float> q_row(Q.begin() + i * D, Q.begin() + (i + 1) * D);
            vector<float> k_row(K.begin() + i * D, K.begin() + (i + 1) * D);
            
            vector<float> q_normed = rmsNorm(q_row, q_norm, D);
            vector<float> k_normed = rmsNorm(k_row, k_norm, D);
            
            for (size_t j = 0; j < D; ++j) {
                Q[i * D + j] = q_normed[j];
                K[i * D + j] = k_normed[j];
            }
        }
        
        // Self-attention
        cout << "\n2.4: Self-attention computation" << endl;
        auto attn_output = attention(Q, K, V, SEQ_LEN, D);
        
        cout << "\nAttention output:" << endl;
        for (size_t i = 0; i < SEQ_LEN; ++i) {
            cout << "  [";
            for (size_t j = 0; j < D; ++j) {
                cout << fixed << setprecision(3) << setw(6) << attn_output[i * D + j];
                if (j < D - 1) cout << ", ";
            }
            cout << "]" << endl;
        }
        
        // Output projection
        auto o_output = matmul(attn_output, o_proj, SEQ_LEN, D, D);
        
        // Residual connection
        cout << "\n2.5: Residual connection" << endl;
        for (size_t i = 0; i < SEQ_LEN * D; ++i) {
            hidden_states[i] += o_output[i];
        }
        
        cout << "After attention block:" << endl;
        for (size_t i = 0; i < SEQ_LEN; ++i) {
            cout << "  [";
            for (size_t j = 0; j < D; ++j) {
                cout << fixed << setprecision(3) << setw(6) << hidden_states[i * D + j];
                if (j < D - 1) cout << ", ";
            }
            cout << "]" << endl;
        }
        
        // Step 3: MLP block
        cout << "\n🔍 Step 3: MLP Block" << endl;
        
        // Pre-MLP normalization
        vector<float> mlp_normed(SEQ_LEN * D);
        for (size_t i = 0; i < SEQ_LEN; ++i) {
            vector<float> input_row(hidden_states.begin() + i * D,
                                   hidden_states.begin() + (i + 1) * D);
            vector<float> normed = rmsNorm(input_row, mlp_norm, D);
            
            for (size_t j = 0; j < D; ++j) {
                mlp_normed[i * D + j] = normed[j];
            }
        }
        
        // Gate and Up projections
        auto gate_output = matmul(mlp_normed, gate_proj, SEQ_LEN, D, DI);
        auto up_output = matmul(mlp_normed, up_proj, SEQ_LEN, D, DI);
        
        cout << "Gate projection:" << endl;
        for (size_t i = 0; i < SEQ_LEN; ++i) {
            cout << "  [";
            for (size_t j = 0; j < DI; ++j) {
                cout << fixed << setprecision(3) << setw(6) << gate_output[i * DI + j];
                if (j < DI - 1) cout << ", ";
            }
            cout << "]" << endl;
        }
        
        // SwiGLU activation
        vector<float> swiglu_output(SEQ_LEN * DI);
        for (size_t i = 0; i < SEQ_LEN; ++i) {
            vector<float> gate_row(gate_output.begin() + i * DI, 
                                  gate_output.begin() + (i + 1) * DI);
            vector<float> up_row(up_output.begin() + i * DI,
                                up_output.begin() + (i + 1) * DI);
            
            vector<float> activated = swiGLU(gate_row, up_row);
            for (size_t j = 0; j < DI; ++j) {
                swiglu_output[i * DI + j] = activated[j];
            }
        }
        
        cout << "SwiGLU output:" << endl;
        for (size_t i = 0; i < SEQ_LEN; ++i) {
            cout << "  [";
            for (size_t j = 0; j < DI; ++j) {
                cout << fixed << setprecision(3) << setw(6) << swiglu_output[i * DI + j];
                if (j < DI - 1) cout << ", ";
            }
            cout << "]" << endl;
        }
        
        // Down projection
        auto down_output = matmul(swiglu_output, down_proj, SEQ_LEN, DI, D);
        
        // Residual connection
        for (size_t i = 0; i < SEQ_LEN * D; ++i) {
            hidden_states[i] += down_output[i];
        }
        
        cout << "After MLP block:" << endl;
        for (size_t i = 0; i < SEQ_LEN; ++i) {
            cout << "  [";
            for (size_t j = 0; j < D; ++j) {
                cout << fixed << setprecision(3) << setw(6) << hidden_states[i * D + j];
                if (j < D - 1) cout << ", ";
            }
            cout << "]" << endl;
        }
        
        // Step 4: Output prediction (last token only)
        cout << "\n🔍 Step 4: Output Prediction" << endl;
        
        // Final normalization (last token)
        size_t last_token_idx = SEQ_LEN - 1;
        vector<float> last_hidden(hidden_states.begin() + last_token_idx * D,
                                 hidden_states.begin() + (last_token_idx + 1) * D);
        
        vector<float> final_normed = rmsNorm(last_hidden, output_norm, D);
        cout << "Final normalized state: [";
        for (size_t i = 0; i < D; ++i) {
            cout << fixed << setprecision(3) << final_normed[i];
            if (i < D - 1) cout << ", ";
        }
        cout << "]" << endl;
        
        // Output projection to vocabulary
        auto logits = matmul(final_normed, output_embedding, 1, D, DVOC);
        
        cout << "Output logits: [";
        for (size_t i = 0; i < DVOC; ++i) {
            cout << fixed << setprecision(3) << setw(7) << logits[i];
            if (i < DVOC - 1) cout << ", ";
        }
        cout << "]" << endl;
        
        // Find predicted token (argmax)
        size_t predicted_token = 0;
        float max_logit = logits[0];
        for (size_t i = 1; i < DVOC; ++i) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                predicted_token = i;
            }
        }
        
        cout << "\n✨ FINAL RESULT ✨" << endl;
        cout << "Input sequence: [" << tokens[0] << ", " << tokens[1] << "]" << endl;
        cout << "Predicted next token: " << predicted_token << " (logit: " << fixed << setprecision(3) << max_logit << ")" << endl;
        
        cout << "\n╔══════════════════════════════════════════════════════════╗" << endl;
        cout << "║         FORWARD PASS COMPLETED SUCCESSFULLY! ✅         ║" << endl;
        cout << "╚══════════════════════════════════════════════════════════╝" << endl;
    }
};

int main() {
    cout << "\n╔══════════════════════════════════════════════════════════╗" << endl;
    cout << "║           QW Complete Forward Pass Verification          ║" << endl;
    cout << "║         Manual Implementation with Tiny Matrices        ║" << endl;
    cout << "╚══════════════════════════════════════════════════════════╝" << endl;
    
    try {
        QwForwardPassTest test;
        test.runCompleteForwardPass();
        
    } catch (const exception& e) {
        cout << "\n❌ Fatal error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
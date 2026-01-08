/**
 * JEPA v2 - Aprendizado de Padroes Temporais em C++
 * 
 * Implementacao minimalista de JEPA (Joint Embedding Predictive Architecture)
 * aplicada a series temporais. O modelo aprende a prever representacoes
 * de segmentos futuros a partir do contexto passado.
 * 
 * Compile (MSVC):  cl /EHsc /O2 v2-temporal-patterns.cpp /Fe:jepa_temporal.exe
 * Compile (GCC):   g++ -O2 v2-temporal-patterns.cpp -o jepa_temporal
 * Execute: ./jepa_temporal ou jepa_temporal.exe
 */

#define _USE_MATH_DEFINES  // Necessario para M_PI no MSVC
#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <iomanip>
#include <string>

using Vec = std::vector<double>;
using Matrix = std::vector<Vec>;

// ============== Gerador de numeros aleatorios ==============
std::mt19937 rng(42);
std::uniform_real_distribution<double> uniform(-1.0, 1.0);

double rand_uniform(double a = -0.2, double b = 0.2) {
    return a + (b - a) * (uniform(rng) + 1.0) / 2.0;
}

// ============== Operacoes Vetoriais ==============
inline double tanh_act(double x) { return std::tanh(x); }
inline double dtanh(double y) { return 1.0 - y * y; }

Vec add(const Vec& a, const Vec& b) {
    Vec r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] + b[i];
    return r;
}

Vec subtract(const Vec& a, const Vec& b) {
    Vec r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] - b[i];
    return r;
}

Vec scale(const Vec& a, double s) {
    Vec r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] * s;
    return r;
}

double dot(const Vec& a, const Vec& b) {
    double sum = 0;
    for (size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
    return sum;
}

double norm(const Vec& a) { return std::sqrt(dot(a, a)); }

double cosine_sim(const Vec& a, const Vec& b) {
    double na = norm(a), nb = norm(b);
    if (na < 1e-9 || nb < 1e-9) return 0.0;
    return dot(a, b) / (na * nb);
}

Vec apply_tanh(const Vec& x) {
    Vec r(x.size());
    for (size_t i = 0; i < x.size(); ++i) r[i] = tanh_act(x[i]);
    return r;
}

std::pair<double, Vec> mse_loss(const Vec& pred, const Vec& target) {
    Vec grad(pred.size());
    double loss = 0;
    double n = pred.size();
    for (size_t i = 0; i < pred.size(); ++i) {
        double d = pred[i] - target[i];
        loss += d * d;
        grad[i] = 2.0 * d / n;
    }
    return {loss / n, grad};
}

// ============== Camada Linear ==============
struct Linear {
    Matrix W;
    Vec b;
    Vec last_x;
    int din, dout;
    
    Linear(int in_dim, int out_dim) : din(in_dim), dout(out_dim) {
        W.resize(out_dim, Vec(in_dim));
        b.resize(out_dim, 0.0);
        for (auto& row : W)
            for (double& w : row)
                w = rand_uniform(-0.2, 0.2);
    }
    
    Vec forward(const Vec& x) {
        last_x = x;
        Vec out(dout, 0.0);
        for (int o = 0; o < dout; ++o) {
            out[o] = b[o];
            for (int i = 0; i < din; ++i)
                out[o] += W[o][i] * x[i];
        }
        return out;
    }
    
    Vec backward(const Vec& dy, double lr) {
        Vec dx(din, 0.0);
        for (int o = 0; o < dout; ++o) {
            b[o] -= lr * dy[o];
            for (int i = 0; i < din; ++i) {
                dx[i] += W[o][i] * dy[o];
                W[o][i] -= lr * dy[o] * last_x[i];
            }
        }
        return dx;
    }
};

// ============== MLP (2 camadas) ==============
struct MLP {
    Linear layer1, layer2;
    Vec hidden;
    
    MLP(int in_dim, int hidden_dim, int out_dim)
        : layer1(in_dim, hidden_dim), layer2(hidden_dim, out_dim) {}
    
    Vec forward(const Vec& x) {
        hidden = apply_tanh(layer1.forward(x));
        return layer2.forward(hidden);
    }
    
    Vec backward(const Vec& dy, double lr) {
        Vec dh = layer2.backward(dy, lr);
        for (size_t i = 0; i < dh.size(); ++i)
            dh[i] *= dtanh(hidden[i]);
        return layer1.backward(dh, lr);
    }
};

// ============== EMA Update (para target encoder) ==============
void ema_update(MLP& target, const MLP& online, double momentum = 0.995) {
    auto update_linear = [momentum](Linear& tgt, const Linear& src) {
        for (int o = 0; o < tgt.dout; ++o) {
            tgt.b[o] = momentum * tgt.b[o] + (1 - momentum) * src.b[o];
            for (int i = 0; i < tgt.din; ++i)
                tgt.W[o][i] = momentum * tgt.W[o][i] + (1 - momentum) * src.W[o][i];
        }
    };
    update_linear(target.layer1, online.layer1);
    update_linear(target.layer2, online.layer2);
}

// ============== Geracao de Series Temporais ==============
// Gera padroes interessantes: ondas, rampas, ruido estruturado
enum class Pattern { SINE, RAMP, PULSE, NOISY_SINE, SAWTOOTH };

Vec generate_series(Pattern p, int length, double phase = 0.0) {
    Vec series(length);
    double freq = 0.5 + rand_uniform(0, 0.5);
    double amp = 0.8 + rand_uniform(0, 0.2);
    
    for (int i = 0; i < length; ++i) {
        double t = (i + phase) / length;
        switch (p) {
            case Pattern::SINE:
                series[i] = amp * std::sin(2 * M_PI * freq * t * 4);
                break;
            case Pattern::RAMP:
                series[i] = amp * (std::fmod(t * 3, 1.0) * 2 - 1);
                break;
            case Pattern::PULSE:
                series[i] = amp * (std::sin(2 * M_PI * t * 2) > 0 ? 1.0 : -1.0);
                break;
            case Pattern::NOISY_SINE:
                series[i] = amp * std::sin(2 * M_PI * freq * t * 4) + rand_uniform(-0.1, 0.1);
                break;
            case Pattern::SAWTOOTH:
                series[i] = amp * (2 * std::fmod(t * 5 + 0.5, 1.0) - 1);
                break;
        }
    }
    return series;
}

std::string pattern_name(Pattern p) {
    switch (p) {
        case Pattern::SINE: return "Sine";
        case Pattern::RAMP: return "Ramp";
        case Pattern::PULSE: return "Pulse";
        case Pattern::NOISY_SINE: return "NoisySine";
        case Pattern::SAWTOOTH: return "Sawtooth";
    }
    return "???";
}

// ============== Dataset de Treino ==============
struct Sample {
    Vec context;   // Janela de contexto (passado)
    Vec target;    // Janela alvo (futuro a prever no espaco latente)
    Pattern type;
};

std::vector<Sample> generate_dataset(int num_samples, int ctx_len, int tgt_len) {
    std::vector<Sample> data;
    std::vector<Pattern> patterns = {
        Pattern::SINE, Pattern::RAMP, Pattern::PULSE, 
        Pattern::NOISY_SINE, Pattern::SAWTOOTH
    };
    
    for (int i = 0; i < num_samples; ++i) {
        Pattern p = patterns[i % patterns.size()];
        double phase = rand_uniform(0, 100);
        Vec full = generate_series(p, ctx_len + tgt_len, phase);
        
        Sample s;
        s.context = Vec(full.begin(), full.begin() + ctx_len);
        s.target = Vec(full.begin() + ctx_len, full.end());
        s.type = p;
        data.push_back(s);
    }
    return data;
}

// ============== Treinamento JEPA ==============
void print_bar(double value, int width = 30) {
    int filled = std::min(width, std::max(0, static_cast<int>(value * width)));
    std::cout << "[";
    for (int i = 0; i < filled; ++i) std::cout << '#';
    for (int i = filled; i < width; ++i) std::cout << '.';
    std::cout << "]";
}

void train_jepa() {
    // Hiperparametros
    constexpr int CTX_LEN = 16;      // Janela de contexto
    constexpr int TGT_LEN = 8;       // Janela alvo
    constexpr int LATENT_DIM = 16;   // Dimensao do espaco latente
    constexpr int HIDDEN_DIM = 32;   // Neuronios na camada oculta
    constexpr int EPOCHS = 30;
    constexpr double LR = 0.01;
    constexpr double EMA_MOMENTUM = 0.995;
    constexpr int BATCH_SIZE = 128;
    
    std::cout << "\n";
    std::cout << "+============================================================+\n";
    std::cout << "|       [*] JEPA v2 - Temporal Patterns in C++               |\n";
    std::cout << "+============================================================+\n";
    std::cout << "|  Context: " << CTX_LEN << " samples   |  Target: " << TGT_LEN << " samples              |\n";
    std::cout << "|  Latent: " << LATENT_DIM << " dims      |  Hidden: " << HIDDEN_DIM << " neurons              |\n";
    std::cout << "+============================================================+\n\n";
    
    // Modelos
    MLP context_encoder(CTX_LEN, HIDDEN_DIM, LATENT_DIM);
    MLP target_encoder(TGT_LEN, HIDDEN_DIM, LATENT_DIM);
    MLP predictor(LATENT_DIM, HIDDEN_DIM, LATENT_DIM);
    
    // Inicializa target encoder com pesos do context (simplificacao)
    target_encoder = MLP(TGT_LEN, HIDDEN_DIM, LATENT_DIM);
    
    std::cout << "[PRETRAIN] JEPA Self-Supervised Training\n";
    std::cout << "-------------------------------------------------\n";
    
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        auto dataset = generate_dataset(BATCH_SIZE, CTX_LEN, TGT_LEN);
        double total_loss = 0;
        
        for (auto& sample : dataset) {
            // Forward pass
            Vec z_ctx = context_encoder.forward(sample.context);
            Vec z_tgt = target_encoder.forward(sample.target);  // stop-gradient
            Vec z_pred = predictor.forward(z_ctx);
            
            // Loss: predicao deve bater com embedding alvo
            std::pair<double, Vec> loss_result = mse_loss(z_pred, z_tgt);
            double loss = loss_result.first;
            Vec grad = loss_result.second;
            total_loss += loss;
            
            // Backward pass (so online encoder + predictor)
            Vec dz_ctx = predictor.backward(grad, LR);
            context_encoder.backward(dz_ctx, LR);
            
            // EMA update do target encoder
            ema_update(target_encoder, context_encoder, EMA_MOMENTUM);
        }
        
        double avg_loss = total_loss / BATCH_SIZE;
        std::cout << "  Epoch " << std::setw(2) << epoch + 1 << "/" << EPOCHS 
                  << " | Loss: " << std::fixed << std::setprecision(4) << avg_loss << " ";
        print_bar(1.0 - std::min(1.0, avg_loss / 0.5));
        std::cout << "\n";
    }
    
    // ============== Avaliacao: Classificacao de Padroes ==============
    std::cout << "\n[EVAL] Pattern Similarity Analysis\n";
    std::cout << "-------------------------------------------------\n";
    
    // Gera embeddings representativos de cada padrao
    std::vector<Pattern> patterns = {
        Pattern::SINE, Pattern::RAMP, Pattern::PULSE, 
        Pattern::NOISY_SINE, Pattern::SAWTOOTH
    };
    
    std::vector<Vec> pattern_embeddings;
    for (Pattern p : patterns) {
        Vec avg_emb(LATENT_DIM, 0.0);
        for (int i = 0; i < 10; ++i) {
            Vec series = generate_series(p, CTX_LEN, i * 3);
            Vec emb = context_encoder.forward(series);
            for (int j = 0; j < LATENT_DIM; ++j)
                avg_emb[j] += emb[j] / 10.0;
        }
        pattern_embeddings.push_back(avg_emb);
    }
    
    // Matriz de similaridade
    std::cout << "\n  Cosine Similarity Matrix:\n\n";
    std::cout << "             ";
    for (auto p : patterns)
        std::cout << std::setw(12) << pattern_name(p).substr(0, 9);
    std::cout << "\n";
    
    for (size_t i = 0; i < patterns.size(); ++i) {
        std::cout << "  " << std::setw(10) << pattern_name(patterns[i]);
        for (size_t j = 0; j < patterns.size(); ++j) {
            double sim = cosine_sim(pattern_embeddings[i], pattern_embeddings[j]);
            std::cout << std::setw(12) << std::fixed << std::setprecision(3) << sim;
        }
        std::cout << "\n";
    }
    
    // ============== Demo: Pattern Classification via Embeddings ==============
    std::cout << "\n[DEMO] Pattern Classification (1-NN in embedding space)\n";
    std::cout << "-------------------------------------------------\n";
    std::cout << "  Testing if the model can recognize patterns...\n\n";
    
    // Cria "prototipos" para cada classe (media de varios exemplos)
    std::vector<Vec> prototypes;
    for (Pattern p : patterns) {
        Vec proto(LATENT_DIM, 0.0);
        for (int i = 0; i < 20; ++i) {
            Vec series = generate_series(p, CTX_LEN, i * 7.3);
            Vec emb = context_encoder.forward(series);
            for (int j = 0; j < LATENT_DIM; ++j)
                proto[j] += emb[j] / 20.0;
        }
        prototypes.push_back(proto);
    }
    
    // Testa classificacao em novos exemplos
    int correct = 0, total = 0;
    std::vector<int> correct_per_class(5, 0);
    std::vector<int> total_per_class(5, 0);
    
    for (int t = 0; t < 50; ++t) {
        Pattern true_pattern = patterns[t % 5];
        int true_idx = t % 5;
        
        // Gera amostra com fase aleatoria diferente
        Vec test_series = generate_series(true_pattern, CTX_LEN, t * 13.7 + 50);
        Vec test_emb = context_encoder.forward(test_series);
        
        // Encontra prototipo mais proximo
        int pred_idx = 0;
        double best_sim = -999;
        for (size_t i = 0; i < prototypes.size(); ++i) {
            double sim = cosine_sim(test_emb, prototypes[i]);
            if (sim > best_sim) {
                best_sim = sim;
                pred_idx = i;
            }
        }
        
        total_per_class[true_idx]++;
        total++;
        if (pred_idx == true_idx) {
            correct++;
            correct_per_class[true_idx]++;
        }
    }
    
    // Mostra resultados por classe
    std::cout << "  Results per pattern type:\n\n";
    std::cout << "    Pattern        Accuracy\n";
    std::cout << "    ---------     ----------\n";
    
    for (size_t i = 0; i < patterns.size(); ++i) {
        double acc = 100.0 * correct_per_class[i] / total_per_class[i];
        std::cout << "    " << std::setw(10) << pattern_name(patterns[i]) 
                  << "   " << std::setw(5) << correct_per_class[i] 
                  << "/" << total_per_class[i] 
                  << " = " << std::fixed << std::setprecision(0) << acc << "%";
        
        // Mini barra visual
        int bars = (int)(acc / 10);
        std::cout << "  ";
        for (int b = 0; b < bars; ++b) std::cout << "#";
        for (int b = bars; b < 10; ++b) std::cout << ".";
        std::cout << "\n";
    }
    
    double total_acc = 100.0 * correct / total;
    std::cout << "\n    ---------     ----------\n";
    std::cout << "    TOTAL        " << std::setw(5) << correct << "/" << total 
              << " = " << std::fixed << std::setprecision(1) << total_acc << "%\n";
    
    std::cout << "\n    (Random baseline would be 20%)\n";
    
    std::cout << "\n+============================================================+\n";
    std::cout << "|  [OK] Training complete!                                   |\n";
    if (total_acc > 60) {
        std::cout << "|       The model learned useful pattern representations.    |\n";
    } else if (total_acc > 40) {
        std::cout << "|       The model learned some structure (room to improve).  |\n";
    } else {
        std::cout << "|       Model needs more training or tuning.                 |\n";
    }
    std::cout << "+============================================================+\n\n";
}

int main() {
    train_jepa();
    return 0;
}

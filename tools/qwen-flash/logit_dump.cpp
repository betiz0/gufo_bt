// Scores a text prompt with llama.cpp's libllama and dumps per-position
// logits, for the Flash-Next teacher-forced comparison against Gufo's
// gpu_probe logit dumps (see docs/models/qwen3.8-flash-next/EXPERIMENTS.md).
//
// The tool tokenizes the prompt with the model's own vocabulary, evaluates the
// prompt, greedily continues it for `--continuation N` tokens (recording those
// token ids), and then writes every position's full logit row to a binary
// file: int32 positions, int32 vocab, then positions * vocab float32 rows in
// token order. Position i holds the next-token distribution after tokens
// [0..i], so a forced sequence's per-position comparison uses positions
// [0, prompt_len + continuation - 2].
//
// Build against the Vulkan reference build:
//   g++ -O2 -std=c++17 -I$LLAMA/include -I$LLAMA/ggml/include logit_dump.cpp \
//     -L$LLAMA/build-vulkan/bin -lllama -lggml -lggml-base \
//     -Wl,-rpath,$LLAMA/build-vulkan/bin -o /tmp/logit_dump

#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

[[noreturn]] void Fail(const std::string& message) {
  std::fprintf(stderr, "logit_dump: %s\n", message.c_str());
  std::exit(1);
}

std::string ReadFile(const char* path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    Fail(std::string("cannot open ") + path);
  }
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path, prompt_path, out_path, tokens_path, continuation_path,
      pieces_path;
  int continuation = 64;
  int n_ctx = 4096;
  int n_batch = 2048;
  int n_ubatch = 0;  // llama.cpp's own default
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto next = [&]() -> std::string {
      return i + 1 < argc ? argv[++i] : std::string();
    };
    if (arg == "--model") {
      model_path = next();
    } else if (arg == "--prompt-file") {
      prompt_path = next();
    } else if (arg == "--out") {
      out_path = next();
    } else if (arg == "--tokens-out") {
      tokens_path = next();
    } else if (arg == "--continuation-out") {
      continuation_path = next();
    } else if (arg == "--pieces-out") {
      pieces_path = next();
    } else if (arg == "--continuation") {
      continuation = std::atoi(next().c_str());
    } else if (arg == "--ctx") {
      n_ctx = std::atoi(next().c_str());
    } else if (arg == "--batch") {
      n_batch = std::atoi(next().c_str());
    } else if (arg == "--n-ubatch") {
      n_ubatch = std::atoi(next().c_str());
    } else {
      Fail("unknown argument " + arg);
    }
  }
  if (model_path.empty() || prompt_path.empty() || out_path.empty()) {
    Fail("usage: logit_dump --model M --prompt-file P --out F [--tokens-out T] "
         "[--continuation N] [--ctx N] [--batch N] [--n-ubatch N]");
  }

  ggml_backend_load_all();
  llama_backend_init();

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 999;
  llama_model* model = llama_model_load_from_file(model_path.c_str(),
                                                 model_params);
  if (model == nullptr) {
    Fail("model load failed: " + model_path);
  }
  const llama_vocab* vocab = llama_model_get_vocab(model);
  const int n_vocab = llama_vocab_n_tokens(vocab);

  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = static_cast<uint32_t>(n_ctx);
  ctx_params.n_batch = static_cast<uint32_t>(n_batch);
  if (n_ubatch > 0) {
    ctx_params.n_ubatch = static_cast<uint32_t>(n_ubatch);
  }
  llama_context* ctx = llama_init_from_model(model, ctx_params);
  if (ctx == nullptr) {
    Fail("context creation failed");
  }
  // The requested values can be clamped to the context and to each other, so
  // the harness records what llama.cpp actually runs with.
  const int n_batch_effective = static_cast<int>(llama_n_batch(ctx));
  const int n_ubatch_effective = static_cast<int>(llama_n_ubatch(ctx));
  std::printf("n_batch %d n_ubatch %d\n", n_batch_effective,
              n_ubatch_effective);

  const std::string prompt = ReadFile(prompt_path.c_str());
  std::vector<llama_token> tokens(prompt.size() + 8);
  int count = llama_tokenize(vocab, prompt.data(),
                             static_cast<int32_t>(prompt.size()),
                             tokens.data(),
                             static_cast<int32_t>(tokens.size()),
                             /*add_special=*/true, /*parse_special=*/true);
  if (count < 0) {
    tokens.resize(-count);
    count = llama_tokenize(vocab, prompt.data(),
                           static_cast<int32_t>(prompt.size()),
                           tokens.data(),
                           static_cast<int32_t>(tokens.size()),
                           /*add_special=*/true, /*parse_special=*/true);
  }
  if (count <= 0) {
    Fail("tokenization failed");
  }
  tokens.resize(count);
  std::printf("prompt_tokens %d vocab %d\n", count, n_vocab);

  // `--continuation N` generates N greedy tokens and dumps one logit row per
  // generated token on top of the prompt rows.
  const int positions = count + (continuation > 0 ? continuation : 0);
  std::vector<float> logits(static_cast<std::size_t>(positions) * n_vocab);

  // Prompt pass: every position needs its logits.
  {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (int i = 0; i < count; ++i) {
      batch.token[i] = tokens[i];
      batch.pos[i] = i;
      batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = 0;
      batch.logits[i] = 1;
    }
    batch.n_tokens = count;
    if (llama_decode(ctx, batch) != 0) {
      Fail("prompt decode failed");
    }
    for (int i = 0; i < count; ++i) {
      const float* row = llama_get_logits_ith(ctx, i);
      std::memcpy(logits.data() + std::size_t(i) * n_vocab, row,
                  std::size_t(n_vocab) * sizeof(float));
    }
    llama_batch_free(batch);
  }

  std::vector<llama_token> generated;
  for (int step = 0; step < (continuation > 0 ? continuation : 1); ++step) {
    const float* row = logits.data() +
                       std::size_t(count + step - 1) * n_vocab;
    const auto best = std::max_element(row, row + n_vocab);
    const llama_token token = static_cast<llama_token>(best - row);
    generated.push_back(token);
    tokens.push_back(token);
    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.token[0] = token;
    batch.pos[0] = static_cast<llama_pos>(count + step);
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;
    batch.n_tokens = 1;
    if (llama_decode(ctx, batch) != 0) {
      Fail("continuation decode failed");
    }
    const float* next = llama_get_logits_ith(ctx, 0);
    std::memcpy(logits.data() + std::size_t(count + step) * n_vocab, next,
                std::size_t(n_vocab) * sizeof(float));
    llama_batch_free(batch);
  }
  std::printf("scored_positions %d continuation_tokens %zu\n", positions,
              generated.size());

  {
    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
      Fail("cannot write " + out_path);
    }
    out.write(reinterpret_cast<const char*>(&positions), sizeof(positions));
    out.write(reinterpret_cast<const char*>(&n_vocab), sizeof(n_vocab));
    out.write(reinterpret_cast<const char*>(logits.data()),
              static_cast<std::streamsize>(logits.size() * sizeof(float)));
    out.flush();
    if (!out) {
      Fail("logit write failed");
    }
  }
  if (!tokens_path.empty()) {
    std::ofstream out(tokens_path);
    for (const auto token : tokens) {
      out << token << '\n';
    }
  }
  if (!pieces_path.empty()) {
    // One JSON string array of the token pieces in order, so the harness can
    // rebuild any token-count prefix as text for Gufo's probe.
    std::ofstream out(pieces_path);
    out << "[";
    std::vector<char> piece(256);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      const int size = llama_token_to_piece(
          vocab, tokens[i], piece.data(), static_cast<int32_t>(piece.size()), 0,
          /*special=*/false);
      std::string text(piece.data(), size > 0 ? static_cast<std::size_t>(size) : 0);
      if (i != 0) {
        out << ",";
      }
      out << "\"";
      for (const char c : text) {
        switch (c) {
          case '"': out << "\\\""; break;
          case '\\': out << "\\\\"; break;
          case '\n': out << "\\n"; break;
          case '\r': out << "\\r"; break;
          case '\t': out << "\\t"; break;
          default:
            if (static_cast<unsigned char>(c) < 0x20) {
              char buffer[8];
              std::snprintf(buffer, sizeof(buffer), "\\u%04x",
                            static_cast<unsigned char>(c));
              out << buffer;
            } else {
              out << c;
            }
        }
      }
      out << "\"";
    }
    out << "]";
  }
  if (!continuation_path.empty()) {
    // Decode the greedy continuation back to text so the same sequence can be
    // scored by Gufo's text-prompt probe.
    std::string text;
    std::vector<char> piece(128);
    for (const auto token : generated) {
      const int size = llama_token_to_piece(vocab, token, piece.data(),
                                            static_cast<int32_t>(piece.size()),
                                            0, /*special=*/false);
      if (size > 0) {
        text.append(piece.data(), static_cast<std::size_t>(size));
      }
    }
    std::ofstream out(continuation_path, std::ios::binary);
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
  }
  for (const auto token : generated) {
    std::printf("generated %d\n", token);
  }

  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}

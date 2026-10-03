// Phase-0 KV quantization quality harness (openspec
// add-kvarn-kv-cache-quantization 2.6 / 2.7, design.md D13 / D14 / D16).
//
// Teacher-forced comparison: each corpus is tokenized once, then the FP16-KV
// teacher and every candidate `--kv-cache` value (and, for KVarN, every
// `--kv-cache-tail`) see the same token history and are compared on the full
// next-token distribution at each of the `--window` positions that follow each
// `--depths` prefix. Only the KV format differs between runs; model, build,
// chunking and kernels are identical.
//
// The run is a list of jobs, one per (format, tail, corpus, depth) unit, and
// is meant to be started detached and to survive interruption (D16). Every
// finished unit is written atomically under `<output>/<label>/`, an existing
// unit is never recomputed, and teacher logits are cached outside Git next to
// the units, keyed by model, corpus tokens, depth, window and chunk.
//
//   <output>/<label>/manifest.json           identity, config, memory table
//   <output>/<label>/teacher/c<C>-d<D>.logits  cached teacher logits
//   <output>/<label>/units/<format>__<tail>__c<C>-d<D>.json
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/kvarn_reference.hpp"

namespace {

namespace fs = std::filesystem;
using gufo::hip::QwenExecutionPolicy;
using gufo::hip::QwenGpuExecutor;
using Token = gufo::tokenization::TokenId;
using Clock = std::chrono::steady_clock;

constexpr int kSchemaVersion = 2;
/// KV capacity the memory table is computed for (design.md D1).
constexpr std::size_t kMemoryCapacityTokens = 262144;
constexpr double kSecondsPerHour = 3600.0;
/// Time model used for the estimate before the first job finishes, fitted to
/// the first phase-0 run on gfx1151 (27B Q8_0): a decode step per scored
/// position plus prefill per prefix token. The observed ratio replaces it as
/// jobs complete.
constexpr double kAssumedSecondsPerWindowPosition = 0.125;
constexpr double kAssumedSecondsPerPrefixToken = 0.00385;

struct Options {
  std::string model;
  std::vector<std::string> corpora;
  std::vector<std::uint32_t> depths{4096, 32768, 131072};
  std::vector<std::string> candidates;
  /// `--kv-cache-tail` values applied to every KVarN candidate; 0 = not given.
  std::vector<std::uint32_t> tails{0};
  std::uint32_t window{256};
  std::uint32_t chunk{256};
  std::string output;
  std::string label{"kvarn-quality"};
  std::string model_sha256;
};

/// One evaluated KV format: a `--kv-cache` value with its `--kv-cache-tail`.
struct Config {
  std::string kv_cache;
  std::uint32_t tail{0};
  QwenExecutionPolicy policy;

  [[nodiscard]] std::string TailLabel() const {
    return tail == 0 ? "none" : std::to_string(tail);
  }
  [[nodiscard]] std::string Id() const { return kv_cache + "__" + TailLabel(); }
};

std::vector<std::string> Split(const std::string& text) {
  std::vector<std::string> out;
  std::stringstream stream(text);
  for (std::string item; std::getline(stream, item, ',');) {
    if (!item.empty())
      out.push_back(item);
  }
  return out;
}

std::uint32_t ParseUint(const std::string& text) {
  std::size_t used = 0;
  const unsigned long value = std::stoul(text, &used);
  if (used != text.size())
    throw std::invalid_argument("not an integer: " + text);
  return static_cast<std::uint32_t>(value);
}

Options ParseArgs(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto value = [&]() -> std::string {
      if (i + 1 >= argc)
        throw std::invalid_argument(arg + " needs a value");
      return argv[++i];
    };
    if (arg == "--model")
      o.model = value();
    else if (arg == "--corpus")
      o.corpora = Split(value());
    else if (arg == "--depths") {
      o.depths.clear();
      for (const auto& d : Split(value()))
        o.depths.push_back(ParseUint(d));
    } else if (arg == "--candidates")
      o.candidates = Split(value());
    else if (arg == "--tails") {
      o.tails.clear();
      for (const auto& t : Split(value()))
        o.tails.push_back(t == "none" ? 0 : ParseUint(t));
    } else if (arg == "--window")
      o.window = ParseUint(value());
    else if (arg == "--chunk")
      o.chunk = ParseUint(value());
    else if (arg == "--output")
      o.output = value();
    else if (arg == "--label")
      o.label = value();
    else if (arg == "--model-sha256")
      o.model_sha256 = value();
    else
      throw std::invalid_argument("unknown option " + arg);
  }
  if (o.model.empty() || o.corpora.empty() || o.output.empty() ||
      o.depths.empty() || o.tails.empty() || o.window == 0 || o.chunk == 0) {
    throw std::invalid_argument(
        "usage: qwen_kvarn_quality_eval --model M --corpus a.txt[,b.txt] "
        "--output DIR [--label NAME] [--depths 4096,32768,131072] "
        "[--window 256] [--chunk 256] [--candidates v1,v2] "
        "[--tails none,256,1024,2048] [--model-sha256 HEX]\n"
        "KVarN candidates run once per tail; baselines have no tail.");
  }
  return o;
}

/// Candidate list x tails, with `--kv-cache-tail` applied only where it is
/// accepted (KVarN). A baseline appears once, without a tail.
std::vector<Config> BuildConfigs(const Options& opt) {
  std::vector<Config> configs;
  for (const std::string& name : opt.candidates) {
    const bool takes_tail = gufo::hip::QwenKvCacheAcceptsTail(name);
    const std::vector<std::uint32_t> tails =
        takes_tail ? opt.tails : std::vector<std::uint32_t>{0};
    for (const std::uint32_t tail : tails) {
      Config config;
      config.kv_cache = name;
      config.tail = tail;
      config.policy = QwenExecutionPolicy::Production();
      if (!gufo::hip::ResolveQwenKvCache(name, tail, &config.policy)) {
        throw std::invalid_argument("unsupported --kv-cache " + name +
                                    " with tail " + config.TailLabel());
      }
      configs.push_back(std::move(config));
    }
  }
  return configs;
}

std::string ReadFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("cannot read " + path);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

/// Writes through a temporary file and renames, so an interrupted run leaves
/// only complete units behind.
void WriteAtomically(const fs::path& path, const std::string& content) {
  const fs::path temporary = path.string() + ".tmp";
  {
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out)
      throw std::runtime_error("cannot write " + temporary.string());
    out << content;
    out.flush();
    if (!out)
      throw std::runtime_error("write failed: " + temporary.string());
  }
  fs::rename(temporary, path);
}

template<typename T>
std::string HashSpan(std::span<const T> values) {
  return gufo::crypto::Sha256Hex(
      {reinterpret_cast<const std::uint8_t*>(values.data()),
       values.size_bytes()});
}

/// Full-vocabulary log-softmax in double.
std::vector<double> LogSoftmax(std::span<const float> logits) {
  double maximum = -INFINITY;
  for (const float v : logits) {
    if (std::isfinite(v))
      maximum = std::max(maximum, static_cast<double>(v));
  }
  double sum = 0.0;
  for (const float v : logits)
    sum += std::exp(static_cast<double>(v) - maximum);
  const double log_z = maximum + std::log(sum);
  std::vector<double> out(logits.size());
  for (std::size_t i = 0; i < logits.size(); ++i) {
    out[i] = static_cast<double>(logits[i]) - log_z;
  }
  return out;
}

struct Window {
  std::vector<float> logits;  // window x vocab, row-major
  std::size_t vocab{0};

  [[nodiscard]] std::span<const float> Row(std::size_t i) const {
    return {logits.data() + i * vocab, vocab};
  }
};

Window RunWindow(QwenGpuExecutor& executor, std::span<const Token> tokens,
                 std::uint32_t depth, std::uint32_t window,
                 std::uint32_t chunk) {
  executor.Reset();
  for (std::uint32_t offset = 0; offset < depth; offset += chunk) {
    const std::uint32_t count = std::min(chunk, depth - offset);
    (void)executor.ForwardPromptBatch(tokens.subspan(offset, count), offset,
                                      false);
  }
  Window out;
  out.vocab = executor.GetConfig().vocab_size;
  out.logits.reserve(static_cast<std::size_t>(window) * out.vocab);
  for (std::uint32_t i = 0; i < window; ++i) {
    const std::uint32_t position = depth + i;
    (void)executor.ForwardToken(tokens[position], position, true);
    const auto view = executor.CopyLastLogits();
    out.logits.insert(out.logits.end(), view.begin(), view.end());
  }
  return out;
}

double Percentile(std::vector<double> values, double q) {
  if (values.empty())
    return 0.0;
  std::sort(values.begin(), values.end());
  const double rank = q * static_cast<double>(values.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(std::floor(rank));
  const std::size_t hi = static_cast<std::size_t>(std::ceil(rank));
  return values[lo] +
         (values[hi] - values[lo]) * (rank - static_cast<double>(lo));
}

void Json(std::ostream& out, const std::vector<double>& v) {
  out << '[';
  for (std::size_t i = 0; i < v.size(); ++i) {
    out << (i ? "," : "");
    if (std::isfinite(v[i]))
      out << v[i];
    else
      out << "null";
  }
  out << ']';
}

// ---- Teacher logits cache -------------------------------------------------

/// Fixed header of a cached teacher file, followed by window x vocab floats.
struct TeacherHeader {
  char magic[8];
  std::uint32_t version;
  std::uint32_t depth;
  std::uint32_t window;
  std::uint32_t chunk;
  std::uint64_t vocab;
  char model_sha256[64];
  char tokens_sha256[64];
  double seconds;
};
static_assert(std::is_trivially_copyable_v<TeacherHeader>);
constexpr char kTeacherMagic[8] = {'G', 'U', 'F', 'O', 'K', 'V', 'T', 'L'};
constexpr std::uint32_t kTeacherVersion = 1;

void CopyHex(char (&target)[64], const std::string& hex) {
  if (hex.size() != sizeof(target))
    throw std::runtime_error("expected a 64-character SHA-256: " + hex);
  std::memcpy(target, hex.data(), sizeof(target));
}

bool SameHex(const char (&field)[64], const std::string& hex) {
  return hex.size() == sizeof(field) &&
         std::memcmp(field, hex.data(), sizeof(field)) == 0;
}

// ---- Job plan ---------------------------------------------------------------

struct Paths {
  fs::path root;

  [[nodiscard]] fs::path Teacher(std::size_t corpus,
                                 std::uint32_t depth) const {
    return root / "teacher" /
           ("c" + std::to_string(corpus) + "-d" + std::to_string(depth) +
            ".logits");
  }
  [[nodiscard]] fs::path Unit(const Config& config, std::size_t corpus,
                              std::uint32_t depth) const {
    return root / "units" /
           (config.Id() + "__c" + std::to_string(corpus) + "-d" +
            std::to_string(depth) + ".json");
  }
};

[[nodiscard]] double EstimateSeconds(std::uint32_t depth,
                                     std::uint32_t window) {
  return kAssumedSecondsPerWindowPosition * window +
         kAssumedSecondsPerPrefixToken * depth;
}

/// Job progress in D16's log format: total, done, elapsed and remaining time.
class Progress {
public:
  Progress(std::size_t total, std::size_t done, double estimated_seconds)
      : total_(total), done_(done), remaining_estimate_(estimated_seconds) {
    std::cerr << "plan: " << total_ << " jobs, " << done_ << " already done, "
              << total_ - done_ << " pending, estimated "
              << remaining_estimate_ / kSecondsPerHour << " h remaining\n"
              << std::flush;
  }

  /// Registers one finished job that was estimated at `estimate` seconds and
  /// took `seconds`, and reports what is left.
  void Finished(const std::string& what, double seconds, double estimate) {
    ++done_;
    ++finished_now_;
    spent_ += seconds;
    estimated_spent_ += estimate;
    remaining_estimate_ = std::max(0.0, remaining_estimate_ - estimate);
    const double ratio =
        estimated_spent_ > 0.0 ? spent_ / estimated_spent_ : 1.0;
    std::cerr
        << "job " << done_ << "/" << total_ << " done: " << what << " in "
        << seconds << " s | elapsed "
        << std::chrono::duration<double>(Clock::now() - started_).count() /
               kSecondsPerHour
        << " h | remaining " << total_ - done_ << " jobs ~"
        << remaining_estimate_ * ratio / kSecondsPerHour << " h\n"
        << std::flush;
  }

  void Done() const {
    std::cerr << "finished: " << done_ << "/" << total_ << " jobs\n"
              << std::flush;
  }

private:
  std::size_t total_;
  std::size_t done_;
  std::size_t finished_now_{0};
  double remaining_estimate_;
  double spent_{0.0};
  double estimated_spent_{0.0};
  Clock::time_point started_{Clock::now()};
};

struct Corpora {
  std::vector<std::vector<Token>> tokens;
  std::vector<std::string> sha256;
};

Corpora Tokenize(QwenGpuExecutor& executor, const Options& opt) {
  Corpora out;
  const std::uint32_t max_depth =
      *std::max_element(opt.depths.begin(), opt.depths.end());
  for (const auto& path : opt.corpora) {
    out.tokens.push_back(executor.GetTokenizer().Encode(ReadFile(path)));
    out.sha256.push_back(
        HashSpan<Token>(std::span<const Token>(out.tokens.back())));
    if (out.tokens.back().size() < max_depth + opt.window + 1) {
      throw std::runtime_error(
          path + " has only " + std::to_string(out.tokens.back().size()) +
          " tokens; " + std::to_string(max_depth + opt.window + 1) +
          " are needed");
    }
  }
  return out;
}

/// Writes the manifest, or verifies it against the earlier one: an existing
/// label must keep the same model, corpora, window and chunk, or its units
/// would mix incomparable runs.
void WriteManifest(const Options& opt, const Paths& paths,
                   const std::string& model_hash, const Corpora& corpora,
                   const std::vector<Config>& configs,
                   const QwenGpuExecutor& executor) {
  std::ostringstream identity;
  identity << "\"model_sha256\": \"" << model_hash
           << "\", \"window\": " << opt.window << ", \"chunk\": " << opt.chunk
           << ", \"corpora\": [";
  for (std::size_t c = 0; c < corpora.sha256.size(); ++c) {
    identity << (c ? "," : "") << "\"" << corpora.sha256[c] << "\"";
  }
  identity << "]";
  const fs::path path = paths.root / "manifest.json";
  if (fs::exists(path)) {
    const std::string previous = ReadFile(path.string());
    if (previous.find(identity.str()) == std::string::npos) {
      throw std::runtime_error(
          path.string() +
          " belongs to a different model, corpus set, window or chunk; use a "
          "new --label");
    }
  }
  const auto& model_config = executor.GetConfig();
  const std::size_t layers = model_config.FullAttentionLayerCount();
  const std::size_t heads = model_config.num_key_value_heads;
  std::ostringstream out;
  out << std::setprecision(10) << "{\n  \"schema_version\": " << kSchemaVersion
      << ",\n  \"label\": \"" << opt.label << "\",\n  \"model\": {\"path\": \""
      << opt.model << "\", \"sha256\": \"" << model_hash
      << "\", \"vocab\": " << model_config.vocab_size << "},\n  \"config\": {"
      << identity.str() << ", \"depths\": [";
  for (std::size_t i = 0; i < opt.depths.size(); ++i)
    out << (i ? "," : "") << opt.depths[i];
  out << "]},\n  \"corpus_files\": [";
  for (std::size_t c = 0; c < opt.corpora.size(); ++c) {
    out << (c ? "," : "") << "{\"path\":\"" << opt.corpora[c]
        << "\",\"tokens\":" << corpora.tokens[c].size() << "}";
  }
  // D5 size formula: total bpv, and GiB per session at 262,144 tokens.
  out << "],\n  \"memory\": {\"capacity_tokens\": " << kMemoryCapacityTokens
      << ", \"attention_layers\": " << layers << ", \"kv_heads\": " << heads
      << ", \"formats\": [";
  bool first = true;
  for (const Config& config : configs) {
    const auto& preset = config.policy.kvarn;
    const std::uint32_t tail_rows = gufo::hip::QwenKvarnTailRows(preset);
    const gufo::models::qwen::KvarnMemoryEstimate estimate =
        gufo::models::qwen::EstimateKvarnMemory(
            preset.k_bits, preset.v_bits, tail_rows, gufo::hip::kQwenKvarnW,
            kMemoryCapacityTokens, layers, heads);
    out << (first ? "" : ",") << "\n    {\"kv_cache\":\"" << config.kv_cache
        << "\",\"tail\":"
        << (config.tail == 0 ? "null" : std::to_string(config.tail))
        << ",\"tail_length\":" << tail_rows
        << ",\"body_bpv\":" << estimate.body_bpv
        << ",\"total_bpv\":" << estimate.total_bpv
        << ",\"session_kv_gib\":" << estimate.session_gib << "}";
    first = false;
  }
  out << "\n  ]}\n}\n";
  WriteAtomically(path, out.str());
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opt = ParseArgs(argc, argv);
    const std::vector<Config> configs = BuildConfigs(opt);
    Paths paths{fs::path(opt.output) / opt.label};
    fs::create_directories(paths.root / "teacher");
    fs::create_directories(paths.root / "units");

    std::string error;
    auto reader = gufo::core::GgufReader::OpenFile(opt.model, &error);
    if (!reader)
      throw std::runtime_error("open model: " + error);
    const std::shared_ptr<const gufo::core::GgufReader> shared_reader(
        std::move(reader));
    const std::uint32_t max_depth =
        *std::max_element(opt.depths.begin(), opt.depths.end());
    const std::uint32_t context = max_depth + opt.window + 8;
    const std::string model_hash = opt.model_sha256.empty()
                                       ? gufo::crypto::Sha256FileHex(opt.model)
                                       : opt.model_sha256;

    // The job list: one teacher job per (corpus, depth), then one candidate
    // job per (format, tail, corpus, depth).
    struct Job {
      const Config* config;  // null for the teacher
      std::size_t corpus;
      std::uint32_t depth;
    };
    std::vector<Job> jobs;
    for (std::size_t c = 0; c < opt.corpora.size(); ++c)
      for (const std::uint32_t depth : opt.depths)
        jobs.push_back({nullptr, c, depth});
    for (const Config& config : configs)
      for (std::size_t c = 0; c < opt.corpora.size(); ++c)
        for (const std::uint32_t depth : opt.depths)
          jobs.push_back({&config, c, depth});
    const auto is_done = [&](const Job& job) {
      return fs::exists(job.config == nullptr
                            ? paths.Teacher(job.corpus, job.depth)
                            : paths.Unit(*job.config, job.corpus, job.depth));
    };
    std::size_t done = 0;
    double pending_estimate = 0.0;
    for (const Job& job : jobs) {
      if (is_done(job)) {
        ++done;
      } else {
        pending_estimate += EstimateSeconds(job.depth, opt.window);
      }
    }
    Progress progress(jobs.size(), done, pending_estimate);

    Corpora corpora;
    bool manifest_written = false;
    const auto prepare = [&](QwenGpuExecutor& executor) {
      if (corpora.tokens.empty()) {
        corpora = Tokenize(executor, opt);
      }
      if (!manifest_written) {
        WriteManifest(opt, paths, model_hash, corpora, configs, executor);
        manifest_written = true;
      }
    };

    // Runs every pending job of one format with a single executor, which is
    // created only if something is pending.
    const auto run_format = [&](const Config* config) {
      std::vector<const Job*> pending;
      for (const Job& job : jobs) {
        if (job.config == config && !is_done(job))
          pending.push_back(&job);
      }
      if (pending.empty())
        return;
      const QwenExecutionPolicy policy = config == nullptr
                                             ? QwenExecutionPolicy::Production()
                                             : config->policy;
      auto executor = QwenGpuExecutor::CreateFromGguf(shared_reader, &error,
                                                      context, policy);
      if (!executor)
        throw std::runtime_error("create executor: " + error);
      prepare(*executor);
      const std::string what_format =
          config == nullptr ? "teacher"
                            : config->kv_cache + " tail " + config->TailLabel();
      for (const Job* job : pending) {
        const std::string what = what_format + " corpus " +
                                 std::to_string(job->corpus) + " depth " +
                                 std::to_string(job->depth);
        const auto& tokens = corpora.tokens[job->corpus];
        const auto started = Clock::now();
        const Window window =
            RunWindow(*executor, tokens, job->depth, opt.window, opt.chunk);
        const double seconds =
            std::chrono::duration<double>(Clock::now() - started).count();

        if (config == nullptr) {
          TeacherHeader header{};
          std::memcpy(header.magic, kTeacherMagic, sizeof(header.magic));
          header.version = kTeacherVersion;
          header.depth = job->depth;
          header.window = opt.window;
          header.chunk = opt.chunk;
          header.vocab = window.vocab;
          CopyHex(header.model_sha256, model_hash);
          CopyHex(header.tokens_sha256, corpora.sha256[job->corpus]);
          header.seconds = seconds;
          std::string bytes(reinterpret_cast<const char*>(&header),
                            sizeof(header));
          bytes.append(reinterpret_cast<const char*>(window.logits.data()),
                       window.logits.size() * sizeof(float));
          WriteAtomically(paths.Teacher(job->corpus, job->depth), bytes);
          progress.Finished(what, seconds,
                            EstimateSeconds(job->depth, opt.window));
          continue;
        }

        // Candidate: load the cached teacher and score every position.
        const fs::path teacher_path = paths.Teacher(job->corpus, job->depth);
        std::ifstream teacher_file(teacher_path, std::ios::binary);
        TeacherHeader header{};
        if (!teacher_file.read(reinterpret_cast<char*>(&header),
                               sizeof(header)) ||
            std::memcmp(header.magic, kTeacherMagic, sizeof(header.magic)) !=
                0 ||
            header.version != kTeacherVersion || header.depth != job->depth ||
            header.window != opt.window || header.chunk != opt.chunk ||
            header.vocab != window.vocab ||
            !SameHex(header.model_sha256, model_hash) ||
            !SameHex(header.tokens_sha256, corpora.sha256[job->corpus])) {
          throw std::runtime_error(
              teacher_path.string() +
              " does not match this model, corpus, window or chunk; delete it "
              "or use a new --label");
        }
        Window teacher;
        teacher.vocab = window.vocab;
        teacher.logits.resize(window.logits.size());
        if (!teacher_file.read(reinterpret_cast<char*>(teacher.logits.data()),
                               static_cast<std::streamsize>(
                                   teacher.logits.size() * sizeof(float)))) {
          throw std::runtime_error(teacher_path.string() + " is truncated");
        }
        const std::string teacher_sha =
            HashSpan<float>(std::span<const float>(teacher.logits));

        std::vector<double> kl, nll_teacher, nll_candidate;
        std::size_t top1_match = 0, non_finite = 0;
        const auto argmax = [](std::span<const float> x) {
          return static_cast<std::size_t>(
              std::distance(x.begin(), std::max_element(x.begin(), x.end())));
        };
        for (std::uint32_t i = 0; i < opt.window; ++i) {
          const auto cand = window.Row(i);
          const auto ref = teacher.Row(i);
          const bool finite =
              std::all_of(cand.begin(), cand.end(),
                          [](float v) { return std::isfinite(v); });
          const Token target = tokens[job->depth + i + 1];
          const auto lt = LogSoftmax(ref);
          nll_teacher.push_back(-lt[target]);
          if (!finite) {
            ++non_finite;
            kl.push_back(INFINITY);
            nll_candidate.push_back(INFINITY);
            continue;
          }
          const auto lc = LogSoftmax(cand);
          double divergence = 0.0;
          for (std::size_t v = 0; v < lt.size(); ++v) {
            const double p = std::exp(lt[v]);
            if (p > 0.0)
              divergence += p * (lt[v] - lc[v]);
          }
          kl.push_back(divergence);
          nll_candidate.push_back(-lc[target]);
          if (argmax(cand) == argmax(ref))
            ++top1_match;
        }
        double mean_kl = 0.0, nll_diff = 0.0;
        for (std::size_t i = 0; i < kl.size(); ++i) {
          mean_kl += kl[i];
          nll_diff += nll_candidate[i] - nll_teacher[i];
        }
        const double n = static_cast<double>(kl.size());
        mean_kl /= n;
        nll_diff /= n;

        std::ostringstream unit;
        unit << std::setprecision(10)
             << "{\"schema_version\":" << kSchemaVersion << ",\"kv_cache\":\""
             << config->kv_cache << "\",\"tail\":"
             << (config->tail == 0 ? "null" : std::to_string(config->tail))
             << ",\"tail_length\":"
             << gufo::hip::QwenKvarnTailRows(config->policy.kvarn)
             << ",\"corpus\":" << job->corpus << ",\"depth\":" << job->depth
             << ",\"positions\":" << kl.size()
             << ",\"prefix_tokens\":" << job->depth + opt.window
             << ",\"kl_mean\":" << mean_kl
             << ",\"kl_median\":" << Percentile(kl, 0.5)
             << ",\"kl_p999\":" << Percentile(kl, 0.999)
             << ",\"top1_agreement\":" << static_cast<double>(top1_match) / n
             << ",\"nll_diff_mean\":" << nll_diff
             << ",\"non_finite_positions\":" << non_finite
             << ",\"seconds\":" << seconds << ",\"teacher_logits_sha256\":\""
             << teacher_sha << ",\"kl\":";
        Json(unit, kl);
        unit << ",\"nll_teacher\":";
        Json(unit, nll_teacher);
        unit << ",\"nll_candidate\":";
        Json(unit, nll_candidate);
        unit << "}\n";
        WriteAtomically(paths.Unit(*config, job->corpus, job->depth),
                        unit.str());
        std::ostringstream summary;
        summary << what << " mean KL " << mean_kl << " median "
                << Percentile(kl, 0.5) << " top1 "
                << static_cast<double>(top1_match) / n << " non-finite "
                << non_finite;
        progress.Finished(summary.str(), seconds,
                          EstimateSeconds(job->depth, opt.window));
      }
    };

    run_format(nullptr);
    for (const Config& config : configs)
      run_format(&config);
    progress.Done();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
#else
int main() {
  return 77;
}
#endif

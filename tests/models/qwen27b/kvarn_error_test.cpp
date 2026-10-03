// Task 3.8: a KVarN seal non-finite error fails the request at the existing
// logits synchronization point, poisons its session (no reuse, no snapshot)
// and leaves concurrently running sessions unaffected (design D15).
//
// The seal kernel's real non-finite detection is covered by the seal oracle
// test; here the device flag is raised directly to drive the executor-level
// contract without a multi-hour prompt.
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/gguf_reader.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/execution_policy.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/kvarn.hpp"

namespace {

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

gufo::hip::QwenExecutionPolicy KvarnPolicy() {
  gufo::hip::QwenExecutionPolicy policy =
      gufo::hip::QwenExecutionPolicy::Production();
  if (!gufo::hip::ResolveQwenKvCache("kvarn-k4v4", /*tail_tokens=*/0,
                                     &policy)) {
    throw std::runtime_error("kvarn-k4v4 must resolve");
  }
  return policy;
}

void Run(const char* model_path) {
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  Expect(reader != nullptr, error);
  const std::shared_ptr<const gufo::core::GgufReader> shared(std::move(reader));
  auto model = gufo::hip::QwenGpuModel::CreateFromGguf(shared, &error);
  Expect(model != nullptr, error);

  auto poisoned = gufo::hip::QwenGpuExecutor::Create(
      model, &error, /*max_context=*/256, KvarnPolicy());
  Expect(poisoned != nullptr, error);
  auto healthy = gufo::hip::QwenGpuExecutor::Create(
      model, &error, /*max_context=*/256, KvarnPolicy());
  Expect(healthy != nullptr, error);

  // One KVarN decode step populates the logits and runs without error.
  (void)poisoned->ForwardToken(/*token_id=*/0, /*pos=*/0);
  (void)poisoned->CopyLastLogits();

  // Raise the device flag the seal kernel sets on a non-finite tile.
  const std::uint32_t raised = 1;
  HIP_CHECK(hipMemcpy(poisoned->GetKvarnSealState()->error_flag(), &raised,
                      sizeof(raised), hipMemcpyHostToDevice));

  bool request_failed = false;
  try {
    (void)poisoned->CopyLastLogits();
  } catch (const std::exception&) {
    request_failed = true;
  }
  Expect(request_failed,
         "a KVarN seal error must fail the request at the logits sync point");

  bool reuse_failed = false;
  try {
    (void)poisoned->ForwardToken(/*token_id=*/1, /*pos=*/1);
  } catch (const std::exception&) {
    reuse_failed = true;
  }
  Expect(reuse_failed, "a poisoned KVarN session must not continue");

  bool snapshot_failed = false;
  try {
    (void)poisoned->SaveSnapshot(/*valid_context=*/1);
  } catch (const std::exception&) {
    snapshot_failed = true;
  }
  Expect(snapshot_failed,
         "a poisoned KVarN session must not be snapshotted or reused");

  // A concurrently running session keeps producing finite logits.
  (void)healthy->ForwardToken(/*token_id=*/1, /*pos=*/0);
  const auto logits = healthy->CopyLastLogits();
  for (const float value : logits) {
    Expect(std::isfinite(value),
           "another KVarN session must be unaffected by the poison");
  }
  std::cout << "Qwen KVarN non-finite error handling test passed.\n";
}

}  // namespace

int main(int argc, const char* const* argv) {
  const char* model_path =
      argc > 1 ? argv[1] : std::getenv("GUFO_QWEN27B_MODEL");
  if (model_path == nullptr) {
    std::cout << "SKIP: pass a Qwen GGUF path for the gfx1151 KVarN error "
                 "test\n";
    return 77;
  }
  try {
    Run(model_path);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Qwen KVarN error test failed: " << error.what() << '\n';
    return 1;
  }
}
#else
int main() {
  std::cout << "HIP disabled, skipping Qwen KVarN error test.\n";
  return 77;
}
#endif

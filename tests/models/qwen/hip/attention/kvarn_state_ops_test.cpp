// KVarN compact snapshot checks (openspec add-kvarn-kv-cache-quantization 5.1,
// T-06): the version-3 wire format round-trips the sink, sealed-tile body and
// tail ring bit for bit, its size follows the design D5 formula, mismatched or
// corrupt payloads are rejected, and FP16 keeps the unchanged version-1 format.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/kvarn_reference.hpp"
#include "tests/models/qwen/hip/support/device.hpp"
#include "tests/testing/test_common.hpp"

namespace {

using gufo::hip::QwenExecutionPolicy;
using gufo::hip::QwenGpuArena;
using gufo::hip::QwenKvCacheStorage;

// Declared in QUALITY.md alongside the compact snapshot tests.
constexpr std::uint32_t kKvarnCompactHeaderBytes = 128;
constexpr std::uint32_t kCompactHeaderBytes = 80;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

gufo::core::ModelConfig ProductionQwen27BConfig() {
  gufo::core::ModelConfig config;
  config.model_name = "qwen3.8-27b";
  config.num_layers = 64;
  config.hidden_size = 5120;
  config.intermediate_size = 17408;
  config.num_attention_heads = 24;
  config.num_key_value_heads = 4;
  config.head_dim = 256;
  config.vocab_size = 248320;
  config.context_length = 262144;
  config.full_attention_interval = 4;
  config.mtp_num_layers = 1;
  config.ssm_conv_kernel = 4;
  config.ssm_state_size = 128;
  config.ssm_group_count = 16;
  config.ssm_time_step_rank = 48;
  config.ssm_inner_size = 6144;
  config.rotary_dim = 64;
  config.rope_theta = 10000000.0F;
  return config;
}

QwenExecutionPolicy KvarnPolicy(std::string_view name, std::uint32_t tail) {
  auto policy = QwenExecutionPolicy::Production();
  Expect(gufo::hip::ResolveQwenKvCache(name, tail, &policy),
         "a KVarN preset must resolve");
  return policy;
}

std::vector<std::uint8_t> Pattern(std::size_t bytes, std::uint8_t seed) {
  std::vector<std::uint8_t> data(bytes);
  for (std::size_t i = 0; i < bytes; ++i) {
    data[i] = static_cast<std::uint8_t>(seed + i * 7U);
  }
  return data;
}

struct RegionImage {
  std::vector<std::uint8_t> sink;
  std::vector<std::uint8_t> body;
  std::vector<std::uint8_t> tail;
  std::vector<std::uint32_t> sealed;
};

void UploadRegions(QwenGpuArena& arena, std::uint8_t seed) {
  const auto& regions = arena.GetKvarnRegions();
  const auto sink = Pattern(regions.sink_bytes, seed);
  const auto body =
      Pattern(regions.body_bytes, static_cast<std::uint8_t>(seed + 1));
  const auto tail =
      Pattern(regions.tail_bytes, static_cast<std::uint8_t>(seed + 2));
  HIP_CHECK(hipMemcpy(arena.d_kvarn_sink, sink.data(), sink.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(arena.d_kvarn_body, body.data(), body.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(arena.d_kvarn_tail, tail.data(), tail.size(),
                      hipMemcpyHostToDevice));
  const std::size_t seal_elements = 16U * 4U;
  std::vector<std::uint32_t> sealed(seal_elements);
  for (std::size_t i = 0; i < sealed.size(); ++i) {
    sealed[i] = static_cast<std::uint32_t>(seed + i);
  }
  HIP_CHECK(hipMemcpy(arena.GetKvarnSealState()->sealed(), sealed.data(),
                      sealed.size() * sizeof(std::uint32_t),
                      hipMemcpyHostToDevice));
}

RegionImage DownloadRegions(const QwenGpuArena& arena) {
  const auto& regions = arena.GetKvarnRegions();
  RegionImage image;
  image.sink.resize(regions.sink_bytes);
  image.body.resize(regions.body_bytes);
  image.tail.resize(regions.tail_bytes);
  image.sealed.resize(16U * 4U);
  HIP_CHECK(hipMemcpy(image.sink.data(), arena.d_kvarn_sink, image.sink.size(),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(image.body.data(), arena.d_kvarn_body, image.body.size(),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(image.tail.data(), arena.d_kvarn_tail, image.tail.size(),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(image.sealed.data(), arena.GetKvarnSealState()->sealed(),
                      image.sealed.size() * sizeof(std::uint32_t),
                      hipMemcpyDeviceToHost));
  return image;
}

void TestKvarnCompactRoundTrip(const char* preset, std::uint32_t tail,
                               QwenKvCacheStorage storage) {
  const auto config = ProductionQwen27BConfig();
  constexpr std::uint32_t context = 4096;
  constexpr std::uint32_t valid_context = 300;
  auto policy = KvarnPolicy(preset, tail);
  policy.kv_cache_storage = storage;

  QwenGpuArena source(config, context, policy);
  HIP_CHECK(hipStreamSynchronize(source.stream));
  UploadRegions(source, 11);
  const auto source_image = DownloadRegions(source);

  auto snapshot = source.SaveSnapshot(valid_context);
  Expect(snapshot->KvStorage() == QwenKvCacheStorage::kKvarn,
         "a KVarN snapshot must record the KVarN storage kind");
  Expect(
      snapshot->CompactPayloadBytes() ==
          source.SnapshotPayloadBytes(valid_context) + kKvarnCompactHeaderBytes,
      "the KVarN v3 payload must be the state bytes plus its 128-byte header");
  Expect(snapshot->PayloadBytes() >= source.GetKvarnRegions().total_bytes,
         "the KVarN payload must carry every region");

  std::vector<std::uint8_t> serialized(snapshot->CompactPayloadBytes());
  const std::size_t written = snapshot->SerializeCompact(serialized);
  Expect(written == serialized.size(),
         "the KVarN serializer must fill the reported byte count");
  Expect(serialized[8] == 3 && serialized[9] == 0 && serialized[10] == 0 &&
             serialized[11] == 0,
         "the KVarN payload must use wire version 3");

  QwenGpuArena restored(config, context, policy);
  HIP_CHECK(hipStreamSynchronize(restored.stream));
  UploadRegions(restored, 200);
  restored.RestoreCompactSnapshot(serialized, valid_context);
  const auto restored_image = DownloadRegions(restored);
  Expect(restored_image.sink == source_image.sink &&
             restored_image.body == source_image.body &&
             restored_image.tail == source_image.tail &&
             restored_image.sealed == source_image.sealed,
         "the KVarN snapshot must round-trip every region and seal counter");

  // A different tail length is a different identity: must be rejected, never
  // silently reinterpreted.
  QwenGpuArena other_tail(
      config, context,
      KvarnPolicy(preset, tail == 0 ? std::uint32_t{2048} : std::uint32_t{0}));
  HIP_CHECK(hipStreamSynchronize(other_tail.stream));
  bool rejected_tail = false;
  try {
    other_tail.RestoreCompactSnapshot(serialized, valid_context);
  } catch (const std::exception&) {
    rejected_tail = true;
  }
  Expect(rejected_tail, "a snapshot with another tail length must be rejected");

  // A canonical FP16 arena must not accept a KVarN payload, and a canonical
  // payload must not be read as KVarN.
  QwenGpuArena canonical(config, context, QwenExecutionPolicy::Production());
  HIP_CHECK(hipStreamSynchronize(canonical.stream));
  auto canonical_snapshot = canonical.SaveSnapshot(valid_context);
  std::vector<std::uint8_t> canonical_bytes(
      canonical_snapshot->CompactPayloadBytes());
  Expect(canonical_snapshot->SerializeCompact(canonical_bytes) ==
             canonical_bytes.size(),
         "the FP16 serializer must fill the reported byte count");
  bool rejected_canonical = false;
  try {
    restored.RestoreCompactSnapshot(canonical_bytes, valid_context);
  } catch (const std::exception&) {
    rejected_canonical = true;
  }
  Expect(rejected_canonical,
         "a KVarN arena must reject a canonical snapshot payload");
  bool rejected_kvarn = false;
  try {
    canonical.RestoreCompactSnapshot(serialized, valid_context);
  } catch (const std::exception&) {
    rejected_kvarn = true;
  }
  Expect(rejected_kvarn,
         "a canonical arena must reject a KVarN snapshot payload");

  std::cout << "KVarN compact snapshot preset=" << preset << " tail=" << tail
            << " bytes=" << serialized.size() << " round-trip exact\n";
}

void TestKvarnCompactRejectsCorruption() {
  const auto config = ProductionQwen27BConfig();
  constexpr std::uint32_t context = 4096;
  constexpr std::uint32_t valid_context = 300;
  QwenGpuArena arena(config, context, KvarnPolicy("kvarn-k4v4", 0));
  HIP_CHECK(hipStreamSynchronize(arena.stream));
  UploadRegions(arena, 5);
  auto snapshot = arena.SaveSnapshot(valid_context);
  std::vector<std::uint8_t> serialized(snapshot->CompactPayloadBytes());
  snapshot->SerializeCompact(serialized);

  auto expect_rejected = [&](std::vector<std::uint8_t> payload) {
    QwenGpuArena target(config, context, KvarnPolicy("kvarn-k4v4", 0));
    HIP_CHECK(hipStreamSynchronize(target.stream));
    try {
      target.RestoreCompactSnapshot(payload, valid_context);
    } catch (const std::exception&) {
      return true;
    }
    return false;
  };

  // Bad magic.
  auto bad_magic = serialized;
  bad_magic[0] ^= 0xFFU;
  Expect(expect_rejected(bad_magic), "a corrupt magic must be rejected");
  // Bad version.
  auto bad_version = serialized;
  bad_version[8] = 4;
  Expect(expect_rejected(bad_version), "an unknown version must be rejected");
  // Truncated payload.
  auto truncated = serialized;
  truncated.pop_back();
  Expect(expect_rejected(truncated), "a truncated payload must be rejected");
  // Flipped configuration byte (K bits).
  auto bad_config = serialized;
  bad_config[44] = 3;
  Expect(expect_rejected(bad_config),
         "a corrupt KVarN configuration must be rejected");

  std::cout << "KVarN compact snapshot corruption rejection exact\n";
}

void TestFp16KeepsVersionOne() {
  const auto config = ProductionQwen27BConfig();
  constexpr std::uint32_t context = 1024;
  constexpr std::uint32_t valid_context = 7;
  QwenGpuArena arena(config, context, QwenExecutionPolicy::Production());
  HIP_CHECK(hipStreamSynchronize(arena.stream));
  auto snapshot = arena.SaveSnapshot(valid_context);
  Expect(snapshot->CompactPayloadBytes() ==
             snapshot->PayloadBytes() + kCompactHeaderBytes,
         "FP16 must keep the unchanged 80-byte compact header");
  std::vector<std::uint8_t> serialized(snapshot->CompactPayloadBytes());
  snapshot->SerializeCompact(serialized);
  Expect(serialized[8] == 1 && serialized[9] == 0 && serialized[10] == 0 &&
             serialized[11] == 0,
         "FP16 must keep writing wire version 1");
  std::cout << "FP16 compact snapshot stays on version 1\n";
}

}  // namespace

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status = gufo::test::GateHipDevice(
      gufo::test::HipDeviceRequirement::kOptional, "Qwen KVarN state ops test");
  if (device_status != 0) {
    return device_status;
  }
  TestKvarnCompactRoundTrip("kvarn-k4v4", 0, QwenKvCacheStorage::kKvarn);
  TestKvarnCompactRoundTrip("kvarn-k4v4", 2048, QwenKvCacheStorage::kKvarn);
  TestKvarnCompactRoundTrip("kvarn-k6v5", 0, QwenKvCacheStorage::kKvarn);
  TestKvarnCompactRejectsCorruption();
  TestFp16KeepsVersionOne();
  std::cout << "Qwen KVarN state tests passed.\n";
#endif
  return 0;
}
#endif  // defined(ENGINE_ENABLE_HIP)

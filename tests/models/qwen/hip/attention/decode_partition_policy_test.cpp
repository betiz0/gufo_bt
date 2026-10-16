// Partition invariants of the FP16 decode/verification attention blocks
// (openspec change update-qwen-decode-attention-blocks, tasks 2.1/2.4, design
// D11). Pure policy logic, so it runs on the host without a GPU.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "src/models/qwen/hip/detail/attention_policy.hpp"

namespace {

using gufo::hip::detail::DecodeAttentionBlockRange;
using gufo::hip::detail::kDecodeAttentionBlockTokens;
using gufo::hip::detail::kSplitKDecodeAttentionMaxSplits;
using gufo::hip::detail::SelectDecodeAttentionBlockRange;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::fprintf(stderr, "decode attention partition policy: %s\n",
                 message.c_str());
    std::abort();
  }
}

/// One partition per split, contiguous, 128-aligned and covering exactly
/// [0, sequence_length).
void CheckInvariants(std::size_t sequence_length) {
  std::size_t previous_end = 0;
  std::size_t covered = 0;
  std::uint32_t non_empty = 0;
  for (std::uint32_t split = 0; split < kSplitKDecodeAttentionMaxSplits;
       ++split) {
    const DecodeAttentionBlockRange range =
        SelectDecodeAttentionBlockRange(sequence_length, split);
    const std::string where = "n=" + std::to_string(sequence_length) +
                              " split=" + std::to_string(split);
    Expect(range.begin <= range.end, where + ": begin after end");
    Expect(range.end <= sequence_length, where + ": end past the sequence");
    Expect(range.begin % kDecodeAttentionBlockTokens == 0,
           where + ": begin is not a tile boundary");
    Expect(range.end == sequence_length ||
               range.end % kDecodeAttentionBlockTokens == 0,
           where + ": interior end is not a tile boundary");
    Expect(sequence_length >= 128 ||
               range.begin % kDecodeAttentionBlockTokens == 0 ||
               range.begin == sequence_length,
           where + ": an interior begin must be a tile boundary");
    Expect(range.begin == previous_end, where + ": gap or overlap");
    previous_end = range.end;
    if (!range.empty()) {
      ++non_empty;
      covered += range.end - range.begin;
    }
  }
  Expect(non_empty <= kSplitKDecodeAttentionMaxSplits,
         "more partitions than the scratch layout holds");
  Expect(covered == sequence_length, "n=" + std::to_string(sequence_length) +
                                         ": coverage is " +
                                         std::to_string(covered));
  if (sequence_length != 0) {
    Expect(non_empty >= 1, "no partition holds a key");
  }
}

/// The split count the FP16 path uses stays the shared 32 for every length at
/// or above the split-K threshold.
void CheckSplitCount() {
  using gufo::hip::detail::SelectDecodeAttentionSplitCount;
  Expect(SelectDecodeAttentionSplitCount(127) == 1, "127 must stay online");
  Expect(
      SelectDecodeAttentionSplitCount(128) == kSplitKDecodeAttentionMaxSplits,
      "128 must use the split-K path");
  Expect(SelectDecodeAttentionSplitCount(131072) ==
             kSplitKDecodeAttentionMaxSplits,
         "131072 must use the split-K path");
}

/// The partitions of a row depend on its sequence length alone.
void CheckRowIndependence() {
  for (std::size_t sequence_length : {std::size_t{4096}, std::size_t{131072}}) {
    for (std::uint32_t split = 0; split < kSplitKDecodeAttentionMaxSplits;
         ++split) {
      const auto first =
          SelectDecodeAttentionBlockRange(sequence_length, split);
      const auto second =
          SelectDecodeAttentionBlockRange(sequence_length, split);
      Expect(first.begin == second.begin && first.end == second.end,
             "partitions changed between calls");
    }
  }
}

}  // namespace

int main() {
  CheckSplitCount();
  CheckRowIndependence();
  for (std::size_t sequence_length :
       {std::size_t{128}, std::size_t{129}, std::size_t{256}, std::size_t{4095},
        std::size_t{4096}, std::size_t{4097}, std::size_t{131064},
        std::size_t{131072}, std::size_t{262143}, std::size_t{262144}}) {
    CheckInvariants(sequence_length);
  }
  // Below the split-K threshold the block kernel is not used at all: split 0
  // covers the whole sequence and every other split is empty. The 128-token
  // boundary invariant applies only at or above the threshold.
  for (std::size_t sequence_length : {std::size_t{1}, std::size_t{127}}) {
    const auto first = SelectDecodeAttentionBlockRange(sequence_length, 0);
    Expect(first.begin == 0 && first.end == sequence_length,
           "split 0 must cover the whole sequence at or below the threshold");
    for (std::uint32_t split = 1; split < kSplitKDecodeAttentionMaxSplits;
         ++split) {
      Expect(SelectDecodeAttentionBlockRange(sequence_length, split).empty(),
             "a split above 0 must be empty at or below the threshold");
    }
  }
  std::printf("Qwen decode attention partition policy test passed.\n");
  return 0;
}

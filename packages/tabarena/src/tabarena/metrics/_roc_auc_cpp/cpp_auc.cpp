#include <cstdint>
#include <cstring>
#include <algorithm>
#include <limits>
#include <vector>


namespace {

inline uint32_t mantissa(uint32_t a) { return a & 0x7FFFFF; }

// Two radix passes covering the 23 mantissa bits: 12 + 11.
// The histograms (4096 + 2048 uint32 = 24KB) fit comfortably in L1/L2, and two
// scatter passes move less memory than three (the sort passes are memory-bound,
// so pass count ~ runtime).
constexpr int RADIX0_BITS = 12;
constexpr int RADIX1_BITS = 23 - RADIX0_BITS;
constexpr uint32_t RADIX0_SIZE = 1u << RADIX0_BITS;
constexpr uint32_t RADIX1_SIZE = 1u << RADIX1_BITS;
constexpr uint32_t RADIX0_MASK = RADIX0_SIZE - 1;
constexpr uint32_t RADIX1_MASK = RADIX1_SIZE - 1;

// Scratch buffers reused across calls (grow-only). Ensemble simulation calls
// this metric thousands of times with the same `len`, so per-call allocation —
// and std::vector's zero-fill, two hidden write passes — would dominate.
thread_local std::vector<uint32_t> scratch1;
thread_local std::vector<uint32_t> scratch2;

}  // namespace

/*
 * Fast, single-threaded implementation of the binary non-weighted ROC AUC metric.
 * The primary speed-up comes from a radix sort on the 23 bits of the mantissa of float values.
 * This function assumes y_true and y_pred are both of size `len`.
 * If entries in y_pred are outside the range [0.0, 1.0], they are treated as clamped into that range.
 * Inputs are read-only; callers may pass their own buffers without a defensive copy.
 */
double radix_roc_auc(const bool* y_true, const float* y_pred, size_t len) {

  if (scratch1.size() < len) {
    scratch1.resize(len);
    scratch2.resize(len);
  }
  uint32_t* __restrict storage1 = scratch1.data();
  uint32_t* __restrict storage2 = scratch2.data();

  // 24KB total; small enough for the stack, avoids a heap allocation per call
  uint32_t histogram0[RADIX0_SIZE] = {0};
  uint32_t histogram1[RADIX1_SIZE] = {0};

  // Single fused pass: clamp each prediction into [1.0, 2.0) to exploit the
  // IEEE-754 float layout (mantissa order == value order on that interval),
  // pack the bool label into bit 24 and the 23 mantissa bits into the LSBs,
  // and build both radix histograms.
  // The clamp is computed in double and rounded to float exactly as the
  // previous in-place version did, so scores are bit-identical to it.
  for (size_t i = 0; i < len; ++i) {
    const float clamped = static_cast<float>(
        std::clamp(static_cast<double>(y_pred[i]) + 1.0, 1.0,
                   2.0 - static_cast<double>(std::numeric_limits<float>::epsilon())));
    uint32_t bits;
    std::memcpy(&bits, &clamped, sizeof(bits));
    const uint32_t entry = (static_cast<uint32_t>(y_true[i]) << 24) | mantissa(bits);
    storage1[i] = entry;
    histogram0[entry & RADIX0_MASK]++;
    histogram1[(entry >> RADIX0_BITS) & RADIX1_MASK]++;
  }

  // Exclusive prefix sums turn the histograms into scatter offsets
  uint32_t sum0 = 0;
  for (uint32_t i = 0; i < RADIX0_SIZE; ++i) {
    const uint32_t count = histogram0[i];
    histogram0[i] = sum0;
    sum0 += count;
  }
  uint32_t sum1 = 0;
  for (uint32_t i = 0; i < RADIX1_SIZE; ++i) {
    const uint32_t count = histogram1[i];
    histogram1[i] = sum1;
    sum1 += count;
  }

  // Sort radix0 (least significant 12 bits)
  for (size_t i = 0; i < len; ++i) {
    const uint32_t entry = storage1[i];
    storage2[histogram0[entry & RADIX0_MASK]++] = entry;
  }

  // Sort radix1 (most significant 11 bits); result lands sorted in storage1
  for (size_t i = 0; i < len; ++i) {
    const uint32_t entry = storage2[i];
    storage1[histogram1[(entry >> RADIX0_BITS) & RADIX1_MASK]++] = entry;
  }

  // Perform binary non-weighted roc_auc summation on sorted entries
  uint64_t total_true_cnt = 0;
  uint64_t total_false_cnt = 0;
  uint64_t last_unique_true_cnt = 0;
  uint64_t last_unique_false_cnt = 0;
  uint64_t rect_auc = 0;
  uint64_t tri_auc = 0;

  for (size_t i = 0; i < len; ++i) {
    const uint32_t entry = storage1[len - 1 - i];
    const bool label = entry >> 24 & 0xFF;

    total_true_cnt += label;
    total_false_cnt += !label;
    rect_auc += !label * last_unique_true_cnt;

    // Avoid code branches from if statements by using branchless assignments
    const bool diff = (i == len - 1) || (mantissa(entry) != mantissa(storage1[len - 2 - i]));
    tri_auc += diff * (total_true_cnt - last_unique_true_cnt) * (total_false_cnt - last_unique_false_cnt);
    last_unique_true_cnt = diff * total_true_cnt + !diff * last_unique_true_cnt;
    last_unique_false_cnt = diff * total_false_cnt + !diff * last_unique_false_cnt;
  }

  return (rect_auc + tri_auc / 2.0) / (total_true_cnt * total_false_cnt);
}

extern "C" {
    double cpp_auc_ext(const float* ts, const bool* st, size_t len) {
        return radix_roc_auc(st, ts, len);
    }
}

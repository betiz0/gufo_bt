// SPDX-License-Identifier: MIT
#pragma once
#include <hip/hip_runtime.h>
#include <stddef.h>
#include <stdint.h>
extern "C" {
int qfn_mmq_init(int device);

void qfn_mmq_set_routed_max_expert_rows(int rows);

void qfn_mmq_set_routed_tile_cols(int cols);

int qfn_mmq_routed_tile_cols_for_counts(const unsigned int *counts,
                                        int n_experts);

int qfn_mmq_q8_0_dense(
    const void  * W_q8_0,
    const float * X_f32,
    float       * out_f32,
    int           M,
    int           N,
    int           K,
    hipStream_t  stream);

// Q6_K dense projection: prefill MMQ over F32 activations (`qfn_mmq_*_dense`),
// or decode GEMV over activations already quantized with
// qfn_mmq_quantize_q8_1 (`qfn_mmq_*_dense_vec_preq`). K must be a whole number
// of weight blocks (256 for Q6_K, 32 for IQ4_NL); the decode entry takes at
// most 8 tokens.
int qfn_mmq_q6_K_dense(const void* W_q6_K, const float* X_f32,
                       float* out_f32, int M, int N, int K,
                       hipStream_t stream);

int qfn_mmq_q6_K_dense_vec_preq(const void* W_q6_K, const void* X_q8,
                                float* out_f32, int M, int N, int K,
                                hipStream_t stream);

// IQ4_NL dense projection, same contract as the Q6_K entries.
int qfn_mmq_iq4_nl_dense(const void* W_iq4_nl, const float* X_f32,
                         float* out_f32, int M, int N, int K,
                         hipStream_t stream);

int qfn_mmq_iq4_nl_dense_vec_preq(const void* W_iq4_nl, const void* X_q8,
                                  float* out_f32, int M, int N, int K,
                                  hipStream_t stream);

// Q4_K / Q5_K / Q5_0 dense projections (the Swift-1.5 Q4_K_M mix), same
// contract as the Q6_K entries: K must be a whole number of weight blocks
// (256 for Q4_K/Q5_K, 32 for Q5_0).
int qfn_mmq_q4_K_dense(const void* W_q4_K, const float* X_f32,
                       float* out_f32, int M, int N, int K,
                       hipStream_t stream);

int qfn_mmq_q4_K_dense_vec_preq(const void* W_q4_K, const void* X_q8,
                                float* out_f32, int M, int N, int K,
                                hipStream_t stream);

int qfn_mmq_q5_K_dense(const void* W_q5_K, const float* X_f32,
                       float* out_f32, int M, int N, int K,
                       hipStream_t stream);

int qfn_mmq_q5_K_dense_vec_preq(const void* W_q5_K, const void* X_q8,
                                float* out_f32, int M, int N, int K,
                                hipStream_t stream);

int qfn_mmq_q5_0_dense(const void* W_q5_0, const float* X_f32,
                       float* out_f32, int M, int N, int K,
                       hipStream_t stream);

int qfn_mmq_q5_0_dense_vec_preq(const void* W_q5_0, const void* X_q8,
                                float* out_f32, int M, int N, int K,
                                hipStream_t stream);

int qfn_mmq_q4_K_moe_pair_unique(const void* W_a, const void* W_b,
                                 const float* X_f32, const int32_t* ids,
                                 float* out_a, float* out_b, int M, int K,
                                 int n_tokens, int n_experts, int n_expert_used,
                                 hipStream_t stream);

// Routed vector projection for GGML Q4_K/Q5_K/Q5_1/Q8_0. Optional paired
// weights use the same shape, routing and quantized input; both outputs are
// [n_tokens][n_expert_used][M]. W_b and out_b must be supplied together.
int qfn_mmq_moe_vec(int weight_type, const void* W, const float* X_f32,
                    const int32_t* ids, float* out, int M, int K, int n_tokens,
                    int n_experts, int n_expert_used, hipStream_t stream,
                    const void* W_b = nullptr, float* out_b = nullptr);

// Q4_K/Q5_K/Q8_0 gate/up for 1–8 tokens, with the SwiGLU result in out.
// Q4_K also accepts up to 64 independent rows, grouped by expert.
int qfn_mmq_moe_gated_vec(int weight_type, const void* gate, const void* up,
                          const float* x, const int32_t* ids, float* out, int m,
                          int k, int tokens, int experts, int experts_used,
                          hipStream_t stream);

size_t qfn_mmq_q8_1_bytes(int N, int K);

int qfn_mmq_quantize_q8_1(const float* X_f32, void* X_q8, int N, int K,
                          hipStream_t stream);

// Exact decode arithmetic for 1–8 rows, or up to 32 ungated rows.
// Wide projections (M >= 8192, K = 2560) accept up to 48 ungated rows.
// The 320×10240 HC projection also accepts larger multiples of eight.
// Only N output rows are written.
int qfn_mmq_q8_0_dense_vec_preq(const void* W_q8_0, const void* W_gate,
                                const void* X_q8, float* out_f32, int M, int N,
                                int K, hipStream_t stream);

int qfn_mmq_build_ids_maps(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, hipStream_t stream);

int qfn_mmq_q8_0_moe_raw(
    const void * W, const float * X_f32, const int32_t * ids, float * out,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    hipStream_t stream);

int qfn_mmq_q4_K_moe_raw(
    const void * W, const float * X_f32, const int32_t * ids, float * out,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    hipStream_t stream);

int qfn_mmq_q5_1_moe_raw(
    const void * W, const float * X_f32, const int32_t * ids, float * out,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    hipStream_t stream);

int qfn_mmq_q5_K_moe_raw(
    const void * W, const float * X_f32, const int32_t * ids, float * out,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    hipStream_t stream);

// IQ4_NL routed projection (the down-experts of the AP tier). Same contract
// as qfn_mmq_q4_K_moe_raw; the vector route additionally accepts IQ4_NL.
int qfn_mmq_iq4_nl_moe_raw(
    const void * W, const float * X_f32, const int32_t * ids, float * out,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    hipStream_t stream);

// Q5_0 routed projection (the down-experts of the Swift-1.5 tier). Same
// contract as qfn_mmq_q4_K_moe_raw; the vector route accepts Q5_0 as well.
int qfn_mmq_q5_0_moe_raw(
    const void * W, const float * X_f32, const int32_t * ids, float * out,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    hipStream_t stream);

}

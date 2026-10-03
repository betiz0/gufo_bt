# G1 判定レポート — `add-kvarn-kv-cache-quantization`

- 対象変更: `openspec/changes/add-kvarn-kv-cache-quantization`（schema: spec-driven）
- 判定対象: tasks.md 3.11（**G1** = T-01〜T-03、T-05 の decode 部分、性能契約の AR tg）
- 判定: **FAIL**（d0/d4K は合格、d32K/d128K は全プリセット未達）
- 測定日: 2026-10-02 〜 2026-10-03
- 生データ: `artifacts/g1/`（最適化前）、`artifacts/g2/`（最適化後、いずれも Git 管理外）
- 評価スクリプト: `/home/betiz/g1/run_g1.sh`（D16 の切り離し実行・単位ごとの成果物・再開）、`/home/betiz/g1/evaluate_g1.py`

---

## 1. G1 の合格条件

`design.md` の「検証 ID と成果物の対応」と spec「性能契約」より、AR tg は FP16 KV 比で:

| 深度 | 基準 |
|---|---|
| 0〜4K | ≥ 0.95 |
| 32K 以上 | ≥ 1.00 |
| 128K（メモリ優先 `kvarn-k4v4`） | ≥ 1.15 |

対象: `kvarn-k4v4` / `kvarn-k5v4` / `kvarn-k6v5` の各プリセット、テール長 未指定 と `--kv-cache-tail 2048` の両方。

---

## 2. 測定方法

- モデル: `/home/betiz/models/Qwen3.8-27B-Q8_0.gguf`（26.90 GiB、27.32 B params）
- ハードウェア: gfx1151（AMD Radeon 8060S、VRAM 112640 MiB）
- バイナリ: `build/release/gufo`（`RelWithDebInfo` + `NDEBUG`、`/opt/rocm/bin/hipcc`）
  - `/nix` が 100% 満杯で `nix develop` / `nix build` 不可のため、system toolchain で構成:
    ```sh
    cmake --preset release -DCMAKE_C_COMPILER=/opt/rocm/bin/hipcc \
      -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc
    cmake --build build/release --target gufo --parallel 32
    ```
- ベンチマーク（各点 1 回のウォームアップ済み測定、greedy、FP16 で各深度を挟み込み）:
  ```sh
  ./build/release/gufo bench -m /home/betiz/models/Qwen3.8-27B-Q8_0.gguf \
    --n-prompt 2048 --n-gen 128 --n-depth <0|4096|32768|131072> \
    --kv-cache <fp16|kvarn-k4v4|kvarn-k5v4|kvarn-k6v5> [--kv-cache-tail 2048]
  ```
- 分母 FP16 は各深度の前後 2 回（`fp16`, `fp16b`）の平均。32 ジョブ（8 構成 × 4 深度）、1 深度あたり 2〜17 分。
- 長時間測定は design D16 の方法（切り離しプロセス、`.done` マーカー、単位ごとの成果物と再開）で実行。

---

## 3. 結果

### 3.1 最適化後（`artifacts/g2`、現行ツリー）— 最終判定

tg128 tok/s、比 = FP16 ブラケット平均に対する比率。

| 候補 | d0 | d4,096 | d32,768 | d131,072 | 判定 |
|---|---:|---:|---:|---:|---|
| FP16 参照 (tok/s) | 7.69 | 7.63 | 7.19 | 6.03 | — |
| kvarn-k4v4, tail 未指定 | 7.64 (0.993) | 7.50 (0.983) | 6.73 (**0.936**) | 4.90 (**0.813**) | FAIL |
| kvarn-k4v4, tail 2048 | 7.64 (0.993) | 7.49 (0.982) | 6.70 (**0.932**) | 4.88 (**0.809**) | FAIL |
| kvarn-k5v4, tail 未指定 | 7.64 (0.993) | 7.47 (0.979) | 6.54 (**0.910**) | 4.51 (**0.748**) | FAIL |
| kvarn-k5v4, tail 2048 | 7.64 (0.993) | 7.48 (0.980) | 6.49 (**0.903**) | 4.48 (**0.743**) | FAIL |
| kvarn-k6v5, tail 未指定 | 7.64 (0.993) | 7.42 (0.972) | 6.20 (**0.862**) | 3.88 (**0.643**) | FAIL |
| kvarn-k6v5, tail 2048 | 7.64 (0.993) | 7.45 (0.976) | 6.15 (**0.855**) | 3.86 (**0.640**) | FAIL |

- 最悪: `k6v5` d131072 = **0.640**（基準 1.00、余裕 −0.360）。
- 合格は d0（0.993）と d4K（0.972〜0.983）のみ。

### 3.2 最適化前（`artifacts/g1`）— 参考

| 候補 | d0 | d4,096 | d32,768 | d131,072 |
|---|---:|---:|---:|---:|
| kvarn-k4v4, tail 未指定 | 0.993 | 0.959 | 0.807 | 0.556 |
| kvarn-k4v4, tail 2048 | 0.993 | 0.966 | 0.794 | 0.544 |
| kvarn-k5v4, tail 未指定 | 0.993 | 0.952 | 0.766 | 0.488 |
| kvarn-k5v4, tail 2048 | 0.992 | 0.961 | 0.751 | 0.479 |
| kvarn-k6v5, tail 未指定 | 0.993 | **0.936** | 0.687 | 0.357 |
| kvarn-k6v5, tail 2048 | 0.993 | 0.950 | 0.676 | 0.358 |

（この時点では d4K の `k6v5` 未指定も単独で FAIL。現行ツリーでは 0.972 に改善。）

### 3.3 最適化の効果

retained 変更（`src/models/qwen/hip/kernels/kvarn_read.hip`）:
- split 内のキーを 128 キータイル単位で走査（D2 のビュー規則では split 内のチャンクは一様に封印済みか F16）
- tile payload ポインタと per-lane の RTN スケール/ゼロ点・チャネルスケールを **タイルごとに 1 回だけ** ロード
- 低ビット面を 16bit ワード単位で展開する `KvarnUnpackLaneCodes`（コードはビット一致）
- ファイルスコープの `#pragma clang fp contract(off)`（収縮を明示化。これがないと machine combiner がレジスタ圧で MUL+ADD を融合し、T-05 のビット一致が 15/128 スコアで 1 ulp ずれる）

改善: k4v4 d32K 0.807 → 0.936、d128K 0.556 → 0.813。**基準には届かず**。

---

## 4. 正しさの検証（現行ツリー）

```sh
cmake --build build/gpu-test --target gufo --parallel 32
ctest --test-dir build/gpu-test -R 'kvarn|speculative_verification|kv_storage' --output-on-failure
# 100% tests passed, 0 tests failed out of 9（約 18 秒）
```

- `qwen_attention_kvarn_read_ops_test` / `_seal_` / `_prefill_` / `_prefill_read_ops_test` / `qwen_kvarn_quant_cpu_test` / `qwen_kvarn_fwht_cpu_test` / `qwen_attention_kv_storage_ops_test` / `speculative_verification_test`
- `qwen_kvarn_error_test` は従来どおり Skipped（外部モデル必須）。
- 最適化は加算順序を変えておらず、上記はビット一致を維持したまま通過。

---

## 5. 問題点

### P1. G1 未達（最重要）
d32K で 0.855〜0.936、d128K で 0.640〜0.813。基準は ≥1.00（k4v4 の 128K は ≥1.15）。**全プリセット・全テール長で未達**。

### P2. 性能契約の実現可能性に疑義
エージェントの計測（`BENCHMARKS.md` の「KVarN KV read path」節）:
- d128K で FP16 の読み出しは **35 ms ≒ 8.6 GB/token ÷ ~245 GB/s** の純帯域律速。KVarN の読み出しは最適化後 **73 ms**（メモリ床は 2.1 GB → ~8.6 ms）。
- 契約の 1.15×（=144 ms/token）は KVarN の読み出しに **≤13 ms** を要求。FP16 が 35 ms の帯域下に隠している softmax 計算を ~8.6 ms 以内に収める必要があり、同じ softmax をしつつビット展開も行う現行アルゴリズムでは非現実的と見られる。
- 32K の 1.00×（KVarN 読み出し ≤8 ms、現在 17.6 ms）は削減余地があるが、128K の 1.15× は不確実。

### P3. 計画の順序不整合（設計上の問題）
G1 はフェーズ 1（decode のみ）のゲートとして定義されているが、その測定手段 `gufo bench` は **chunk prefill** を必ず実行するため、実際にはフェーズ 2 の 4.1（チャンクワークスペース）・4.2（prefill KVarN 読み出し）と、`-d>0` のための 5.1 の一部（インメモリ KVarN スナップショット）を先に実装する必要があった。今回はこれらを先行実装して測定した。

### P4. batched decode の KVarN 未配線（修正済み、回帰テスト未整備）
静的レビューで発見。`src/models/qwen/hip/batched_decode.cpp` の `ForwardVerificationBatch`（`serve llm` の投機検証経路）が KVarN を考慮せず canonical FP16 平面（KVarN では null）を読み書きし、**投機デコードの出力が黙って壊れる**状態だった。
- 修正: `kvarn_enabled` を融合フラグに含め、未融合時に明示エラー、`LaunchBatchedFusedQKNormRoPEKvWrite` に `KvarnWriteParams`、`LaunchCausalDecodeAttention` に `KvarnReadParams` を渡す。
- 残課題: この経路（DFlash2 の greedy 一致）の回帰テストは未整備（タスク 4.4 の範囲）。smoke のみ（`--speculative dflash2 --dflash-model …DFlash2-Q8_0.gguf --kv-cache kvarn-k4v4` 完走、tg16-dflash2 7.42）。

### P5. KVarN のディスクスナップショット版 3 が未実装（タスク 5.1/5.2 の残り）
`CompactPayloadBytes` / `SerializeCompact` / `ParseCompactSnapshotLayout` は canonical 平面前提で、KVarN セッションの永続保存は拒否される。今回は `bench -d` が使う **インメモリ `QwenGpuSnapshot` のみ** 実装した。G1 には不要だが、変更の完了条件には残る。

### P6. 測定証拠の制約
- g2 の sweep プロセスは 32 単位すべて `.done`（rc=0）を残したが、`run.log` に最終の "sweep complete" 行を書かずに終了した（各単位の成果物は完全）。再開可能な設計のため実害はない。
- g2 は FP16 ブラケットを原則 g1 と同値で扱っている（FP16 読み出し経路は当該変更で不変。d32K で 7.20 対 7.19 を再測）。
- 反復回数は 1 回/点（D16 の「各点 1 回のウォームアップ済み測定」）。±0.00 表示のためラン間ノイズは未評価。

### P7. テストハーネスの自己バグ（修正済み）
新規 `qwen_attention_kvarn_prefill_read_ops_test` が当初 `context=2305, tail=2048` でハングした。原因はハーネス側の KVarN body 確保が KV ヘッド分を欠き、seal 書き込みが確保外へ出ていたこと（production は影響なし）。修正後 22 の T-03 ケースと 20 の T-04 フラッシュ検査が通過。

### P8. 未コミットの広範な変更
phase 1 以降の変更はすべて未コミット（`src/models/qwen/hip/**`、`src/cli/**`、`tests/**`、`docs/**`、`tools/qwen27b/check.py` など）。`kvarn_read.hip` は未追跡ファイル。

---

## 6. タスク状態

| タスク | 状態 |
|---|---|
| 1.1〜3.10（フェーズ 0/1 実装） | 完了 |
| 4.1 prefill 融合 + チャンクワークスペース | 完了（`check.py` にテスト登録済み） |
| 4.2 prefill KVarN 読み出し（WMMA/tile/batched） | 完了 |
| 4.3 GEMM 経路の明示エラー | 完了 |
| **3.11 G1** | **未完了（本レポートの FAIL）** |
| 4.4〜6.6 | 未着手 |

---

## 7. 選択肢

1. **もう一段の最適化** — 占有を落とさない LDS ステージングと softmax 命令列の削減。32K の 1.00× は射程内、128K の 1.15× は不確実（P2）。
2. **性能契約の見直し** — design/spec の速度基準を達成可能値へ改訂、または基準をメモリ削減・C8 合計スループット等へ変更。要承認。
3. **G1 未達として確定し、変更を再策定/停止**。

---

## 付録: 再現コマンド

```sh
# ビルド
cmake --preset release -DCMAKE_C_COMPILER=/opt/rocm/bin/hipcc \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc
cmake --build build/release --target gufo --parallel 32

# G1 スイープ（D16: 切り離し実行・単位ごとの成果物・再開）
BIN=$PWD/build/release/gufo \
MODEL=/home/betiz/models/Qwen3.8-27B-Q8_0.gguf \
OUT=$PWD/artifacts/g2 \
setsid nohup /home/betiz/g1/run_g1.sh > /dev/null 2>&1 &

# 集計
python3 /home/betiz/g1/evaluate_g1.py artifacts/g2

# 正しさ
cmake --build build/gpu-test --target gufo --parallel 32
ctest --test-dir build/gpu-test -R 'kvarn|speculative_verification|kv_storage' --output-on-failure
```

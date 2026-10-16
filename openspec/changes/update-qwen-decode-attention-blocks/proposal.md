# Proposal

## Why

Qwen3.8 27B の decode・検証の注意は、KV を 1 キーずつ処理する構造のため、帯域の限界と計算の限界に同時に近い。

- 系列長 512 以上では、1 ブロックが 1 つの KV ヘッドの 6 クエリヘッドを受け持ち、4 トークンの K/V を共有メモリに置いて共有している（`src/models/qwen/hip/kernels/attention_decode.hip:868-887,735-760`）。それでも各ヘッドは、キーごとのレーン間縮約、2 回の `expf` による再スケール、値の蓄積を別々に行う（同 `676-733`）。系列長 128 以上 512 未満では、1 ブロックが 1 クエリヘッドを受け持つ。
- FP16 KV の AR decode は、KV の読み出しが 9.0 ms（32K）と 35.7 ms（128K、約 240 GB/s）かかる。KV を常駐させても計算だけで 33.9 ms（128K、1 トークン、16 フル注意層）かかる（`docs/models/qwen3.8-27b/BENCHMARKS.md:283-291`）。
- DFlash2 の検証では、2K から 32K への追加コストが、約 100 ms のサイクルに対して 0.11（1 行）〜0.35（8 行）を占める（Q4 ターゲットの C1 で測った制御器の表、`src/models/qwen/dflash_policy.hpp:300-306`）。Q8 DFlash2 mixed の C1 の 128K は llama.cpp より 23.7% 遅い（8.87 対 11.62 tok/s、`BENCHMARKS.md:73-80`）。
- KVarN は読み出しがキーごとの展開で計算律速になり、`kvarn-k5v4` テール 2048 の AR は 128K で FP16 比 0.752 にとどまる（`BENCHMARKS.md:206-214`）。

6 クエリヘッドを WMMA の M 次元に並べ、128 トークンの K・V タイルを 1 回だけ共有メモリに置く構造に組み替えれば、ヘッドごとの重複した縮約と再スケールを削れる可能性がある。そうなれば、DFlash2 の検証と KVarN の decode のコストを下げられる。

今取り組む理由: 変更 `add-kvarn-kv-cache-quantization` は design D19 で、GQA でのタイル共有など softmax の構造を変える作り直しを「別の change で行う」と繰り延べた。また、KVarN の品質本測定（T-08）と変更 `improve-kvarn-k6v5-quality-and-speed` が完了し、読み出しを作り直すときの比較の基準がそろった。

## What Changes

- decode・検証の注意を、「1 ブロック =（KV ヘッド, 行, パーティション）」で 6 クエリヘッドを WMMA の M 次元に並べ、その KV ヘッドの K・V タイルを 1 回だけ共有メモリに置く構造に組み替える。新しい FP16 の経路は専用の分割と統合を持ち、FP32 参照 KV の経路と KVarN の読み出しが使う分割と統合は変えない。
- Q と P は、f16 の上位と残差に分けて WMMA で積和する。採否の条件は、FP64 参照に対する最大絶対誤差と RMS 誤差が、同じ入力に対する基点のカーネルの値以下であること。分けない版は、単体ベンチで速度の参考として測るだけにする。
- 各行の結果は、他の行の値にも行数 R にも依存させない。そのため、行数によらず計算順序（head_dim 方向の分担、タイル分割、WMMA の累算順序、縮約木）を固定する。WMMA の外のスカラーの積和（再スケール、統合、後処理）は、明示的な `fmaf` に固定する。
- パーティションは 128 トークン境界に整列させ、数は 32 以下とし、各行の系列長だけから決める。分割の方式は、Phase 0 の単体ベンチで次の 2 つを両方測り、design に定めた基準で選ぶ。
  - A: `round_up(ceil(n/32), 128)` の長さで分け、長さの違う行を群に分けて起動する。
  - B: KVarN の既存方式。128 トークンのタイルを 32 個のパーティションに均等に配り、行ごとにカーネル内で決める。
- 系列長 128 未満のオンライン経路、HIP グラフ経路、FP32 参照 KV の経路（MTP の予測器の注意を含む）は変えない。これらの出力が基点のビルドとバイト一致することを、同じダンプを基点と候補のソースツリーでビルドして確かめる。
- 採用したら旧カーネルを削除し、切り替えのオプションや環境変数は残さない。Phase 1 では FP16 の split-K の部分注意カーネル、Phase 2 では KVarN の split-K と因果 decode の読み出しカーネルが対象になる。
- Phase 2 で、KVarN の decode 読み出しを同じ考え方の構造に移す。1 ブロック =（KV ヘッド, パーティション）でその起動の全行を処理し、封印タイルは 1 回だけ展開して、6 ヘッドと全行で共有する。WMMA は行ごとに回し、各行の算術は R に依存させない（利用者の決定、2026-10-13）。KVarN の折り込みは、WMMA の A 側と、タイルごとの FP32 の後処理に割り当てる。
- 採用するビルドでは、DFlash2 制御器の文脈コスト表を新しいカーネルで測り直してから、本番の DFlash2 を判定する。この表は Q4 ターゲットの判断と、Q8 ターゲットの複数リクエストの判断に使われ、Q8 ターゲットの単一リクエストの判断には使われない（`dflash_policy.hpp:109-113`）。
- **FP16 の互換同一性**: FP16 の decode と検証行の注意の算術が変わるため、FP16 の互換同一性に、その算術を表す項目を加える。FP32 と KVarN の互換同一性には加えない。既存要求 `qwen/kv-cache-quantization` の「KV ストレージ形式の選択」と「セッション状態の整合」を改め、FP16 の不変と upstream との一致の対象から decode と検証行の注意の算術を外す（利用者の決定、2026-10-13）。
- **BREAKING** は無い（利用者の決定、2026-10-13）。
  - 保つ契約: 検証行とスカラー decode の byte 一致（`tests/models/qwen/hip/attention/kv_storage_ops_test.cpp:84-118`）と、KVarN の T-03・T-05 のビット一致。
  - 出力の変化: 既定の FP16 の decode の算術が変わるため、出力は基点のビルドと一致しなくなりうる。ビルド間の出力一致は契約に含まれない。
  - キャッシュの再計算: 基点より前のビルドと upstream のビルドで作った FP16 のディスクキャッシュは、互換同一性の不一致により再利用されず、自動で再計算される。正しさは保たれるため、BREAKING として扱わない。KVarN の状態 ABI の更新を BREAKING としていない既存の扱いと揃える。
  - KVarN の読み出しの算術を変える場合は、既存要求 `qwen/kv-cache-quantization` の「KVarN の形式・演算の変更と版の更新」に従って状態 ABI を上げ、旧キャッシュを再利用しない。
- **基準値（`qwen/kv-cache-quantization` の性能契約）**: FP16 の decode の注意を置き換えると、KVarN の tg の FP16 比の分母が変わる。そこで Phase 1 を採用したビルドで、深度 32K 以上の AR と DFlash2 の tg の FP16 比をプリセット・テール長・深度ごとに測り直し、それを新しい基準値とする。pp は FP16 の分母が変わらないため、既存の基準値を据え置く（利用者の決定、2026-10-13）。Phase 2 は、その基準値と、既存の下限（深度 0〜4K の AR tg、深度 0 の DFlash2 tg、C8 AR の合計スループット、起動時間、p99 / p50）で判定し、変更 `improve-kvarn-k6v5-quality-and-speed` が繰り延べた判定もあわせて適用する。
- **基点**: 比較の対照は、この変更の候補が `kvarn-kv-cache-quantization` 上で分岐したリビジョン（基点）とする。`main` と `upstream/main` は KVarN を含まないため、対照に使わない。
- **測定**:
  - Phase 0: 単体ベンチで、基点のカーネルを逐語コピーした対照と比べる。cold と resident の両方を、5 回以上の中央値で測る。resident は、1 層分の KV がキャッシュに収まる文脈（2304〜9216 トークン）の掃引から深度 128K に換算する。FP64 の参照と、飽和しない入力（`long_context_ops_test` と同じ生成式）を使う。
  - 本番: `BENCHMARKS.md` と同じ方法で基点と交互に測る。リクエスト・フェーズごとの時間で、差が 5% と 3 ms の両方を超える場合を有意な差とする（AGENTS.md）。
  - GPU を使う測定は、他の GPU ジョブや、変更 `update-qwen38-flash-next-decode-attention-blocks` の測定と時間を重ねない。
- **ゲート**:
  - 単体ベンチ: G0F（FP16。精度・独立性・計算の下限・検証速度・AR 退行なし）と G0K（KVarN。主構成 `kvarn-k5v4` テール 2048 の精度・独立性・速度）の閾値と評価点は確定値とする。速度は深度 32K と 128K の cold で判定する（利用者の決定、2026-10-13）。
  - 本番 G1F: Q8_0 の重みで、`tools/qwen27b/check.py` の `kernels`・`model`・`multi-session`・`serving` と、`tests/functional/` の `long-context`・`cache` に AR と DFlash2 で通ること。C1 の AR の tg と pp と、C8 の AR と DFlash2 に、基点からの有意な退行が無いこと。C1 の DFlash2 の tg が深度 32K・64K・128K で有意に改善すること（利用者の決定、2026-10-13）。
  - 本番 G1K: T-01〜T-05、上記の基準値と下限、繰り延べた判定、`kvarn-k5v4` テール 2048 の T-08 の q8→q4 内分位置 p が、基点から 0.05 を超えて悪化しないこと。
- ゲートに届かない場合は、作業を止めて、測定値と原因の分析と選択肢を利用者に返す。
- **文書**:
  - 採用可否、ゲートの測定値、不採用の場合の理由を、`docs/models/qwen3.8-27b/{BENCHMARKS,EXPERIMENTS,QUALITY}.md` に記録する。
  - KVarN の FP16 比の表、状態 ABI の更新、FP16 の互換同一性の更新とディスクキャッシュの再計算は、`docs/models/qwen3.8-27b/README.md` と `docs/CLI.md` にも記録する。

### Qwen3.8 Flash-Next を対象外とする理由

Flash-Next にも同じ考え方（KV ヘッドごとにクエリヘッドを WMMA の M 次元へまとめる）を適用するが、別の変更 `update-qwen38-flash-next-decode-attention-blocks` で扱う（利用者の決定、2026-10-13）。カーネル・テスト・数値契約はモデルごとに持ち、採否・ゲート・時期も独立している。同じ変更に含めると、27B の Phase 2 を待つ間、Flash-Next の仕様の確定が止まる。この変更は Flash-Next のコード・テスト・文書を変更しない。Flash-Next の互換同一性（`state_abi=qwen38-flash-next-rocm-session-v1`）も変えない。

## Capabilities

### New Capabilities

- `qwen/decode-attention`: Qwen3.8 27B の decode・検証の注意を、KV ヘッドごとにクエリヘッドを WMMA の M 次元へまとめる構造で計算する機能。次の規則を定める。
  - 数値一致と基点の定義
  - ブロックの受け持ちとタイルの共有
  - 行の独立性とビット一致
  - FP64 に対する精度の非劣化
  - パーティションの不変条件
  - 変更しない経路（基点とのバイト一致）
  - FP16 の互換同一性の更新
  - 旧カーネルの削除（段階的な併用はしない）
  - KVarN 読み出しの同じ構造への移行と、折り込みと状態 ABI
  - ゲート（繰り延べた判定の適用を含む）と測定方法

### Modified Capabilities

- `qwen/kv-cache-quantization`: 次の 3 つの要求を変える。
  - 「KV ストレージ形式の選択」: FP16 の保持バイト列の不変から、FP16 の decode と検証行の注意の算術による KV の値の違いを除き、その場合は FP16 の互換同一性を変えることを求める。
  - 「セッション状態の整合」: FP16 の同一性キーとスナップショットの upstream との一致を、FP16 の decode と検証行の注意の算術が upstream と同じビルドに限る。算術を変えたビルドでは、upstream の同一性キーにその算術を表す項目を加えたものと一致させ、FP32 と KVarN の同一性キーには加えない。
  - 「性能契約」: FP16 の decode または検証の注意の演算を置き換えた場合は、置き換え後のビルドで深度 32K 以上の AR と DFlash2 の tg の FP16 比を測り直して新しい基準値とする。pp の基準値は据え置く。下限と 0.97 倍の退行規則そのものは変えない。

  KVarN の読み出しの算術を変える場合の状態 ABI の更新は、既存の「KVarN の形式・演算の変更と版の更新」で扱い、要求文は変えない。

## Impact

- **コード**:
  - `src/models/qwen/hip/kernels/attention_decode.hip`: FP16 の新しい split-K の部分注意と統合、`LaunchAttention`、`LaunchCausalDecodeAttention`。FP32 参照 KV の経路の `QwenDecodeSplitKAttentionPartialsKernel` と `QwenDecodeSplitKAttentionReduceKernel` は変えない
  - `src/models/qwen/hip/detail/attention_policy.hpp`: FP16 の新しい経路のパーティションの境界と数（専用の関数を加える。FP32 と KVarN が使う `SelectDecodeAttentionSplitCount` は Phase 1 で変えない）
  - `src/models/qwen/hip/ops/attention.hpp`: 起動 API
  - `src/models/qwen/hip/kernels/kvarn_read.hip`: Phase 2 の読み出し構造
  - `src/cli/serve/inference_backend.cpp`: FP16 の互換同一性の項目（`QwenCompatibilityIdentity`）と、Phase 2 の KVarN の状態 ABI 名（`QwenStateAbiName`）
  - `src/models/qwen/dflash_policy.hpp`: 文脈コスト表 `kQ4ContextCost` の再測定値
  - 対象外: 系列長 128 未満のオンライン経路、`attention_decode_graph.hip`、FP32 参照 KV の経路
  - MTP の予測器の注意は FP32 平面を渡して FP32 経路に入るため、変わらない（`src/models/qwen/hip/mtp/executor.cpp:100-106,226-230`、`attention_decode.hip:922-926`）。ただし MTP の入力となる対象モデルの隠れ状態は FP16 の decode の変化で変わりうる。
  - `src/cli/bench/kernel_bench_main.cpp` は `SelectDecodeAttentionSplitCount` で FP32 の decode 注意の分割数を表示している。この関数は変えないため、変更しない
- **テスト・ツール**:
  - `tests/models/qwen/hip/attention/{long_context_ops_test,kv_storage_ops_test,backend_ops_test,kvarn_read_ops_test}.cpp`
  - `tests/models/qwen/hip/attention_policy_test.cpp`（新しい分割関数の検査を加える。既存の `SelectDecodeAttentionSplitCount` の検査は変えない）
  - 新規の Phase 0 単体ベンチと、変更しない経路のダンプ（`tools/bench/` の standalone HIP、`tools/bench/build.sh`）
  - `tools/qwen27b/kvarn_read_compute_bench.hip`（resident 条件の枠組みと、KVarN の封印の固定入力）
  - `tests/models/qwen/hip/quality/kvarn_quality_eval.cpp`（T-08）
  - `tools/qwen27b/check.py` の各スイート、`tests/functional/` の `long-context`・`cache`
  - CTest `kernel_resources_test`（`tools/ci/kernel-resources.json`。新しいカーネルの scratch が 0 であること）
- **文書**: `docs/models/qwen3.8-27b/{BENCHMARKS,EXPERIMENTS,QUALITY,README}.md` と `docs/CLI.md`。DFlash2 制御器の文脈コスト表の測り直し、KVarN の tg の基準値の測り直し、FP16 の互換同一性の更新を含む。
- **依存・互換**:
  - 新しい外部依存は無い（HIP のみ）。新しいオプション・環境変数・切り替え経路も追加しない。
  - Phase 1 で FP16 の互換同一性に項目を加える。基点より前のビルドと upstream のビルドで作った FP16 のディスクキャッシュは再計算になる。FP16 のスナップショットの版と形式、FP32 と KVarN の互換同一性は変わらない。
  - KVarN の読み出しの演算が変わる場合（Phase 2）は、状態 ABI を上げる。
  - 既定の FP16 の出力はビルド間で一致しなくなりうる（BREAKING ではない）。
- **関連変更**: `update-qwen38-flash-next-decode-attention-blocks`（Flash-Next。採否と時期は独立）。先行計画 `openspec/changes/update-qwen27b-block-decode-attention` は存在しない（`openspec/changes/`、全ブランチ、作業ツリーを検索して確認済み）。

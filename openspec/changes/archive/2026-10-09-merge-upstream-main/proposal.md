# Proposal

## Why

フォークの `kvarn-kv-cache-quantization` ブランチは upstream（`gufo-org/gufo`）の `main` に対して 72 コミット遅れ、39 コミット先行している（`upstream/main` = `f17e37b`、2026-10-07。merge-base = `f783fed`、2026-09-30）。先行分の大半は Qwen3.8 27B の KVarN KV 量子化とその検証（ハーネス・文書）で、遅れ分はサーバ機能（セッション/メトリクス、Responses API、ツール呼び出しの整形、RAM/ディスクキャッシュの予算）と Qwen3.8-Flash-Next の性能改修（#445・#463・#470・#421・#459）である。遅れが広がるほど衝突面と回帰範囲が増えるため、いま取り込んでビルドと KVarN 系テストで確認する。

衝突面は確認済みで、7 ファイル 22 ハンクのいずれも「両側が同じ位置に挿入」型であり、どちらかの変更を捨てる必要のある衝突はない。

## What Changes

- `upstream/main`（`f17e37b`）を `kvarn-kv-cache-quantization` へマージする。衝突の解決は両側の変更を保持することを原則とし、フォーク側の KV 形式・テール長の意味と公開オプションの契約を変えない。
- `load()` の衝突（`src/cli/serve/inference_backend.hpp` 5 ハンク、`inference_backend.cpp` 8 ハンク、`serve.cpp` 2 ハンク）を、両側の新引数を残して解決する。KVarN 側の `TextKvCacheConfig` と upstream 側の `TextRunnerRamCacheOptions` を、4 つのオーバーロード（宣言と定義）と 4 つの呼び出し（`inference_backend.cpp` の転送 3・`serve.cpp` 1）で同じ順序に揃える。`vision_model_path` を持つのはパス版のオーバーロードだけである。`serve.cpp` の起動検証は KVarN の `--kv-cache`・`--kv-cache-tail` 検証と upstream の `--log-level` 検証（`--log-progress` との組み合わせを含む）の両方を残す。
- `InferenceBackend` のアクセサ衝突（`kv_cache_format()` と `session_states()`）を両方残し、`src/models/qwen38_flash_next/kernels/rocm/executor.cpp` の衝突は、upstream の `Session::PreserveSnapshots` ほか SnapshotAllocator 群（437 行）とブランチの `Executor::RecordRoute`（8 行、d64a25c の route recording）を両方残して解決する。
- `tests/models/qwen38_flash_next/CMakeLists.txt` は、ブランチの CLI load-error・route_ops・ap_upload・swift_upload テストと、upstream のベンチ実行ファイル `dense_gemm_bench`・`attn_bench`（`EXCLUDE_FROM_ALL`、ctest には登録されない）を両方残す。`docs/models/qwen3.8-27b/EXPERIMENTS.md` と `docs/models/qwen3.8-flash-next/QUALITY.md` は両側の表行を両方残す。
- 自動マージされる 18 の両側変更ファイルのうち、意味が重なる箇所を確認する: `QwenCompatibilityIdentity` の構成（ブランチの `QwenKvStorageIdentity`・`QwenStateAbiName` と upstream の `checkpoint_frontier=complete-prefix-v1` が同居していること）、27B の compact snapshot 検証（upstream の `image_count` を payload 由来の上限にする修正と KVarN 版の検証の両立）、`QwenGpuExecutor` のサンプラ呼び出し（`NeedsConstraintMask()`）、flash-next の性能改修と route recording の共存。
- **BREAKING**: 同一性キーへの upstream の `checkpoint_frontier=complete-prefix-v1` の追加を受け入れる。この項目は base（`f783fed`）に無く upstream が追加したもので（#369）、KV 形式に関係なく常に入る。このため、マージ前のビルドで作成した FP16・KVarN の `.kvc` ディスクキャッシュは一度だけ再利用されなくなる（再計算で回復し、移行作業は不要）。これに合わせて、`--kv-cache` を指定しない場合の FP16 の互換同一性キー・スナップショット・出力の基準を「変更前」から「同じコミットの upstream のビルド」に改める（`kv-cache-selection` の「既定の KV 形式」と `qwen/kv-cache-quantization` の「セッション状態の整合」を MODIFIED）。以後 upstream が同一性キーを変えても spec の改定は要らない。
- サーバの識別子は upstream の正規名を維持し、改名・エイリアス化・削除をしない（`--log-level`、`-v/--verbose`、`--trace`、`--max-pending` を含む。`-v/--verbose` は upstream の定義どおり `--log-level=debug` の短縮形として残す）。KVarN の `--kv-cache`・`--kv-cache-tail` の値・検証・終了コード・help 表記も変えない。スナップショット形式の版と既存版（v1/v2）の読み込み互換は変えない。
- upstream が取り込むキャッシュ機構（RAM プレフィックスキャッシュ、実行中プレフィックスの共有、学習した境界、安定チェックポイント、ディスク staging）が KVarN の状態契約を満たすことを、upstream の functional cache スイートを KVarN 構成で実行して確認する。再利用・復元後の受け入れ基準は greedy（temperature 0）のトークン列がキャッシュなしの実行と一致することとする。
- 検証のビルドは Nix で別ディレクトリ（`build/gpu-test-nix`）に構成する。既存の hipcc 構成の `build/gpu-test` は残し、マージ前（`d64a25c`）の参照ビルドとしてマージ前キャッシュの作成に使う。VRAM と GPU を使うテストは並列実行せず、`ctest` に並列オプションを付けない。KVarN の構成は `kvarn-k4v4` + `--kv-cache-tail 512` と `kvarn-k6v5` + `--kv-cache-tail 2048`、文脈長は 256 と 131,072 に絞る。品質は 131,072 で `kvarn-k6v5`/2048 が p ≤ 0.15、`kvarn-k4v4`/512 が p ≤ 1 を満たすことを判定し、256 は封印済みタイルを読まないため記録のみとする。KVarN の既定は `fp16` のままとする。

公開オプションの改名・削除はなく、マージ前の `--kv-cache`・`--kv-cache-tail` の指定は同じ結果になる。破壊的な点は上記 **BREAKING** のキャッシュ鍵の更新だけで、upstream が同じ理由で全利用者に適用している意図的な変更である。

## Capabilities

### New Capabilities

なし。この変更は既存機能の維持と upstream の取り込みであり、新しい機能をフォーク側で定義しない。

### Modified Capabilities

- `kv-cache-selection`: `serve llm` で `--kv-cache`・`--kv-cache-tail` がサーバのログ・トレース・キューのオプションと同じコマンド行で併存し、検証がモデル読み込み前に完結することを要求する（ADDED）。`--kv-cache` を指定しない場合の KV 形式・出力・スナップショット・互換同一性キーの基準を、同じコミットの upstream のビルドに改める（MODIFIED「既定の KV 形式」）。
- `qwen/kv-cache-quantization`: RAM プレフィックスキャッシュ、実行中プレフィックスの共有、学習した境界、安定チェックポイント、ディスク staging の下で、同一性キーが KV 形式・state abi・`checkpoint_frontier` を含み、KV 形式の異なる状態を再利用せず、再利用・復元後の greedy トークン列がキャッシュなしの実行と一致することを要求する（ADDED）。FP16 のスナップショットと同一性キーの基準を同じコミットの upstream に改める（MODIFIED「セッション状態の整合」）。

## Impact

- マージ対象: `upstream/main`（`f17e37b`）72 コミット。衝突 7 ファイル、自動マージで通る両側変更 18 ファイル（計 25 ファイル）。
- serve: `src/cli/serve/{inference_backend.hpp,inference_backend.cpp,serve.cpp}`（`load()` の引数、起動検証、`QwenCompatibilityIdentity`、`session_states()`、RAM キャッシュオプション）。
- flash-next: `src/models/qwen38_flash_next/kernels/rocm/executor.cpp`（`RecordRoute` と SnapshotAllocator 群）、`src/models/qwen38_flash_next/{engine.cpp,engine.hpp,kernels/rocm/*,ngram.cpp}`（性能改修と route recording の共存）。
- 27B: `src/models/qwen/hip/{arena.cpp,executor.cpp}`＋`src/models/qwen/CMakeLists.txt`（compact snapshot の検証、サンプラ API、KVarN 変更との共存）。
- CLI・テスト: `src/cli/prompt/prompt.cpp`、`CMakeLists.txt`、`cmake/Checks.cmake`、`tests/models/qwen38_flash_next/{CMakeLists.txt,projection_ops_test.cpp}`、`tests/models/qwen/hip/attention/kv_storage_ops_test.cpp`。upstream が追加する `tests/functional/` の cache スイートは変更せず実行だけする。
- 文書: `docs/CLI.md`、`docs/models/qwen3.8-27b/EXPERIMENTS.md`、`docs/models/qwen3.8-flash-next/{EXPERIMENTS.md,QUALITY.md,artifacts/model-identities.json}`。
- 仕様: `openspec/specs/kv-cache-selection/spec.md` と `openspec/specs/qwen/kv-cache-quantization/spec.md`（ADDED 各 1、MODIFIED 各 1）。
- 利用者への影響: マージ前に作成した `.kvc` ディスクキャッシュは一度だけ再計算になる（**BREAKING**、上記）。
- ビルドと成果物: Nix の構成 `build/gpu-test-nix` を新設する。`build/gpu-test`（hipcc）はマージ前の参照ビルドとして `d64a25c` で再ビルドし、マージ後は再ビルドしない。測定の記録は `artifacts/test-ledger.jsonl` と `~/.cache/gufo/` 配下の新しいラベルに残す。
- 依存: 新規外部依存なし。リモート `upstream`（`git@github.com:gufo-org/gufo.git`）の登録と fetch は確認済みで、`upstream/main` の ref を検証の基準にする。
- リスク: upstream の flash-next 性能改修（#445・#463・#470 など）と本ブランチの route recording は同じ関数群に触れるため、自動マージ後にコンパイルと数値の両方を確認する必要がある。KVarN の品質・性能の契約値は再測定せず、品質は 131,072 で既存アンカー（`g0-tail`）に対する p の閾値だけを判定する。

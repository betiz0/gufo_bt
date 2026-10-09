# Design

## Context

- 取り込み元: `upstream/main` = `f17e37b`（2026-10-07、72 コミット）。取り込み先: `kvarn-kv-cache-quantization`（`d64a25c`、39 コミット）。merge-base は `f783fed`（2026-09-30）。リモート `upstream`（`git@github.com:gufo-org/gufo.git`）は登録・fetch 済み。
- ドライランの結果（`git merge-tree --write-tree` と一時 worktree での実マージで確認済み）: 衝突 7 ファイル 22 ハンク、両側が変更したファイルは 25、うち 18 は自動マージ。
- 7 つの衝突はすべて「両側が同じ位置に挿入」型で、どちらかを捨てる必要のある衝突はない。内訳: `load()` の新引数（`inference_backend.hpp` 5・`inference_backend.cpp` 8・`serve.cpp` 2）、`InferenceBackend` のアクセサ（`kv_cache_format()` 対 `session_states()`）、flash-next `executor.cpp` の `Executor::RecordRoute`（フォーク、8 行）と upstream の SnapshotAllocator/Session 群（437 行）、テスト登録 CMake（`tests/models/qwen38_flash_next/CMakeLists.txt`）、文書 2 ファイルの表行。
- 自動マージ側の意味が重なる箇所: `QwenCompatibilityIdentity` の構成（フォークの `QwenKvStorageIdentity`・`QwenStateAbiName` と upstream の `checkpoint_frontier=complete-prefix-v1`）、27B compact snapshot の検証（upstream が `image_count` の上限を payload 由来に変更）、`QwenGpuExecutor` の `NeedsConstraintMask()` への置換。
- upstream の `checkpoint_frontier` は KV 形式に関係なく同一性キーに入る（`upstream/main:src/cli/serve/inference_backend.cpp` の `QwenCompatibilityIdentity`、#369）。そのため FP16 のキーも変わり、現行 spec の「FP16 の同一性キーは変更前とバイト一致」「変更前の `.kvc` を再利用できる」と両立しない。この変更は該当する 2 つの Requirement を MODIFIED し、キャッシュ機構の契約を ADDED する（proposal の Capabilities を参照）。
- 既存ビルド: `build/gpu-test` は `CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc` の Ninja 構成で、`gufo` は 2026-10-04 のビルド（`d64a25c` より古い）。この端末では `nix develop` が失敗した記録がある（2026-10-08 時点、未再確認）。
- 検証資産: upstream は `tests/functional/run.py` と cache スイート（`cache`・`cache-edits`・`cache-growth`・`cache-shared-prefix`・`cache-rotation`・`cache-concurrency`・`cache-depth`）を追加する。各スイートは temperature 0 の応答をキャッシュなしの応答と完全一致で比較し、`cache` はディスクキャッシュ（staging 1 GiB）と再起動を使う。フォークの multi-session テスト（`qwen27b_multisession_target_test`・`qwen_dflash_multisession_gpu_test`・`inference_backend_multisession_gpu_test`）は `multi-session` ラベルで、`gpu-full` プリセットから除外される。

## Goals / Non-Goals

**Goals:**
- 7 つの衝突を両側保持で解決し、自動マージされた 18 ファイルの両側変更が意味的に共存していることをビルドとテストで確認する。
- Nix の別ディレクトリ構成でビルドし、KVarN 系・flash-next 系・CLI・serve の対象テストを通す。
- upstream のキャッシュ機構が KVarN 構成で greedy トークン列の一致を保つこと、マージ前のキャッシュが鍵の不一致で再計算されることを確認する。
- 公開オプションと KV 選択契約、フォーク側の KV 形式・テール長の意味を変えない。

**Non-Goals:**
- upstream の新機能（Responses API、メトリクス、`--trace` など）の仕様化・再実装。
- KVarN の品質・性能の再測定（131,072 での p の閾値判定以外は既存の記録を基準にする）。
- upstream のビルドを別に作り、FP16 の出力やキャッシュを実行で比較すること（D7 の静的確認と既存の FP16 テストで代える。利用者の決定 2026-10-08、レビュー Q-1）。
- フォークの `main` の更新、upstream の側ブランチ（`fedeizzo/cache-redesign` など）の取り込み、リリース・CHANGELOG の更新。
- ヒストリの書き換え（rebase・cherry-pick・push 後の reset）。

## Decisions

### D1: 取り込み方法はマージコミット

`git merge --no-commit --no-ff upstream/main` で取り込み、解決と検証の後に 1 つのマージコミットにする。`d64a25c` が先行するため fast-forward にはならない。メッセージは Conventional Commits の 1 行（`chore: merge upstream/main f17e37b`）。コミット前に AGENTS.md の format check（`tools/ci/check-format.py`）を通す。

- 代替案: rebase — 39 コミットそれぞれで衝突を解き直すことになり、解決の記録が分散する。cherry-pick — 同一ファイル（`load()`・flash-next `executor.cpp`）で衝突が繰り返し発生する。いずれも衝突が 1 回で済むマージより高コスト。

### D2: `load()` はフォークの `kv_cache_config` を `disk_cache_config` の直後に置く

4 つのオーバーロード（`inference_backend.hpp` の宣言と `inference_backend.cpp` の定義）と 4 つの呼び出し（`inference_backend.cpp` のパス版から各モデル版への転送 3・`serve.cpp` 1）の引数順を、オーバーロードごとに次に統一する。

- パス版（`const std::string& model_path`）: `..., const TextDiskCacheConfig& disk_cache_config = {}, const TextKvCacheConfig& kv_cache_config = {}, const std::string& vision_model_path = {}, TextRunnerRamCacheOptions ram_cache_config = {}`
- モデル共有版 3 つ（`QwenGpuModel`・DeepSeek・Flash-Next）: `..., TextDiskCacheConfig disk_cache_config = {}, TextKvCacheConfig kv_cache_config = {}, TextRunnerRamCacheOptions ram_cache_config = {}`。`vision_model_path` はどちらの側にも無いので追加しない。

- 理由: upstream が定めた末尾の相対順（`vision_model_path` → `ram_cache_config`）を変えないので、以後の upstream 差分が小さく保たれる。フォークの `kv_cache_config` は既存の `disk_cache_config` の直後に置き、フォーク側の呼び出し（`tests/models/qwen27b/serving_test.cpp` の 9 番目の位置引数など）をそのまま使える。upstream のテストは `ram_cache_config` を位置引数で渡していない（`git grep` で確認）。
- 代替案: upstream の並びの後ろに `kv_cache_config` を足す — フォーク側の位置引数の意味がずれ、`vision_model_path` を受け取る呼び出しの変更量が増える。
- 型と渡し方は各側のものをそのまま使う（`kv_cache_config` はパス版が `const&`、モデル共有版が値渡し。`ram_cache_config` は値渡し）。

### D3: `InferenceBackend` のアクセサは両方残す

`kv_cache_format()`（フォーク）と `session_states()`（upstream）の宣言と out-of-line 定義を両方残す。どちらも既存の呼び出し元があるため、片方を落とすとリンク時またはテストで失敗する。

### D4: flash-next `executor.cpp` は「フォーク → upstream」の順で連結する

衝突アンカーの位置（無名名前空間の閉じた直後）で、フォークの `Executor::RecordRoute` 定義を先に、upstream の `Session::PreserveSnapshots` ほか SnapshotAllocator 群を後に置く。両者とも名前空間スコープのメンバー定義なので順序はコンパイル結果に影響しない。

### D5: テスト登録は両側を保持する

`tests/models/qwen38_flash_next/CMakeLists.txt` は、フォークの `qwen38_flash_next_cli_load_error`・`route_ops`・`ap_upload`・`swift_upload` のテスト登録と、upstream の `qwen38_flash_next_dense_gemm_bench`・`qwen38_flash_next_attn_bench` の実行ファイル定義をすべて残す。upstream の 2 つは `EXCLUDE_FROM_ALL` の `add_executable` で ctest には登録されないため、ターゲットを明示してビルドできることで確認する。フォークのテストは `ctest -N` の一覧で確認する。

### D6: 文書は両側の行を保持する

`docs/models/qwen3.8-27b/EXPERIMENTS.md`、`docs/models/qwen3.8-flash-next/{EXPERIMENTS.md,QUALITY.md}` は、フォークの行を残したまま upstream の行を追記する。表の行順は意味を持たないため、解決の差分を最小にする順（フォーク → upstream）を採る。

### D7: 同一性キーは両側の項目を保持し、FP16 の基準を upstream に移す

`QwenCompatibilityIdentity` は upstream の `checkpoint_frontier=complete-prefix-v1` と、フォークの `state_abi`（`QwenStateAbiName` による KV 形式込みの名前）・`kv_storage=`（`QwenKvStorageIdentity`）の両方を含む（自動マージの結果をそのまま採る）。この追加によりマージ前に作成した `.kvc` ディスクキャッシュは再利用されなくなる。これは upstream の #369 が同じ理由（古い学習境界の logits を再利用しない）で全利用者に適用している意図的な変更であり、再計算で回復するため移行作業は行わない（**BREAKING**、利用者の決定 2026-10-08）。

spec は、`--kv-cache` を指定しない場合の KV 形式・出力・スナップショット・FP16 の同一性キーの基準を「変更前」から「同じコミットの upstream のビルド」に改める（`kv-cache-selection`「既定の KV 形式」、`qwen/kv-cache-quantization`「セッション状態の整合」を MODIFIED）。FP16 の同一性キーが upstream と一致することは、`QwenStateAbiName`・`QwenKvStorageIdentity` の非 KVarN 分岐が upstream の式と同じ値を返し、`QwenCompatibilityIdentity` の行の並びが `upstream/main` と同じであることをコードで確認する。FP16 の出力とスナップショットは既存の FP16 テストで退行を確認し、upstream のビルドとの実行比較は行わない（Non-Goals）。MODIFIED の該当 2 シナリオは、規範文（SHALL）を維持したまま、担保の記述をこの検証方法に合わせる（同一性キーはコード上の等価確認、出力とスナップショットは既存の FP16 テストの退行確認。利用者の決定 2026-10-08、レビュー Q-1）。退行確認を担うテストは `inference_backend_gpu_test` と `qwen27b_target_test`（タスク 5.7）。

- 代替案: 基準をこのマージ後のキーに固定する — upstream がキーを変えるたびに spec の改定が要る。FP16 だけ `checkpoint_frontier` を外す — 現行 spec はそのまま成り立つが、#369 が防ぐ古い学習境界の logits の再利用が FP16 で起こりうる状態が残り、upstream との差分も増える。

### D8: ビルドの構成

- Nix で別ディレクトリに構成する: `nix develop -c cmake --preset gpu-test -B build/gpu-test-nix`、`nix develop -c cmake --build build/gpu-test-nix --parallel 8`。`EXCLUDE_FROM_ALL` の `qwen_kvarn_quality_eval`・`qwen38_flash_next_model_tests`・`qwen38_flash_next_dense_gemm_bench`・`qwen38_flash_next_attn_bench` はターゲットを指定して追加でビルドする（利用者の決定 2026-10-08）。
- この端末で flakes が無効な場合は `nix --extra-experimental-features 'nix-command flakes' develop` とする。`nix develop` 自体が失敗した場合は作業を止めて利用者に報告し、hipcc の構成へ切り替えない。
- 既存の `build/gpu-test`（hipcc）は残す。マージ前に `d64a25c` の作業ツリーで `gufo` を再ビルドし、マージ前の参照ビルドとして D11 のキャッシュ作成に使う。マージ後は再ビルドしない。
- ビルドが 600 秒を超える見込みの場合は `nohup ... > /tmp/merge-upstream-build.log 2>&1 &` で起動し、`ps -p` とログ末尾の単発確認で追う（ポーリングループは使わない）。

### D9: テストの実行規則と対象

- 実行規則: VRAM と GPU を使うテストは並列実行しない。`ctest` に `-j`／`--parallel` を付けず、複数の GPU テストを同時に起動せず、GPU を使うサーバやベンチと同時に走らせない（`ctest` の既定は直列で、`gpu-full` プリセットにも並列設定は無い）。CPU のみのテストは並列でよい。
- ctest は別ディレクトリを対象にするため、`gpu-full` と同じ除外を明示する: `nix develop -c ctest --test-dir build/gpu-test-nix -LE multi-session -R '<regex>' --output-on-failure`。multi-session スイートは `-L multi-session` で別に実行する。
- 数分を超える実行は `tools/qwen27b/ledger_run.py run` で 1 本ずつ起動する（アーカイブ済みの変更 `2026-10-08-add-kvarn-kv-cache-quantization` の design D22・D23）。`--id`・`--task`・`--purpose`・`--estimate-seconds`・`--estimate-basis`・`--pass-criteria` を指定し、見積もりは 1 時間以内に分割する。見積もりの 2 倍に達すると `ledger_run.py` がハーネスのプロセスグループだけを停止する。完了は `ledger_run.py status`／`wait` かログの単発確認で待ち、1 回のツール呼び出し内で完了を待たない。
- `ledger_run.py` は起動したプロセスを監視するため、ハーネスは前面で実行する。`kvarn_consistency.py run` には `--foreground` を付ける。`kvarn_quality.py run` はハーネスの場所を `build/gpu-test` に固定しているため使わず、`build/gpu-test-nix/tests/models/qwen/qwen_kvarn_quality_eval` を直接起動し、集計だけ `kvarn_quality.py report` で行う。
- ハーネスは同じラベルの成果物（unit と教師 logits）を再利用して再開する。unit 名にビルドの識別は無いため、実行ごと・ビルドごとに新しいラベル（`merge-upstream-` で始める）を使い、既存のラベルを再利用しない。
- 対象:
  - KVarN: `qwen_kvarn_(quant|fwht)_cpu_test`、`qwen_attention_kvarn_(seal|read|prefill|prefill_read|state)_ops_test`、`qwen_attention_kv_storage_ops_test`、`qwen_graph_ops_test`、`qwen_kvarn_error_test`、`qwen27b\.dflash-kvarn`
  - flash-next（衝突・自動マージの影響面）: `qwen38_flash_next\.(config|ngram|format_policy|cpu_ops|projection_ops|attention_ops|mmq_ops|route_ops|ap_upload|swift_upload)`、`qwen38_flash_next_cli_load_error`、`qwen38_flash_next_model_tests` の `qwen38_flash_next_session_test`・`qwen38_flash_next_snapshot_test`（`--model`・`--mtp-model` を渡す。`mmq_ops` は両側が変更した `mmq/mmvq.hip.cpp`、session/snapshot は衝突した SnapshotAllocator 群に対応する。利用者の決定 2026-10-08）
  - CLI・serve: `kv_cache_cli_test`、`serve_cli_test`、`inference_backend_gpu_test`、`qwen_dflash_gpu_test`、`qwen27b_target_test`
  - multi-session: `qwen27b_multisession_target_test`、`qwen_dflash_multisession_gpu_test`、`inference_backend_multisession_gpu_test`
  - キャッシュ機構: D10 の functional cache スイート、D11 のマージ前キャッシュ
- KVarN の構成は `kvarn-k4v4` + `--kv-cache-tail 512` と `kvarn-k6v5` + `--kv-cache-tail 2048`、文脈長（深度）は 256 と 131,072 に絞る。構成を指定できる実行（`kvarn_consistency.py`、`qwen_kvarn_quality_eval`、functional cache スイート）にこの絞り込みを適用し、固定マトリクスのテスト（カーネルオラクル・CLI・`qwen27b.dflash-kvarn`）はそのまま実行する。
- 品質の判定（利用者の決定 2026-10-08）: 深度 131,072 で、`g0-tail` アンカー（E_q8/E_q4）に対する p を平均 KL・中央値 KL・大逸脱率・top-1 不一致率の 4 指標で求め、`kvarn-k6v5`/2048 はすべて p ≤ 0.15、`kvarn-k4v4`/512 はすべて p ≤ 1 を合格とする。アンカーと同じ重み（`/home/betiz/models/Qwen3.8-27B-Q8_0.gguf`、SHA-256 `d17816b6…`）と同じコーパス（`~/.cache/gufo/kvarn-quality/corpus-{0,1,2}.txt`）を使う。深度 256 は記録のみとする。最初の封印は文脈長 128+L+128（L=512 で 768、L=2048 で 2304）を超えたときで、256 では封印済みタイルを読まないため、量子化経路の判定にならない。一致性スイートの 256 も同じ理由で F16 経路の確認として扱う。
- 静的確認: 解決後にリポジトリ全体で `^<<<<<<<` が 0 件であること、`RecordRoute(` の出現数がマージ前と一致すること、`load(` の各定義・呼び出しで引数の個数が揃っていることを確認する。
- モデル依存テストは環境変数（`GUFO_QWEN27B_MODEL`、`GUFO_QWEN27B_DFLASH_MODEL` など）や引数が無いとスキップ 77 になる。スキップは合格として扱わない。

### D10: upstream のキャッシュ機構は functional cache スイートで確認する

利用者の決定（2026-10-08）により、upstream のキャッシュ機構ごとに KVarN 構成で検証する。受け入れ基準は、再利用・復元後の greedy（temperature 0）トークン列がキャッシュなしの実行と一致することとする（logits のビット一致は既存の「セッション状態の整合」のスナップショット復元に限る）。

| 機構 | スイート |
| --- | --- |
| RAM プレフィックスキャッシュ | `cache-shared-prefix`、`cache-rotation` |
| 実行中プレフィックスの共有 | `cache-concurrency` |
| 学習した境界（履歴の編集） | `cache-edits`、`cache-growth` |
| 学習した境界（RAM 予算競合下の保持） | `cache-bridge`（単独の起動。`--cache-ram-bytes` を、会話チェックポイントを `SNAPSHOT_RATIO` = 16 個以下しか保持できない予算にする） |
| 安定チェックポイント・ディスク staging・再起動 | `cache`（ランナーが `--cache-disk` と staging 1 GiB を与える） |
| 封印境界をまたぐ長い履歴 | `cache-depth`（`--context 32768`） |

- 実行: `nix develop -c python3 tests/functional/run.py --output <dir> --sampling-preset qwen38 --suite ... -- build/gpu-test-nix/gufo serve llm --model "$GUFO_QWEN27B_MODEL" --speculative off --kv-cache <preset> --kv-cache-tail <L>` を、2 つの KVarN 構成のそれぞれで実行する。`cache-depth` は `--context 32768` の別の起動にする。1 回の見積もりが 1 時間を超える場合はスイートを分ける。
- `cache-bridge` は RAM 予算を専用に設定するため単独の起動にする（利用者の決定 2026-10-08、レビュー Q-2）。予算が大きすぎると圧力が再現せず不合格（unqualified）になるため、失敗時は `--cache-ram-bytes` を下げて再実行する。`cache_bridge.py` の docstring にある Flash-Next の例（2 GiB）を 27B に流用しない。ランナーは server log の `snapshot_cache_configured` から `--snapshot-capacity-bytes` を自動で渡す。
- `--speculative off` に限る。DFlash2 の KVarN 下の一致は `qwen27b.dflash-kvarn` と一致性スイートで確認する。
- スイートが KVarN 構成で失敗した場合は、同じスイートを `--kv-cache` なし（FP16）で 1 回実行し、upstream 側の問題か KVarN 側の問題かを切り分ける。

### D11: マージ前キャッシュの不再利用は参照ビルドで作って確認する

マージ前に、`d64a25c` で再ビルドした `build/gpu-test/gufo` で FP16 と `kvarn-k6v5`/2048 の `.kvc` を構成ごとの別ディレクトリに作る。2,048 トークン以上の固定プロンプトを `temperature: 0`・`reasoning_effort: "low"`・`max_tokens: 32` で送り、サーバを正常終了させる。マージ後に `build/gpu-test-nix/gufo` で同じディレクトリと同じ構成を指定して同じ要求を送り、応答の `cache_disk_hit` が偽であること、トークン列が空のディレクトリでのコールド実行と一致することを確認する。

KV 形式の不一致も同じ方法で確認する。マージ後のビルドで `kvarn-k6v5`/2048 の `.kvc` を作り、FP16・`kvarn-k4v4`/512・`kvarn-k6v5`（テール長なし）で読み込ませる。既存の `inference_backend_gpu_test` は書き手を `kvarn-k4v4`（テール長なし）とする別の組み合わせなので、spec の Scenario の値はこの手順で確認する。

### D12: 性能・品質は既存の記録を基準にする

KVarN の契約値（`docs/models/qwen3.8-27b/BENCHMARKS.md`・`QUALITY.md`）と upstream の flash-next の性能記録は再測定しない。品質は D9 の 131,072 での閾値判定だけを行い、その他はテストでマージ起因の退行がないことを確認する。再測定は本変更のスコープ外とする。

### D13: テストの FP16 エンコードは移植可能なヘルパーに置換する（追記、2026-10-08、利用者の決定）

Nix の gpu-test 構成ではホスト側のテスト TU が g++ でコンパイルされるため、clang HIP 専用の `__half_as_ushort` を使うフォークの KVarN テスト 5 ファイル（6 箇所）がビルドできない（対象ファイルは `d64a25c` から未変更で、マージ起因ではない既存の問題。他の全ターゲットは `-k 0` のビルドで成功）。`tests/models/qwen/hip/support/fp16.hpp` に `gufo::test::FloatToFp16Bits`（`_Float16` への丸め + `memcpy`、round-to-nearest）を追加し、`__half_as_ushort(__float2half_rn(x))` を置換した。nix g++ 15.3 と host hipcc（clang）で同一ビット（1.5 → 0x3E00、-2.0 → 0xC000、1.0 → 0x3C00）を確認。D8/D9 の構成・実行計画は変更しない。

## Risks / Trade-offs

- [flash-next の大規模書き換え（+1472 行）と route recording の自動マージ] → マージ後に `RecordRoute(` の出現数と各呼び出しのコンテキストをマージ前と突き合わせ、`qwen38_flash_next.route_ops`・`ap_upload`・`swift_upload`・`mmq_ops`、session/snapshot テスト、`inference_backend_gpu_test` で確認する。
- [compact snapshot の検証規則の重なり（upstream の payload 由来 `image_count` 上限と KVarN 版の検証）] → 解決時に両方の検査が残っていることをコードで確認し、`qwen_attention_kv_storage_ops_test` と `qwen_attention_kvarn_state_ops_test` で確認する。
- [同一性キーの更新でマージ前のキャッシュが無効化される] → **BREAKING** として proposal に記載し、D11 で再計算されることを確認する。ロールバック時は元の鍵（マージ前のビルド）で再利用できる。
- [`nix develop` がこの端末で失敗する] → 作業を止めて利用者に報告する。hipcc の構成へ黙って切り替えない（D8）。
- [ハーネスが古いラベルの成果物を再利用する] → 実行ごと・ビルドごとに新しいラベルを使う（D9）。
- [functional スイートの失敗が KVarN 以外の原因による] → FP16 で同じスイートを 1 回実行して切り分ける（D10）。
- [テストの期待値が upstream の意図的変更と衝突する] → 失敗を隠さず、解決ミスか upstream の仕様変更かを切り分け、後者なら根拠を添えて報告して利用者判断を仰ぐ。
- [外部モデル依存のテストがスキップされる] → スキップを合格と見なさず、実行できた範囲だけを合格として報告する（AGENTS.md）。
- [ビルドとテストの所要時間] → `ledger_run.py` で見積もりを 1 時間以内に分割し、2 倍で自動停止させる。バックグラウンド実行と単発確認で待ち、1 回のツール呼び出し内で完了を待たない。

## Migration Plan

- デプロイはない。影響はローカルブランチの履歴、ビルドディレクトリ（`build/gpu-test-nix` の新設、`build/gpu-test` の `d64a25c` での再ビルド）とキャッシュ鍵に限られる。
- push 前のロールバック: マージを取り消して `d64a25c` に戻す（コミット前は `git merge --abort`、コミット後は `git reset --hard d64a25c`）。
- push は利用者の確認後に `origin/kvarn-kv-cache-quantization` へ fast-forward で行う。push 後のロールバックは、履歴を書き換えずに `git revert -m 1 <マージコミット>` で行う。
- マージ後に作成したキャッシュはマージ前のビルドでは鍵が一致しないため再利用されない（再計算で回復）。

## Open Questions

- フォークの `main` を `upstream/main` へ進めるか（本変更では Non-Goal）。将来 upstream へ PR を出す場合の前提になる。
- upstream 側ブランチ `fedeizzo/cache-redesign` が `main` に入った場合の KVarN との統合方針（本変更の時点で変更ファイルの重複はない）。

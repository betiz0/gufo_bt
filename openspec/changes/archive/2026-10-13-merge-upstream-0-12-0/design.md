# Design

## Context

- 取り込み元: タグ `v0.12.0` = `88090af`（2026-10-11、release commit）。取り込み先: `kvarn-kv-cache-quantization`（`74b74c2`、2026-10-12）。merge-base は `f17e37b`（2026-10-07、前回のマージの取り込み元と同じ）。`upstream/main` は `f37fb01` で `v0.12.0` より 4 コミット先（#551 ほか）だが、本変更の対象はタグ `v0.12.0` とする。
- ドライランの結果（`git merge-tree --write-tree` と一時 worktree での実マージで確認済み）: 衝突は 2 ファイル 3 ハンク（`cmake/Checks.cmake` 2、`src/models/qwen38_flash_next/kernels/rocm/executor.hpp` 1）。両側が変更したファイルは 21、うち 19 は自動マージ。上流側は 51 コミット・224 ファイル・+51,636 / −4,036 行。
- 3 ハンクはいずれも「両側が同じ位置に挿入」型である。`cmake/Checks.cmake` はフォークの `qwen38_flash_next_format_policy_test` と upstream の `qwen38_flash_next_continuation_layout_test`／cache 系テストの登録、`executor.hpp` はフォークの `RouteRecord routes_`・`record_routes_` と upstream の `ReadyRows`・`xn_q8_ready_`・`mixed_q8_ready_` の同時追加である。
- upstream 0.12.0 の内容: 新しいキャッシュ層 `src/cache/*`（`gufo_cache`・`gufo_cache_hip`。`CMakeLists.txt` が `add_subdirectory(src/cache)` と `tests/cache` を追加するが、`gufo_cache` をリンクするのはベンチとテストだけで、`gufo` 本体には入らない）、serve の Messages 系（ツール・画像・`count_tokens`）、`continuation_cache.cpp` の分岐点の学習、flash-next の継続アダプタと性能改修、core（`gguf_reader`、`committed_backing`、`transfer_pool`、`staging`、`json_constraint`、`image`）、開発シェルと ccache（#484）。
- 同一性キーの構成（`QwenCompatibilityIdentity`、`src/cli/serve/inference_backend.cpp`）: `artifact_id`（GGUF の identity scheme + 指紋）、`chat_template=qwen38-reasoning-compiled-v3`、`chat_template_reference_sha256`（`QwenChatTemplate::OfficialTemplateSha256()`）、`state_abi`、`payload_layout=qwen-gfx1151-live-prefix-v3`、`checkpoint_frontier=complete-prefix-v1`、`numerics`・`rmsnorm`・`prefill_attention`・`attention_split_*`、`kv_storage`（`QwenKvStorageIdentity`）、`recurrent_storage`、`context_tokens`、`position_policy`、`rope_window_policy`。upstream 0.12.0 はこの関数・`OfficialTemplateSha256`・`payload_layout`・`checkpoint_frontier`・KVarN 本体（`src/models/qwen/hip/*`）のいずれも変更しない。したがってフォークの互換同一性キーは不変で、マージ前の `.kvc` は再利用できる（前回のマージの **BREAKING** とは逆）。
- 変更される周辺: `continuation_cache.cpp` は復元先から分岐するプロンプトを分岐点として学習し、借りた行を他の会話のライブフロンティアとして復元しない（#476・#552）。`src/core/gguf_reader.cpp` は 8/16 bit 整数メタデータ配列を読めるようになり、メタデータの解釈が変わりうる（GGUF の指紋は同じまま設定の解釈が変わる可能性があるため、鍵ではなく実行で確認する）。
- 既存ビルド: `build/gpu-test-nix`（Nix の `gpu-test`、RelWithDebInfo + assertions）。`build/gpu-test`（hipcc）は前回のマージの参照ビルドで、そのまま残す。
- 検証資産: 前回のマージで使った KVarN のカーネル・一致性・品質ハーネスと `tools/qwen27b/ledger_run.py`、upstream の `tests/functional/run.py`（0.12.0 で cache 系スイートと画像・ツールのスイートが増え、`--suite all` は cache 系を除外する）。

## Goals / Non-Goals

**Goals:**
- `v0.12.0` を取り込み、3 ハンクを両側保持で解決し、自動マージされた 19 ファイルの両側変更が意味的に共存することをビルドとテストで確認する。
- KVarN の互換同一性キーが変わらないこと（マージ前の `.kvc` が再利用できること）をコードと実行の両方で確認する。
- upstream が変更した分岐点の学習と新しい cache 層の下でも、KVarN の状態契約（KV 形式の混用なし・greedy トークン列の一致）が成り立つことを確認する。
- 公開オプション・KV 形式・テール長の意味・help 表記を変えない。

**Non-Goals:**
- `upstream/main` の `v0.12.0` 以降の 4 コミット、フォークの `main` の更新、upstream への PR。
- `src/cache` を serving へ接続すること、その API の設計。
- KVarN の品質・性能の再測定（KVarN のコードは変更しないため既存の記録を基準にする）。
- upstream の新機能（Messages のツール・画像・`count_tokens` など）のフォーク側仕様化。
- upstream 0.12.0 が追加する functional スイート（Messages のツール・画像系、`cache-workloads` 相当）の実行。これらが検証する新機能のファイルは両側変更の 21 ファイルに含まれず upstream 側のみの変更であり、serve 経路は C++ テスト（D7 の `openai_chat_test`・`http_server_test`・`websocket` 系・`vision_serving_test`）でカバーする（利用者の決定、2026-10-12）。

## Decisions

### D1: 取り込み方法はタグ `v0.12.0` のマージコミット

`git merge --no-commit --no-ff v0.12.0` で取り込み、解決と検証の後に 1 つのマージコミット（Conventional Commits の 1 行、`chore: merge upstream v0.12.0 (88090af)`）にする。`74b74c2` が先行するため fast-forward にはならない。

- 代替案: `upstream/main`（`f37fb01`）を取り込む — `v0.12.0` より 4 コミット先の未リリース変更（#551 など）を含み、利用者の指定（0.12.0）を超える。タグに合わせる。
- 代替案: rebase・cherry-pick — 同じファイル群で衝突を繰り返し解くことになり、`version.txt` と `CHANGELOG.md` のリリースコミットが積み上がる。マージ 1 回の方が解決と検証の記録が集中する。

### D2: `cmake/Checks.cmake` は両側の登録を統合する

`gufo_pr_targets` にフォークの `qwen38_flash_next_format_policy_test` と upstream の `qwen38_flash_next_continuation_layout_test` と cache 系 15 テスト（`cache_adapter_test`・`cache_adapter_lifecycle_test`・`cache_ledger_test`・`cache_checkpoint_test`・`cache_prefix_index_test`・`cache_slot_test`・`cache_idle_spill_test`・`cache_disk_test`・`cache_disk_publication_test`・`cache_streaming_test`・`cache_tiered_test`・`cache_retention_test`・`cache_boundary_test`・`cache_workloads_test`・`cache_disk_faults_test`）を残し、`REMOVE_ITEM` と `APPEND` の両側の項目（`\\.format_policy`・`\\.continuation_layout`・`check_docs_test`・`cache_workloads_test`・`cache_disk_faults_test`・`cache_boundary_test`）を統合する。`check-pr` の `DEPENDS` には upstream の `cache_disk_faults` を残す。

- 理由: どちらのテストも既存の呼び出し元（PR スイートの正規表現）を持ち、片方を落とすとホスト側の検証が静かに減る。
- 確認: `ctest -N` の一覧に両方の名前があり、`cmake --build --preset pr`（またはホスト側の `check-pr`）が `cache_disk_faults` を含めて解決できること。

### D3: `executor.hpp` は両側のメンバーを残す

`RouteRecord routes_`・`record_routes_`（フォーク、テスト用の経路記録）と `ReadyRows`・`xn_q8_ready_`・`mixed_q8_ready_`（upstream、Q8_1 行の producer 書き込み）を同じクラスに残す。どちらも可変メンバーの追加で、初期化子と用途が独立しているため順序は意味を持たない。

### D4: 自動マージ 19 ファイルは「意味が重なる箇所」を重点的に確認する

- flash-next の量子化・カーネル経路: フォークの per-tensor AP-Q4_K_XL 形式と route recording、upstream の producer 書き込み Q8_1 行・高速 decode/prefill（`kernels.hip.cpp`・`mmq/{mmvq.hip.cpp,qfn_mmq.h,qfn_mmq.hip.cpp}`・`executor.cpp`・`engine.*`）。
- serve: フォークの KVarN 同一性・`kv_cache_format()`・`session_states()` と upstream の `count_chat_tokens`（`inference_backend.{hpp,cpp}`）。
- テスト: `vision_serving_test.cpp`、`qwen38_flash_next/{hc_mix_ops,projection_ops,routed_wmma_ops}_test.cpp`、`tests/models/qwen38_flash_next/CMakeLists.txt`（テスト名の重複がないこと）。
- 文書・成果物: `docs/models/qwen3.8-flash-next/{EXPERIMENTS.md,QUALITY.md}` の両側の行、`artifacts/model-identities.json`（JSON として妥当で、両側の節が残っていること）。

### D5: 互換同一性キー不変の確認はコードと実行の両方で行う

コード: `git diff f17e37b..v0.12.0 -- src/cli/serve/inference_backend.cpp src/models/qwen/chat_template.{cpp,hpp}` に `QwenCompatibilityIdentity`・`QwenStateAbiName`・`QwenKvStorageIdentity`・`kQwenPersistentPayloadVersion`・`OfficialTemplateSha256`・`qwen38-reasoning-compiled-v3` の変更がなく、`src/models/qwen/hip/*`（KVarN 本体）が上流で変更されていないことを確認する。

実行（D9）: マージ前のビルドで作った `.kvc` をマージ後のビルドで読み、再利用されることと、トークン列がコールド実行と一致することを確認する。GGUF メタデータの解釈が変わりうる（#532）ため、鍵が同じでも意味が変わる可能性はこの実行確認で検出する。

### D6: ビルドの構成

- `build/gpu-test-nix` を再利用する: `nix develop -c cmake --preset gpu-test -B build/gpu-test-nix`、`nix develop -c cmake --build build/gpu-test-nix --parallel 8`。`src/cache` と `tests/cache` が加わるため再構成が必要で、`gufo_cache`・`gufo_cache_hip`・cache 系テストが新たにビルド対象になる。
- `nix develop` 自体が失敗した場合は作業を止めて利用者に報告し、hipcc の構成へ切り替えない。flakes が無効な端末では `nix --extra-experimental-features 'nix-command flakes' develop` とする。
- upstream の #484 により、`ccache` が PATH にある環境では C/C++ が ccache 経由でコンパイルされ、`nix develop` の shellHook が `CCACHE_BASEDIR`・`CCACHE_COMPILERCHECK=content`・`CCACHE_MAXSIZE=2G`・`CCACHE_NOHASHDIR` を設定する。構成ログの `gufo: compiling C and CXX through ...` を確認し、ビルド所要時間は ccache の状態に依存するため過去の記録と比較しない。HIP カーネルはキャッシュされない。
- 600 秒を超える見込みのビルドは `nohup ... > /tmp/merge-0-12-0-build.log 2>&1 &` で起動し、`ps -p` とログ末尾の単発確認で追う（ポーリングループは使わない）。
- Nix のビルドは git の追跡ファイルを含む。マージで変更するファイルはすべて追跡済みで、未追跡の変更アーティファクトはビルドに影響しない。タスク所有のパス（`openspec/changes/merge-upstream-0-12-0/` と解決した実装ファイル）だけをステージし、無関係な作業ツリーの変更を巻き込まない。
- `EXCLUDE_FROM_ALL` のテスト実行ファイル（`qwen_kvarn_quality_eval`、`qwen38_flash_next_model_tests`、`qwen38_flash_next_{dense_gemm,attn}_bench`、`continuation_probe`、`cache_*_bench`）は、テストで使うものをターゲット指定でビルドする。

### D7: テストの実行規則と対象

- VRAM と GPU を使うテストは並列実行しない（`ctest` に `-j`／`--parallel` を付けない。GPU を使うサーバやベンチと同時に走らせない）。CPU のみのテストは並列でよい。
- 数分を超える実行は `tools/qwen27b/ledger_run.py run` で 1 本ずつ前面起動し、`--id`・`--task`・`--purpose`・`--estimate-seconds`・`--estimate-basis`・`--pass-criteria` を指定する（アーカイブ済み変更の D22・D23）。見積もりは 1 時間以内に分割し、2 倍で自動停止させる。完了は `status`／`wait` かログの単発確認で待つ。
- ctest は `nix develop -c ctest --test-dir build/gpu-test-nix ...` を `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu` の上書き付きで実行する（ホスト glibc と store glibc の混在の既知回避、前回のマージの D9 追記）。
- 対象（ラベル `multi-session` は別に実行する。`gpu-full` の除外に合わせて `-LE multi-session` を付ける）:
  - KVarN: `qwen_kvarn_(quant|fwht)_cpu_test`、`qwen_attention_kvarn_(seal|read|prefill|prefill_read|state)_ops_test`、`qwen_attention_kv_storage_ops_test`、`qwen_graph_ops_test`、`qwen_kvarn_error_test`、`qwen27b\.dflash-kvarn`
  - 新しい cache 層とホスト側チェック: `cache_(adapter|adapter_lifecycle|ledger|checkpoint|prefix_index|slot|idle_spill|disk|disk_publication|streaming|tiered|retention|boundary|workloads|disk_faults)_test`、`check_docs_test`、`check_dependencies_test`、`kernel_resources_parser_test`
  - flash-next（自動マージの影響面）: `qwen38_flash_next\.(config|ngram|format_policy|cpu_ops|projection_ops|attention_ops|mmq_ops|route_ops|ap_upload|swift_upload|upload_gate|embed_ops|mixer_route)`、`qwen38_flash_next\.continuation_layout`、`qwen38_flash_next_cli_load_error`、`qwen38_flash_next_model_tests` の `session_test`・`snapshot_test`（`--model`・`--mtp-model` を渡す）
  - CLI・serve: `kv_cache_cli_test`、`serve_cli_test`、`inference_backend_gpu_test`、`qwen_dflash_gpu_test`、`qwen27b_target_test`、`qwen_chat_template_test`、`qwen_tokenizer_test`、`vision_serving_test`、`openai_chat_test`、`http_server_test`、`websocket` 系
  - multi-session: `-L multi-session` の 3 件
  - functional cache スイート（D8）とマージ前後のキャッシュ（D9）
- KVarN の構成は前回と同じ `kvarn-k4v4` + `--kv-cache-tail 512` と `kvarn-k6v5` + `--kv-cache-tail 2048`、深度は 256 と 131,072 に絞る。KVarN のコードは変更しないため、一致性スイートと品質評価は退行確認として同じ条件で実行し、品質の p の判定は記録済みの値を基準にする（再判定しない）。
- モデル依存のテストは環境変数や引数が無いとスキップ 77 になる。スキップは合格として扱わない。

### D8: upstream のキャッシュ機構は functional cache スイートで再確認する

分岐点の学習が変わったため、前回と同じ受け入れ基準（再利用・復元の後の greedy トークン列がキャッシュなしの実行と一致）で再実行する。機構とスイートの対応は前回の D10 と同じとし、`--cache-disk` 系は `run.py` が所有する。

| 機構 | スイート |
| --- | --- |
| RAM プレフィックスキャッシュ | `cache-shared-prefix`、`cache-rotation` |
| 実行中プレフィックスの共有 | `cache-concurrency` |
| 学習した境界・分岐点（履歴の編集） | `cache-edits`、`cache-growth` |
| 学習した境界（RAM 予算競合下の保持） | `cache-bridge`（`--cache-ram-bytes` を単独起動で指定） |
| 安定チェックポイント・ディスク staging・再起動 | `cache` |
| 封印境界をまたぐ長い履歴 | `cache-depth`（`--context 32768`） |

- 実行: `nix develop -c python3 tests/functional/run.py --output <dir> --sampling-preset qwen38 --suite ... -- build/gpu-test-nix/gufo serve llm --model "$GUFO_QWEN27B_MODEL" --speculative off --kv-cache <preset> --kv-cache-tail <L>` を 2 つの KVarN 構成で実行する。1 回の見積もりが 1 時間を超える場合はスイートを分ける。
- 失敗時は同じスイートを `--kv-cache` なし（FP16）で 1 回実行し、upstream 側か KVarN 側かを切り分ける。`run.py` は 0.12.0 でスイート一覧と引数が変わっているため、`--help` で対象スイート名を確認してから実行する。一覧に現れる本表以外の新規スイート（Messages のツール・画像系、`cache-workloads` 相当）は実行しない（Non-Goals、利用者の決定 2026-10-12）。
- 実行ごとに新しいラベル（`merge-0-12-0-` で始める）を使い、前回のラベルの成果物を再利用しない。

### D9: マージ前の `.kvc` が再利用されることを確認する（鍵は不変）

マージ前に `HEAD`（`74b74c2`）で `build/gpu-test-nix` の `gufo` をビルドし、`gufo.premerge` として複製して参照ビルドにする。この参照ビルドで `--kv-cache kvarn-k6v5 --kv-cache-tail 2048 --cache-disk ~/.cache/gufo/merge-upstream-0-12-0/premerge-kvarn-k6v5-2048` と FP16 のそれぞれについて `.kvc` を作る（KVarN のスナップショットは既定 staging 1 GiB を超えるため `--cache-disk-staging-bytes 17179869184` を指定する。共通プロンプトは前回と同じ `records/prompt.txt` を使い、`temperature 0`・`reasoning_effort: "low"`・`max_tokens: 32`）。

マージ後、同じディレクトリと同じ構成で同じ要求を送り、`cache_disk_hit` が真で、応答のプロンプト長とトークン列が空のディレクトリでのコールド実行と一致することを確認する。FP16 と KVarN の両方で行う。再利用されない場合は、identity の項目がどこかで変わっていないかを `git diff` で特定し、意図しない変更なら解決をやり直す（意図した変更なら **BREAKING** として利用者に報告する）。

### D10: 品質・性能は再測定しない

KVarN のカーネル・量子化・読み出しと同一性は変更しないため、`docs/models/qwen3.8-27b/{BENCHMARKS,QUALITY}.md` の記録はそのまま有効とする。退行の確認は既存のカーネルテスト・一致性スイート・`inference_backend_gpu_test`・functional cache スイートで行う。upstream の flash-next の性能改修（#485・#518）はフォークの記録（`docs/models/qwen3.8-flash-next/*`）に影響しうるが、再測定は本変更のスコープ外とする。

### D11: バージョンと CHANGELOG は upstream の生成物を採る

`version.txt` を `0.12.0` にし、`CHANGELOG.md` に 0.9.1〜0.12.0 の節を取り込む（release-please の生成物）。フォーク側で書き換えたり、独自の節を足したりしない（リリースとバージョンの更新は upstream の release workflow の責務、`docs/RELEASING.md`）。

### D12: 公開オプションの契約は変更しないことを実行で確認する

`serve llm --help` に `--kv-cache`・`--kv-cache-tail`・`--log-level`・`-v/--verbose`・`--log-progress`・`--trace`・`--max-pending` が出て別名がないこと、不正な KV 指定（`--kv-cache kvarn-k9v9`・`--kv-cache-tail 255`・`--kv-cache fp16 --kv-cache-tail 1024`）がモデル読み込み前に非ゼロで終了することを前回と同じ手順で確認する（`serve_cli_test`・`kv_cache_cli_test` と CLI 実行の両方）。

## Risks / Trade-offs

- [flash-next の両側変更（フォーク +1,190 行前後、upstream +1,700 行前後）が同じ量子化・カーネル経路で自動マージされる] → D4 の重点確認を行い、`qwen38_flash_next\.{mmq_ops,projection_ops,routed_wmma_ops,hc_mix_ops,attention_ops,route_ops,ap_upload,swift_upload}` と session/snapshot テスト、`inference_backend_gpu_test` で確認する。失敗は解決ミスか upstream の意図的変更かを切り分けて報告する。
- [KVarN の互換同一性キーが意図せず変わる] → D5 の静的確認と D9 の再利用確認で検出する。変わっていた場合は `git diff` で原因を特定し、意図しない変更なら解決をやり直す。
- [GGUF メタデータの解釈変更（#532）で、指紋が同じでもモデル設定の解釈が変わる] → D9 のトークン列一致（キャッシュ再利用時とコールド実行の一致）と KVarN・27B のテストで確認する。
- [functional cache スイートが KVarN 構成で失敗する] → FP16 で同じスイートを 1 回実行して切り分ける（D8）。upstream の分岐点の学習の変更が原因なら、その旨を根拠付きで報告して利用者の判断を仰ぐ。
- [`nix develop` がこの端末で失敗する、または ccache の状態でビルド時間が跳ねる] → 失敗時は作業を止めて報告する（hipcc の構成へ切り替えない）。時間は ccache のヒット率に依存するため、過去のビルド時間と比較しない。
- [新しいホスト側チェック（`check_docs_test`・`check_dependencies_test`・`cache_boundary_test`）がフォークの文書・依存で失敗する] → 失敗内容を確認し、フォーク側の記述（リンク・依存表・`src/cache` の include 境界）を直す。upstream の規約に合わせるのは取り込みの一部とする。
- [外部モデル依存のテストがスキップされる] → スキップを合格と見なさず、実行できた範囲だけを合格として報告する（AGENTS.md）。
- [所要時間] → `ledger_run.py` で見積もりを 1 時間以内に分割し、2 倍で自動停止させる。ビルドとテストはバックグラウンド起動と単発確認で待つ。

## Migration Plan

- デプロイはない。影響はローカルブランチの履歴、`build/gpu-test-nix` の内容、`~/.cache/gufo/merge-upstream-0-12-0/` のキャッシュに限られる。
- ロールバック: コミット前は `git merge --abort`、コミット後は `git reset --hard 74b74c2`。push 後のロールバックは履歴を書き換えず `git revert -m 1 <マージコミット>`。
- push は利用者の確認後に `origin/kvarn-kv-cache-quantization` へ fast-forward で行う。
- マージ前後の `.kvc` は互換同一性キーが同じため、双方向に再利用できる。互換性が崩れた場合（D9 で再利用されない場合）は再計算で回復し、移行作業は発生しない。

## Open Questions

- `upstream/main` の `v0.12.0` 以降の 4 コミット（#551 ほか）をいつ取り込むか（本変更では Non-Goal）。
- `src/cache` を serving へ接続する際に、KVarN の KV 形式と state abi をどう identity へ渡すか（本変更の spec が要求する項目の実現方法。後続の変更で判断する）。
- フォークの `main` を `upstream` へ進めるか（前回のマージからの継続課題。本変更では Non-Goal）。

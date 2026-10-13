# Proposal

## Why

フォークの `kvarn-kv-cache-quantization` ブランチは upstream（`gufo-org/gufo`）のリリース `v0.12.0`（`88090af`、2026-10-11）に対して 51 コミット遅れている（merge-base = `f17e37b`、2026-10-07。フォークは 45 コミット先行）。遅れ分には `/v1/messages` のツール呼び出しとストリーミング、画像ブロック、`count_tokens`、WebSocket とヘッダの修正、GGUF の 8/16 bit 整数メタデータ、flash-next の継続アダプタと性能改修、そして新しいキャッシュ層 `src/cache/*` が含まれる（224 ファイル、+51,636 / −4,036 行）。リリース 4 本分（0.9.1・0.10.0・0.11.0・0.12.0）が溜まっており、衝突面が 2 ファイル 3 ハンクと小さいうちに取り込む。

## What Changes

- `v0.12.0`（`88090af`）を `kvarn-kv-cache-quantization` へマージし、1 つのマージコミットにする。衝突の解決は両側の変更を保持することを原則とし、フォーク側の KV 形式・テール長の意味と公開オプションの契約を変えない。
- 衝突は 2 ファイル 3 ハンクで、いずれも「両側が同じ位置に挿入」型である。どちらかを捨てる必要はない。
  - `cmake/Checks.cmake`（2 ハンク）: フォークの `qwen38_flash_next_format_policy_test` と upstream の `qwen38_flash_next_continuation_layout_test` を両方登録し、upstream の cache 系テスト（`cache_adapter_test`・`cache_adapter_lifecycle_test`・`cache_ledger_test`・`cache_checkpoint_test`・`cache_prefix_index_test`・`cache_slot_test`・`cache_idle_spill_test`・`cache_disk_test`・`cache_disk_publication_test`・`cache_streaming_test`・`cache_tiered_test`・`cache_retention_test`・`cache_boundary_test`・`cache_workloads_test`・`cache_disk_faults_test`）と `check_docs_test`、`DEPENDS` の `cache_disk_faults` を残す。
  - `src/models/qwen38_flash_next/kernels/rocm/executor.hpp`（1 ハンク）: フォークの `RouteRecord routes_`・`record_routes_` と upstream の `ReadyRows`・`xn_q8_ready_`・`mixed_q8_ready_` を両方残す（同じクラスの別メンバーで、順序は意味を持たない）。
- 自動マージされる 19 ファイルの両側変更が意味的に共存することを確認する。重点は flash-next の量子化・カーネル経路（`kernels.hip.cpp`、`mmq/{mmvq.hip.cpp,qfn_mmq.h,qfn_mmq.hip.cpp}`、`executor.cpp`、`engine.cpp`）と、`src/cli/serve/inference_backend.cpp`（フォークの KVarN 同一性と upstream の `count_chat_tokens`）、テスト、文書、`docs/models/qwen3.8-flash-next/artifacts/model-identities.json`。
- フォークの KVarN 契約はこのマージで変わらない。upstream は 27B の KVarN 経路（`src/models/qwen/hip/*`）、ペイロード、量子化演算、状態 ABI、`QwenCompatibilityIdentity` を変更しない。したがって同一性キーは変わらず、マージ前に作成した `.kvc` ディスクキャッシュは引き続き再利用できる（前回のマージの **BREAKING** とは異なる）ことを実行で確認する。
- upstream が変更した分岐点の学習（`src/cli/serve/continuation_cache.cpp`。復元先から分岐するプロンプトを学習し、借りた行を他の会話のライブフロンティアとして復元しない）と、新しい cache 層（`src/cache/*`。0.12.0 では `gufo` にリンクされず、テストとベンチからのみ使われる）の下でも、KVarN の状態契約（KV 形式の異なる状態を再利用しない・再利用と復元の後の greedy トークン列がキャッシュなしの実行と一致する）が成り立つことを、functional cache スイートを KVarN 構成で実行して確認する。
- KVarN の状態を保持・再利用するキャッシュ機構の互換同一性が KV 形式（プリセットとテール長）と state abi を含むことを spec に明記し、機構の列挙に upstream の分岐点の学習・checkpoint の選択と退避・有界の転送・クラッシュ耐性のあるディスク publication と起動時 index・新しい cache アダプタ層を加える（spec MODIFIED）。
- ビルドは既存の Nix の gpu-test 構成（`build/gpu-test-nix`）を再利用する。upstream の #484 により、C/C++ は `ccache` が PATH にあれば経由してコンパイルされ、開発シェルに `gdb`・`git`・`ripgrep`・`radeontop`・`ccache` が加わる。HIP カーネルはキャッシュされない。
- バージョンは `version.txt` が `0.9.0` → `0.12.0` になり、`CHANGELOG.md` に 0.9.1・0.10.0・0.11.0・0.12.0 の節が入る（upstream の release workflow の生成物をそのまま取り込み、フォーク側で書き換えない）。
- 公開オプション・KV 形式・テール長の意味を変えない。`--kv-cache`・`--kv-cache-tail` の値・検証・終了コード・help 表記も変えず、サーバの識別子（`--log-level`、`-v/--verbose`、`--log-progress`、`--trace`、`--max-pending`）も upstream の正規名のままとする。
- スコープ外: `upstream/main` の `v0.12.0` 以降のコミット（`f37fb01` までの 4 件）、フォークの `main` の更新、KVarN の品質・性能の再測定、upstream への PR、upstream が 0.12.0 で追加した functional スイート（Messages のツール・画像系、`cache-workloads` 相当）の実行（利用者の決定 2026-10-12、design Non-Goals）。

## Capabilities

### New Capabilities

なし。この変更は upstream のリリースの取り込みと既存機能の維持であり、新しい機能をフォーク側で定義しない。

### Modified Capabilities

- `qwen/kv-cache-quantization`: 「キャッシュ機構との状態契約」を MODIFIED。機構の列挙に分岐点の学習、checkpoint の選択と退避、有界の転送、クラッシュ耐性のあるディスク publication と起動時 index、upstream の cache アダプタ層を加え、KVarN の状態を保持・再利用する機構の互換同一性が KV 形式（プリセットとテール長）と state abi を含まなければならないこと、ペイロード・量子化演算・読み出しの演算を変更しない取り込みでは同一性キーが変わらずマージ前の `.kvc` ディスクキャッシュを再利用できることを要求する。

`kv-cache-selection` は差分を作らない。upstream 0.12.0 は `serve llm` のオプション定義（`src/cli/serve/serve.cpp`）と `QwenCompatibilityIdentity` を変更しないため、KV 形式の選択・報告・サーバオプションとの共存に関する要件は現行のまま成り立ち、このマージでも検証だけを行う。

## Impact

- マージ対象: upstream `v0.12.0`（`88090af`）51 コミット、224 ファイル、+51,636 / −4,036 行。衝突 2 ファイル 3 ハンク、両側が変更するファイルは 21（うち 19 は自動マージ）。
- 新規: `src/cache/*`（`gufo_cache`・`gufo_cache_hip`、`cache_*_bench`）、`tests/cache/*`、`docs/cache-redesign/*`、`src/models/qwen38_flash_next/continuation_{adapter,layout,hooks}.*`、`tests/functional/{test_cache_workloads.py,tool_images.py,tool_streaming.py}`、`tests/tools/check_{docs,dependencies}_test.py`、`src/core/hip/{committed_backing,transfer_pool,staging}.*`。
- 変更: serve（`openai_chat.cpp` +1,005、`http_server.cpp`、`websocket.cpp`、`inference_backend.cpp`、`text_generation_backend.hpp`、`text_model_runner.hpp`、`continuation_cache.cpp`）、core（`gguf_reader.cpp`、`json_constraint.cpp`、`image.cpp`）、flash-next（`engine.*`、`kernels/rocm/*`、`mmq/*`）、`CMakeLists.txt`、`cmake/Checks.cmake`、`flake.nix`、文書（`docs/SERVER.md`、`docs/KV-CACHE.md`、`docs/DEVELOPMENT.md`、`README.md`、`RELEASING` 関連）、`version.txt`、`CHANGELOG.md`、`NOTICE`、`THIRD_PARTY_NOTICES.md`。
- 仕様: `openspec/specs/qwen/kv-cache-quantization/spec.md`（MODIFIED 1 件）。
- 利用者への影響: 公開オプションと KVarN の契約は不変で、`.kvc` ディスクキャッシュも再利用できる（**BREAKING** なし）。upstream の新機能（`/v1/messages` のツール・画像・`count_tokens`、WebSocket のアイドル扱い、エラー文言、GGUF のメタデータ）はそのまま入る。
- ビルドと成果物: `build/gpu-test-nix`（Nix + ccache）を再利用する。マージ前の参照ビルドを `HEAD` で作成し、マージ前の `.kvc` を `~/.cache/gufo/merge-upstream-0-12-0/` 配下に作る。実行の記録は `artifacts/test-ledger.jsonl`（`tools/qwen27b/ledger_run.py`）に残す。
- 依存: 新規の外部依存なし（`gufo_cache` は `Threads` のみ）。リモート `upstream`（`git@github.com:gufo-org/gufo.git`）は登録済みで、タグ `v0.12.0` は fetch 済み。
- リスク: flash-next は前回のマージでフォークも大きく変更しており、upstream 0.12.0 も同じファイル群（カーネル・mmq・executor・engine）を変更するため、自動マージの意味的な共存をビルドとテストで確認する必要がある。functional cache スイートの KVarN での再実行は GPU 時間を要する。`ccache` によりビルド時間の比較条件が変わるため、記録済みのビルド所要時間とは比較しない。

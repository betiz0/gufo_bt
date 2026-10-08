# Proposal

## Why

Qwen3.8 27B の 262,144 トークン容量では FP16 KV が約 16 GiB を占め、セッション並列度とディスクキャッシュ（staging 上限）を直接制限する。長文脈 decode の速度低下（深度 0 の 12.37 tok/s → 131,072 の 8.52 tok/s、Q4 AR 実測、`docs/models/qwen3.8-27b/BENCHMARKS.md`）は KV 読み出し帯域が主因と推定される（実効帯域 約 235 GB/s・ハードウェア上限 約 240 GB/s という値はリポジトリ外の推定値で、本リポジトリの文書では未検証）。KV を量子化すれば、この読み出し量とスナップショットサイズの両方を同時に縮められる。G1 の 1 回目の測定（Q8_0、`docs/models/qwen3.8-27b/G1-REPORT.md`）では、FP16 KV の深度 128K のトークンあたり時間が深度 0 より 35.8 ms 長く、1 トークンで読む KV 8.59 GB を約 240 GB/s で読む時間に相当する。ただし、KV をキャッシュに常駐させて測った FP16 の注意の計算だけの時間も 128K で約 34 ms あり、計算と帯域がほぼ同時に限界に近い。そのため今の注意カーネルの構造では、KV の読み出し量を減らしても長文脈は速くならず、KVarN は展開の計算が加わる分だけ FP16 KV より遅くなる（キーごと処理の採用後で `kvarn-k4v4` の 128K が FP16 比 0.843）。本変更の目的は KV メモリの削減（セッション並列度、スナップショットとディスクキャッシュの縮小）とし、長文脈の速度の低下は受け入れて文書化する（design D19）。

KVarN 方式（Hadamard 回転 + 二軸分散正規化 + 非対称 RTN、校正不要）は、外部の BeeLlama の測定で q8_0 相当の品質を BF16 比 30〜36% のサイズで達成したと報告されている（本リポジトリでは未検証。フェーズ 0 で gufo 上の品質を実測して判断する）。gufo の現行 KV は `QwenKvCacheStorage::{kFp32,kFp16}` の 2 種のみで量子化ストレージを持たないため、本変更で Qwen3.8 27B 専用の KV 量子化を追加する。

## What Changes

- `QwenKvCacheStorage` に KVarN 系ストレージを追加し、K と V のビット幅を独立に保持する（`kFp32`/`kFp16` のバイト列と挙動は不変）。ポリシーの `Fingerprint()` に KV 形式を含める。FP16 の fingerprint・互換同一性キー・スナップショットのバイト列は変更しない。
- KV を 3 領域で構成する: sink（先頭 128 トークン、F16）、本体（封印済み 128 トークン × 256 チャネルのタイル）、テール（F16 のリング。容量は L+128+W 行で、W は最大検証行数と最大巻き戻し幅の大きい方）。テール長 L は `--kv-cache-tail` で 256〜2048 を指定でき、未指定のときはテールトークンを追加せず、構造上の最小値 W とする。
- 各クエリがタイルを量子化値と F16 のどちらで読むかを、クエリの絶対位置だけで決める規則（位置で決まるビュー）を定め、AR・検証・prefill・巻き戻し後の再計算で同一のビューを使う。
- 回転（正規化 FWHT、head_dim 256）を RoPE + KV 書き込みの融合カーネルに追加し、注意出力側で逆回転する。
- 封印演算子を追加する: 回転済み F16 タイル → 二軸分散正規化（反復 16、縮約順序固定）→ 非対称 RTN → タイル格納。封印結果は 128 トークンの KV 値だけの関数として決まる。非有限値はデバイス側で検出し、該当リクエストを失敗させる。
- decode（スカラー / split-K / グラフ版）、検証（因果 decode）、prefill（tile / WMMA / batched）の読み出し経路で 3 領域を単一のオンライン softmax に統合する。テスト・ベンチ専用の GEMM 経路は KVarN 選択時に明示的に拒否する。
- プリセット `kvarn-k4v4`（メモリ優先、適合テール長 256）、`kvarn-k5v4`（中間、テールなしでメモリ優先基準に適合）、品質優先プリセット `kvarn-k6v5`（K 6 bit・V 5 bit、適合テール長 2048）を提供する。品質優先のビット幅、各プリセットの適合テール長（テール長 未指定・256・1024・2048 のうち品質基準を全深度で満たす最小の L。品質基準は q8→q4 内分位置で判定し、品質優先は p ≤ 0.15、メモリ優先は p ≤ 1、design D17）、スケール数値型（FP16）はフェーズ 0 の G0 で確定した。
- 正規ロングオプション `--kv-cache`（`fp16` / プリセット名）を `serve llm`・`chat`・`prompt`・`bench` に追加する。Qwen3.8 27B の既定値は `fp16` のままとする。Qwen3.8 27B 以外のモデルで `--kv-cache` を明示指定した場合（`fp16` を含む）は起動時エラーとする。オプションの登録と実行ポリシーへの受け渡しはフェーズ 0 で先に実装する。
- 正規ロングオプション `--kv-cache-tail <N>`（256〜2048 の整数）を同じ 4 コマンドに追加し、KVarN のテール長を指定できるようにする。KVarN プリセットを選んでいないとき、Qwen3.8 27B 以外のモデル、範囲外の値では起動時エラーとする。未指定ではテールトークンを追加しない。上限 2048 は `kvarn-k4v4` の総量 4.6 bpv 基準を満たす L の上限（2432）から決めた。フェーズ 0 で配線し、G0 の測定に使う。
- コンパクト／永続スナップショットに KVarN 用の版を追加し、KVarN 種別・K/V ビット幅・テール長・各領域の行数・スケール型を記録する（既存版の読み込み互換と FP16 の書き出しバイト列は維持）。KVarN の互換同一性キーには KV 形式（プリセットとテール長）を含める。
- `EstimateMemoryUsage` とセッション容量計算に KVarN のサイズ式を追加し、KV 形式・テール長・実効 bpv・セッションあたり KV 容量を起動ログと `gufo bench` に報告する。
- テール長 未指定・256・1024・2048 の KL 指標、総量 bpv、セッションあたり KV 容量を `docs/models/qwen3.8-27b/README.md` に比較表として掲載する。
- 数時間かかる検証（G0 の品質評価、本測定、一致性スイート、ベンチマーク）は、ツール呼び出しのタイムアウトで失われない実行方法（切り離したプロセス、単位ごとの成果物と再開、完了マーカーの待機）で行う（design D16）。2026-10-08 以後は、1 本のテストの見積もりを 1 時間以内とし、見積もりの 2 倍で自動停止する。長時間の本測定は縮小し、既存の G0・G1・G2・一致性スイートの結果を流用する（design D23）。
- 利用開始は、品質ハーネスがループせずに進むことを T3 の再実行で確かめ、主構成 `kvarn-k5v4` テール長 2048 の短い品質確認（深度 4K・32K）を報告した後に、利用者が判断する（design D23）。
- 品質ゲートに、32K に置いた事実を参照させる長距離参照の短い確認（記録のみ）を加える（教師強制の統計量が遠距離参照の劣化を過小に拾うため、design D17）。デコード型タスクと長距離参照タスクの統計的な非劣性評価は後続の変更で行う（design D23）。Qwen3.8 27B のチャット形式の生成を使うテストでは `reasoning_effort: low` を指定する。
- 品質・性能・メモリ・保守の受入基準とその検証（T-01〜T-09、定義は design.md）をフェーズ 0〜4 のゲート（G0〜G4）に対応づける。
- 性能契約は Q8_0 の重みで測る。深度 0〜4K の AR tg、深度 0 の DFlash2 tg、C8 AR、起動時間は FP16 比の下限で判定し、深度 32K 以上の AR・DFlash2 の tg と全深度の pp は、最初に測る判定で記録した FP16 比を基準値として退行で判定する。プリセット・テール長・深度ごとの FP16 比を README と `docs/CLI.md` に掲載し、長文脈では FP16 KV より遅いことを明記する。性能契約が未達になった時点で作業を止め、利用者に判断を仰ぐ（design D18・D19）。最終の性能測定（T-09）は、G1・G2 の値を流用し、主構成の AR tg の測り直しと、未測定の C8・起動時間・p99 / p50・ピーク割当だけを測る（design D23）。

破壊的変更なし: Qwen3.8 27B の既定は `fp16` のまま、既存 KV バイト列・FP16 の互換同一性キー・旧スナップショット版・既存の `.kvc` ディスクキャッシュを維持する。他モデルでは `--kv-cache` と `--kv-cache-tail` は新規オプションであり、未指定時の挙動は変わらない。

## Capabilities

### New Capabilities

- `qwen/kv-cache-quantization`: Qwen3.8 27B フル注意層の KVarN KV ストレージ（sink / 封印タイル / テール）、位置で決まるビュー、FWHT 回転と逆回転、二軸分散正規化 + 非対称 RTN の封印演算子、全注意読み出し経路の 3 領域統合、rollback・キャンセル・継続・フォーク・スナップショットの状態契約、KV 容量見積と実効 bpv 報告、品質・性能契約。
- `kv-cache-selection`: KV 形式を選ぶ正規ロングオプション `--kv-cache` とテール長を選ぶ `--kv-cache-tail`、Qwen3.8 27B での既定値 `fp16`（テールトークンなし）、他モデルでの明示指定エラー、選択結果の起動ログ・`gufo bench` への報告。

### Modified Capabilities

なし（`openspec/specs/` は空で、既存仕様は存在しない）。

## Impact

- ストレージ／ポリシー: `src/models/qwen/hip/execution_policy.hpp`（`QwenKvCacheStorage`、`Fingerprint()`）、`src/models/qwen/hip/arena.cpp`（KV 確保、コンパクトスナップショットの形式・版・検証）、`src/models/qwen/hip/ssm_replay.cpp`（`SaveState`/`RestoreState` の前提確認）、`src/models/qwen/hip/executor.{hpp,cpp}`（`EstimateMemoryUsage`）、`src/models/qwen/hip/{decode_step,prefill_chunk,batched_decode}.cpp`（封印と読み出しの呼び出し）。
- カーネル: `src/models/qwen/hip/kernels/{fused,prefill_rope}.hip`（回転とテール書き込みの融合）、`attention_decode.hip`、`attention_decode_graph.hip`、`attention_tile.hip`、`attention_wmma.hip`、`attention_batched.hip`（GEMM 経路の拒否を含む）、`LaunchCausalDecodeAttention`（検証行）。新規 `kernels/kvarn_*.hip` と `ops/kvarn.hpp`（Qwen 専用、他モデルと非共有）。
- CPU 参照実装: 新規 `src/models/qwen/` 配下の KVarN オラクル（Qwen 専用。`src/core/` には置かない）。
- 方針判定: `src/models/qwen/hip/detail/attention_policy.hpp`（split-K 境界をタイル境界へ整列、`IsTiledAttentionSupported` の対応判定）。
- CLI・状態: `src/cli/serve/*`（オプション解析、`QwenCompatibilityIdentity`、永続スナップショット、`continuation_disk_store` の同一性キー、非対応モデルの起動時エラーと KV 形式の報告）、`src/cli/prompt/prompt.cpp`（`RegisterTextOptions`）、`src/cli/bench/bench.cpp`（`RegisterBenchOptions`、報告）。
- ビルド: `CMakePresets.json` と `CMakeLists.txt`（フェーズ 0 の疑似量子化を `gpu-test` プリセットだけで有効にするコンパイル定義。フェーズ 1 で削除）。
- テスト・ツール: `tests/models/qwen/hip/attention/*`、`tests/models/qwen/hip/execution_policy_test.cpp`、新規 `tests/models/qwen/hip/attention/kvarn_*_test.cpp`、`tests/models/qwen27b/`（CLI・serving テスト）、フェーズ 0 の品質評価ハーネス（`tests/models/qwen/` と `tools/qwen27b/`）、`tools/qwen27b/check.py` の `SUITES`、`tests/models/qwen/CMakeLists.txt` の `add_qwen_hip_test` 登録。
- 文書: `docs/models/qwen3.8-27b/{README,QUALITY,BENCHMARKS,EXPERIMENTS}.md`（README はテール長ごとの比較表）、`docs/CLI.md`（`--kv-cache`、`--kv-cache-tail`）、`docs/TESTING.md` が必要とする品質ゲート証拠。
- 依存: 新規外部依存なし。HIP のみで実装し（Triton/AOTriton/CK/MIOpen を production 依存にしない）、参照実装のコードを翻案する場合のみライセンス記録（`NOTICE`、`THIRD_PARTY_NOTICES.md`、`licenses/`）を追加する。
- 性能・メモリ: 262K セッションあたり KV が 16.0 GiB（FP16）→ 4.45 GiB（`kvarn-k4v4`、テールなし）〜 4.58 GiB（同、L=2048）。FP16 スケール時の本体 4.4375 bpv に sink とテールを加えて算出した。長文脈の tg と pp は FP16 KV より下がる（G1 の 2 回目の後、キーごと処理で `kvarn-k4v4` テール長未指定が 32K 0.949・128K 0.843、`kvarn-k6v5` L=2048 が 32K 0.808・128K 0.631。pp は G1 の測定で 0.82〜0.98）。性能は Qwen3.8 27B の Q8_0 GGUF で測り、G1 の再判定で記録した値を基準値として退行を防ぐ（design D19）。

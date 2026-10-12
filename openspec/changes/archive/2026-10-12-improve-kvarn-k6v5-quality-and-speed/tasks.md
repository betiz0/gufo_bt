# Tasks

## 1. 評価基盤: gpu-test 専用の基準線（V-01）

- [x] 1.1 `CMakeLists.txt` に基準線用のコンパイル定義オプション（前回と同じ `GUFO_QWEN_PSEUDO_KVARN`、既定 OFF、説明 “Build the Qwen KV quantization quality probes”）を再追加し、`CMakePresets.json` の `gpu-test` だけで ON にする。完了条件: `nix develop -c cmake --preset gpu-test` と `cmake --preset release` が成功し、`grep GUFO_QWEN_PSEUDO_KVARN build/gpu-test/CMakeCache.txt` が `:BOOL=ON`、`build/release/CMakeCache.txt` が `:BOOL=OFF` を返す
- [x] 1.2 `src/cli/kv_cache_options.hpp` と `src/models/qwen/hip/execution_policy.hpp` に `baseline-q8` / `baseline-q4` を追加する（定義時のみ受理、ヘルプ一覧に表示、`--kv-cache-tail` との併用を起動時エラー、専用ストレージ種別として `Fingerprint()` と互換同一性キーに含める）。完了条件: gpu-test の `gufo serve llm --help` に 2 値が含まれ、`--kv-cache baseline-q8 --kv-cache-tail 2048` が起動時エラーになり、release の `gufo` は `--kv-cache baseline-q8` を未知の値として非ゼロ終了する
- [x] 1.3 書き込み時変換（回転 H·x → ブロック 32 の q8_0 / q4_0（FP16 スケール）→ 逆量子化 → 逆回転 Hᵀ）を `fused.hip` と `prefill_rope.hip` の密な FP16 経路へ実装する（sink 0〜127 は対象外、回転は正規化 Hadamard 行列に固定し KVarN 側の回転に追随させない、読み出し経路は変更しない）。完了条件: gpu-test で `baseline-q8` / `baseline-q4` を指定した品質評価が完走し、非有限 logits が 0 件である
- [x] 1.4 G0 の窓（`g0-tail`、深度 4K）で E_q8 / E_q4 を測り、G0 の記録との差を記録する（合否の条件にしない）。完了条件: 4K の 4 指標の G0 との差と、差の要因の仮説（回転領域の量子化結果を逆回転して FP16 に格納するときの追加の丸め。大きさの見積りは未検証）が `docs/models/qwen3.8-27b/QUALITY.md` に実測と並べて記録されている
- [x] 1.5 `tests/cli/kv_cache_test.py` を「gpu-test では 2 値を受理し、`--kv-cache-tail` との併用は起動時エラー、release では未知の値として拒否」に更新する。完了条件: `python3 tests/cli/kv_cache_test.py build/gpu-test/gufo` と `python3 tests/cli/kv_cache_test.py build/release/gufo` が両方成功する（`ctest --preset gpu-full -R '^kv_cache_cli_test$'` も PASS）
- [x] 1.6 FP16 経路の不変性を確認する。完了条件: `ctest --preset gpu-full -R '^(qwen_attention_kv_storage_ops_test|qwen_graph_ops_test)$'` が PASS し、既定 `fp16` の保持バイト列と同一性キーが変わっていない

## 2. 品質ハーネスと評価窓（V-02、V-03）

- [x] 2.1 `tests/models/qwen/hip/quality/kvarn_quality_eval.cpp` の単位 JSON に位置ごとの `top1_match` と `argmax` ID（候補・基準線）を追加する。完了条件: 既存の集計 `top1_agreement` が位置配列からの再計算値と一致し、短い 1 単位（深度 4096、1 コーパス）の出力 JSON に配列が含まれる
- [x] 2.2 `tools/qwen27b/kvarn_quality.py` を同一ラベル基準線とゲート（品質優先 0.13、努力目標 0.10 は記録のみ、メモリ優先 1.0）に更新する（アンカーを使わない）。完了条件: `report` が同一ラベルの `baseline-q8` / `baseline-q4` から p を計算し、`--gate` が k6v5 を 0.13、k4v4 を 1.0 で判定して 0.10 を記録に併記し、`--anchors-*` なしで完走する
- [x] 2.3 煙試験: gpu-test で 1 コーパス・深度 4K・`--candidates baseline-q8,baseline-q4,kvarn-k6v5 --tails none,2048` を `ledger_run.py` 経由（見積もり 1 時間以内）で実行する。完了条件: 同一ラベルの p と窓単位ブートストラップ 95% 区間が出力され、非有限 logits が 0 件である
- [x] 2.4 32K の評価窓を増やす: `t08-reduced` に新規シードのコーパスを追加し（`kvarn_quality.py corpus --seeds …`）、教師を測って D14 の除外を確定する。完了条件: 除外後 6 窓以上が残り、窓・シード・除外が成果物に記録されている
- [x] 2.5 128K の開発用の窓を用意する: G0 のコーパス 0〜2 の窓で、重みとトークン列の SHA-256 が `g0-tail` の manifest と一致することを確かめて教師を再利用する。完了条件: 一致の確認結果と、再利用する教師の場所が成果物に記録されている
- [x] 2.6 128K の判定用の窓を用意する: 除外を見込んで多めに新規シードのコーパスを生成し、教師を 5 窓以内の単位に分けて測る。完了条件: D14 の除外後 6 窓以上が残り、シード・窓・除外と各単位の見積もり・実測が `artifacts/test-ledger.jsonl` と成果物に記録され、この時点で候補は測っていない
- [x] 2.7 評価窓と除外の記録を `docs/models/qwen3.8-27b/QUALITY.md` に反映する。完了条件: 4K / 32K / 128K（開発用・判定用）の窓数・シード・出典・除外理由と、128K の判定用の窓を改善の確認に使わないことが記載されている

## 3. 基準線と現行候補の測定、速度基準点（V-04、V-05）

- [x] 3.1 4K / 32K の全窓で 5 構成（`baseline-q8`、`baseline-q4`、`kvarn-k6v5 --kv-cache-tail 2048`、`kvarn-k5v4 --kv-cache-tail 2048`、`kvarn-k4v4 --kv-cache-tail 256`）を同一ラベルで測る（1 時間以内の単位に分割、台帳経由）。完了条件: 全単位が完了し、非有限 logits が 0 件である
- [x] 3.2 128K の開発用の窓で同じ 5 構成を測る（1 単位 5 窓以内）。完了条件: 全単位が完了し、非有限 logits が 0 件である
- [x] 3.3 128K の判定用の窓で基準線 2 構成（`baseline-q8`、`baseline-q4`）だけを測る（1 単位 5 窓以内）。完了条件: 全単位が完了し、非有限 logits が 0 件で、判定用の窓の候補の単位が存在しない
- [x] 3.4 現行の p（4K / 32K と 128K の開発用、4 指標）を同一ラベルの基準線から算出し、アンカー方式の記録と並べて記録する。完了条件: `report` の点推定と区間が `QUALITY.md` に記録されている
- [x] 3.5 速度基準点を取る: `python3 tools/qwen27b/g1.py run --binary build/release/gufo --depths 8192,131072 --configs k6v5-t2048 --out artifacts/g1-k6v5-8k128k-r1`（r2、r3 も同様）を 3 系列実行し、各系列の `wait` と `evaluate` を行う。完了条件: 8K / 128K の tg と pp の FP16 比がモデルの SHA-256 とともに記録されている

## 4. 品質の原因調査（V-06）

- [x] 4.1 32K のスパイクの対応づけ（オフライン）: `g0-tail` の位置ごとの KL（Phase 0 の試験経路の成果物）と、マージ後の本番経路の成果物（`merge-upstream-quality`）を、G0 の窓の 32K で同じ窓・同じ位置に対応づけ、差の大きい位置（c0 の位置 2、c2 の位置 1、c1 の位置 52 など）を特定する。完了条件: 位置ごとの差の表と、prefill 直後の位置に偏っているかどうかの集計が成果物に残っている
- [x] 4.2 32K のスパイクの切り分け（短い GPU 実行、見積もり 1 時間以内、台帳経由）: 4.1 の位置について、本番経路の prefill チャンク幅（256 と 4096）と、前置文脈の末尾を逐次 decode で通した場合を比べる。完了条件: prefill から decode への切り替え境界の不具合か量子化誤差かの判定と根拠が `docs/models/qwen3.8-27b/EXPERIMENTS.md` に記録され、不具合の場合は原因と修正方針が 7 章の改善候補に加えられている
- [x] 4.3 教師実行のダンプ（gpu-test のダンプオプション）で、128K の開発用の窓の高 d 位置について、一部層の FP16 K/V タイルとクエリを書き出す。完了条件: 再現可能な手順と入出力の SHA-256 が記録され、CPU 側で読み込める
- [x] 4.4 CPU オラクル（`src/models/qwen/kvarn_reference.cpp`）で 7 変種（現行、FP32 メタデータ、K の平均を引く、固定の符号パターンを掛けた Hadamard、雑音のバイアス補正、q8_0、k8）を比較する。完了条件: スコア誤差・量子化領域への注意質量・コピー元重みの変化が表になり、改善策が順位付けされている
- [x] 4.5 仮説 H1〜H4 と 32K のスパイクの切り分け結果、改善策の順位を `EXPERIMENTS.md` に記録する。完了条件: 採用候補と根拠が 1 行以上で記載されている

## 5. 速度スパイクの一次判定（V-07）

- [x] 5.1 `tools/qwen27b/kvarn_read_compute_bench.hip` にタイル単位の試作（KV ヘッド × split あたり 1 ワークグループ、GQA 6 ヘッド共有、タイル単位の最大値・exp・再スケール、V 展開の共有）を追加する。完了条件: `tools/bench/build.sh`（`kvarn_reference.cpp` をリンク）でビルドでき、CPU 参照一致のチェックが PASS する
- [x] 5.2 資源と速度を測る: VGPR / SGPR / LDS / 占有率 / spill と、同じベンチ内の試作・現行カーネル・FP16 の、キャッシュ常駐の直接時間（8K 文脈）と 128K 相当のストリーミングの時間。完了条件: 3 者の比較表が残り、spill と占有率の崩れの有無が記録されている
- [x] 5.3 一次判定: 同じベンチの 128K 相当のストリーミングで、試作の読み出し時間が現行カーネルの 0.5 倍以下で、spill と占有率の崩れがないこと。完了条件: 判定と根拠が `EXPERIMENTS.md` に記録され、no-go の場合は速度の作業を止める（6 章を省略し、7 章へ進む）

## 6. 読み出しの作り直し（一次判定が go の場合のみ、V-13、V-08、V-14）

> 一次判定（5.3）は no-go（試作/現行 = 1.3846 > 0.5）だったため、この章は design D4/D5 の定めどおり省略した（EXPERIMENTS.md の 5.1〜5.3 行）。

- [ ] 6.1 二次判定: 試作を本番の起動経路に仮組みし、`python3 tools/qwen27b/g1.py run --depths 0,131072 --configs k6v5-t2048`（FP16 を同じ系列）で、深度コスト（tg のトークン時間の深度 128K と深度 0 の差）を求める。完了条件: 深度コストが 65.8 ms（131.6 ms の 0.5 倍）以下なら 6.2 へ進み、超えれば仮組みを削除して速度の作業を止め（6.2〜6.8 を省略し、7 章へ進む）、どちらの場合も測定値と判定が `EXPERIMENTS.md` に記録されている
- [ ] 6.2 `src/models/qwen/hip/kernels/kvarn_read.hip` のウォーカーをタイル単位の共通ヘルパーに置き換え、online・split-K・因果検証が同じ順序で使うようにする。完了条件: `ctest --preset gpu-full -R '^qwen_attention_kvarn_read_ops_test$'`（T-05 のビット一致を含む）が PASS する
- [ ] 6.3 因果検証のキー 1 個ごとの同期と要素ごとの展開をタイル単位に置き換える。完了条件: `speculative_verification_test` と `qwen_attention_kvarn_read_ops_test` が PASS し、VGPR と占有率がスパイクの範囲内である
- [ ] 6.4 prefill の要素ごとの展開（`KvarnStoredKElement` / `KvarnStoredVElement`）をタイル単位のベクトル展開に置き換える。完了条件: `qwen_attention_kvarn_prefill_ops_test` と `qwen_attention_kvarn_prefill_read_ops_test` が PASS する
- [ ] 6.5 封印カーネルへの介入要否を 8K / 128K の pp プロファイルで判断し、記録する。完了条件: `tools/prof/prof.py` の結果と判断が `EXPERIMENTS.md` にある
- [ ] 6.6 回帰を確認する: `qwen_attention_kvarn_(seal|read|prefill|prefill_read|state)_ops_test`、`qwen_graph_ops_test`、`qwen_dflash_gpu_test`（`CheckKvarnDflashStateMatchesDecode`）。完了条件: すべて PASS（モデル未設定の skip は合格としない）
- [ ] 6.7 深い文脈での DFlash2 と AR の一致を確かめる: `tools/qwen27b/kvarn_consistency.py` を `--kv-cache kvarn-k6v5 --kv-cache-tail 2048` で、32K の 4 単位以上（1 本）と 128K の 3 単位以上（1 本、配置は `tools/qwen27b/probe_kvarn_128k.py`）について、封印境界を最初の検証窓に含む単位を選んで実行する（各 1 時間以内、台帳経由）。完了条件: 全単位で不一致 0・エラー 0 で、単位の一覧と封印境界の位置が成果物に記録されている
- [ ] 6.8 
<!-- 6.1-6.8 の記録 (2026-10-12): 5.3 の一次判定が no-go (試作/現行 = 1.3846 > 0.5) のため design D4/D5 どおり章 6 を省略。EXPERIMENTS.md の 5.1-5.3 行と QUALITY.md に記録。 -->

作り直し後の 8K / 128K を 3.5 と同じ方法（3 系列）で測る。完了条件: tg と pp の各 FP16 比が 3.5 の値の 0.97 倍以上である比較表が残り、下回った場合は原因を解消するまで作り直しを採用しない

## 7. 品質改善（V-09）

> 利用者の判断（2026-10-11）により改善は採用しない（記録のみ）。4.2 の切り分けは「prefill（WMMA）の f16 二重丸め」で不具合ではなく design D5 が維持する性質、4.4 の変種はいずれも bpv 増か効果 ≤10% のため採用しない。分析と順位は EXPERIMENTS.md の 4.2〜4.5 行、p の未達（128K 開発用 0.178 > 0.13）と対応は QUALITY.md に記録。判定は判定用の窓（task 9.1/9.2）で行う。

- [ ] 7.1 4 章の上位改善（4.2 で見つかった不具合の修正を含む）を共通ヘルパーに実装する。bpv が増える場合は利用者の確認を先に得る。回転の変種（固定の符号パターンを掛けた Hadamard）を採る場合は、書き込み側の回転・注意出力の逆回転・CPU オラクル・回転のテストを同じ回転にそろえ、基準線の回転は変えない。完了条件: `qwen_kvarn_quant_cpu_test`、`qwen_kvarn_fwht_cpu_test`、`qwen_attention_fusion_ops_test`、`qwen_attention_kvarn_seal_ops_test`（封印の決定性）、`qwen_attention_kvarn_prefill_ops_test`（T-04 のチャンク不変性）が PASS する
- [ ] 7.2 簡易確認: 4K / 32K の全窓と 128K の開発用の窓を同一ラベルの基準線で測る（128K の判定用の窓は使わない）。完了条件: p の変化が記録され、非有限 logits が 0 件である
- [ ] 7.3 
<!-- 7.1-7.3 の記録 (2026-10-12): 利用者の判断 (2026-10-11) で改善は採用なし (記録のみ)。4.2 の切り分けは本番 prefill (WMMA) の f16 丸めで design D5 が維持する性質、4.4 の変種は bpv 増かスコア誤差 10% 以内。よって 7.2 の再測定と 7.3 の速度比較は対象外。EXPERIMENTS.md と QUALITY.md に記録。 -->

速度の非劣化を確認する: 8K / 128K を同じ方法（3 系列）で測り、6.8 の値（6 章を省略または中止した場合は 3.5 の値）と比べる。完了条件: tg と pp の各 FP16 比が比較元の 0.97 倍以上である比較表が残り、下回った改善は採用しない

## 8. 版と同一性キーの確定（go / no-go に関係なく行う、V-12）

- [x] 8.1 6 章と 7 章の結果から、ペイロードの配置・量子化演算・読み出し経路の演算のそれぞれについて変更の有無を確定する。完了条件: 判定と根拠（変更したファイルと内容）が design の D6 の区分で `EXPERIMENTS.md` に記録されている
<!-- 8.2 の記録 (2026-10-12): 8.1 の区分判定でペイロード・量子化演算・読み出しの演算がいずれも不変のため 8.3 を適用 (版・state_abi・同一性キーは変更なし)。EXPERIMENTS.md に記録。 -->
- [ ] 8.2 変更があった場合は、区分に応じて `state_abi` の KVarN 側の世代（`QwenStateAbiName`）と、ペイロードの配置または量子化演算を変えた場合は `kKvarnCompactSnapshotVersion`（永続 payload に影響する場合は `kQwenPersistentPayloadVersion` も）を上げる。完了条件: 新しい世代の KVarN `state_abi` と FP16 の文字列が不変であることを固定する検査が追加されて PASS し、旧世代の同一性キーで書いた KVarN のディスクキャッシュが再利用されず再計算になることがテストで確認され、`qwen_attention_kvarn_state_ops_test` が PASS する
- [x] 8.3 変更がなかった場合は、版と同一性キーが変わっていないことを確認する。完了条件: KVarN と FP16 の `state_abi`・コンパクトスナップショットの版・同一性キーの文字列が変更前と同じである

## 9. 最終判定と文書（V-10、V-11）

- [x] 9.1 候補 `kvarn-k6v5 --kv-cache-tail 2048` と `kvarn-k4v4 --kv-cache-tail 256` だけを、4K / 32K の全窓と 128K の判定用の窓で測る（1 単位 5 窓以内、台帳経由）。完了条件: 全単位が完了し、非有限 logits が 0 件で、128K の判定用の窓の候補の測定がこの 1 回だけである
- [x] 9.2 `python3 tools/qwen27b/kvarn_quality.py report --out <out> --label <label> --gate` で、k6v5 の p ≤ 0.13（必達）と p ≤ 0.10（記録）、k4v4 の p ≤ 1 を判定し、区間と非有限数を併記して `QUALITY.md` に記録する。完了条件: 判定結果・窓数・シード・除外が記載されている
- [x] 9.3 8K / 128K の速度を 3 系列で測り、3.5 の基準点と並べて `BENCHMARKS.md` と `README.md` に記録する。完了条件: FP16 比とモデル SHA-256 が記載され、既存の下限・退行規則の再測定と適用が後続の変更であることが明記されている
- [x] 9.4 `README.md` の比較表と `docs/CLI.md` の KVarN 記載を更新する。完了条件: テール長ごとの指標・bpv・容量が値を測った評価窓とともに掲載され、長文脈で FP16 より遅いことと、本変更の記録が反映されている
- [x] 9.5 `EXPERIMENTS.md` に採用・不採用の判断（一次・二次判定、改善策、32K のスパイク）をまとめる。完了条件: 各実験の結果と採否が 1 行以上で記載されている

## 10. 統合確認

- [x] 10.1 演算子と CPU オラクルの一括確認: `ctest --preset gpu-full -R '^qwen_attention_kvarn' --output-on-failure` と `ctest --preset cpu-test -R '^qwen_kvarn_(quant|fwht)_cpu_test$' --output-on-failure`。完了条件: すべて PASS
- [x] 10.2 CLI と end-to-end: `kv_cache_cli_test`（gpu-test と release）、`qwen_dflash_gpu_test`、`speculative_verification_test`。完了条件: すべて PASS（skip は理由を記録し、合格として扱わない）
- [x] 10.3 リリースビルドの煙試験: `cmake --preset release && cmake --build --preset release --parallel 4`、`build/release/gufo diagnose`、既定 `fp16` と `--kv-cache kvarn-k6v5 --kv-cache-tail 2048` で短い生成。完了条件: 起動と生成が成功し、release には baseline 値が無い
- [x] 10.4 フォーマットと spec 対応: `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py` と `openspec validate improve-kvarn-k6v5-quality-and-speed` を実行し、spec の各 Requirement と検証 ID（V-01〜V-14）の対応を確認する。完了条件: フォーマット違反 0 件で、検証が有効である

## Workflow follow-up

- 実装は apply ワークフローで行う（このタスクリストを順に実施する）。
- 実装後に `openspec verify`（必要なら深い検証）で実装とアーティファクトの一致を確認する。
- 検証と利用者の判断の後にアーカイブする。アーカイブで本仕様へ差分が反映される。

<!-- 10.2/10.3 の記録 (2026-10-12): CLI = gpu-test 19.9 s と release 版とも pass (release は --kv-cache CLI checks passed.)。qwen_dflash_gpu_test = モデルパスを与えて ALL TESTS PASSED (KVarN DFlash2 tail 0/2048 が AR greedy と一致、state exact 12 窓)。speculative_verification_test = pass。skip 3 件の理由 = GUFO_MULTI_SESSION_TESTS=1 未設定 (2 件) と GUFO_DEEPSEEK_V4_FLASH_MODEL 未設定 (1 件) で、いずれも本変更の対象外。10.3 = release ビルド (no-op) 成功、gufo diagnose [PASS] (gfx1151 supported)、fp16 と kvarn-k6v5 tail 2048 の 16 トークン生成が成功、release に baseline 値なし (--kv-cache baseline-q8 は exit 2)。 -->

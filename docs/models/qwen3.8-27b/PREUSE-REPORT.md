# KVarN 利用開始前検証 — 経過報告書（2026-10-08 時点）

対象: `openspec/changes/add-kvarn-kv-cache-quantization` の利用開始前確認（design D18/D20/D22）。
環境: AMD Strix Halo `gfx1151`、Qwen3.8-27B Q8_0、host ROCm。
この報告書は経過の記録であり、契約（spec/design）の判定は該当文書と `EXPERIMENTS.md`・`tasks.md` を正とする。

---

## 1. 要約

| 項目 | 状態 |
|---|---|
| G2 の DFlash2 再測定（3 系列） | **完了・合格**（設計 7.8、D19/D24） |
| 4.4 一致性スイート（縮小版、`kvarn-k5v4` t2048） | **完了・合格**: 299 単位すべて一致、不一致 0、封印境界 8/8（128K） |
| 6.1（T-08）縮小版の本測定 | **未完了・停止**: 品質ハーネスで本番 KVarN の候補ジョブが **4 回連続で GPU ウェッジ**（教師 FP16 は毎回正常） |
| 6.1 の教師成果物 | 完了（24/24、8 コーパス × 深度 4K/32K/128K）。候補・基準線は未取得 |
| T4〜T10（比較コード検証、深度別の再開） | 未実施（T2/T3 の停止により分岐「両方が止まった」に該当） |
| 利用開始 | 未判断（6.1 の品質値が未確定） |

---

## 2. 完了した検証

### 2.1 G2（DFlash2 の性能再測定、tasks 7.8 / design D18・D19・D24）

- 3 系列（`artifacts/g2fix-r{1,2,3}`、要約 `artifacts/g2fix-summary.json`）、Q8_0、DFlash2-Q8_0 ドラフト、各系列で FP16 を同時測定。
- FP16 tg 23.48 / 17.97 / 38.36 / 7.56 t/s（深度 0 / 4K / 32K / 128K）。KVarN DFlash2 tg の FP16 比は d0 **0.994〜0.995**（**D19 の深度 0 下限 0.95 を全 6 プリセットで充足**）、4K 0.769〜0.971、32K 0.579〜0.628、128K 0.347〜0.807。pp は全点で G1 基準値の ≥ 0.998 倍。
- 修正効果（無効な修正前値との比較）: d0 は 0.587〜0.588 → 0.994〜0.995（欠陥が受理率 0.277 まで落としていた）、4K は 5/6 改善、32K 不変、128K は 5/6 改善。
- 契約改訂（利用者、2026-10-05、D24）: 深度 4K の DFlash2 tg は下限にも基準値からの退行にも含めず記録のみ。

### 2.2 4.4 一致性スイート（縮小版）

- ハーネス `tools/qwen27b/kvarn_consistency.py`、`serve llm --kv-cache kvarn-k5v4 --kv-cache-tail 2048`、1 セッション、各 1 リクエスト、較正ループなし（`--no-boundary-target`）。
- スイート `artifacts/consistency-160.json` の先頭 97 ケース × 深度 0/4K/32K = 291 単位 ＋ 深度 128K 8 単位 = **299 単位すべて一致・不一致 0・エラー 0**（`completed 291 units in 527.3 min; mismatches 0`、128K は `report --gate` rc=0）。

| 深度 | 単位 | 一致 | 不一致 | 封印境界が検証窓内 |
|---|---:|---:|---:|---:|
| 0 | 97 | 97 | 0 | 1 |
| 4,096 | 97 | 97 | 0 | 96 |
| 32,768 | 97 | 97 | 0 | 29 |
| 131,072 | 8 | 8 | 0 | **8/8**（要件 ≥4） |
| 合計 | 299 | 299 | **0** | 134 |

- 128K の 8 単位はプロンプト長をトークナイザ事前計数で配置（専用 16K filler、`tokens(k·L+r) = probe(r) + k·E`、E=4,530、4 ブロック検証で delta 0）。`chars` 483,056〜483,183 → 133,373〜133,376 トークン。
- 出力: `~/.cache/gufo/kvarn-consistency/44-reduced/`（`summary.json`、`CONSISTENCY-SUMMARY.json`）、スクリプト `run-44-spec.sh` / `run-44-128k.sh`、配置ツール `tools/qwen27b/probe_kvarn_128k.py`。

### 2.3 既存データからの報告（新規測定なし）

G0（`kvarn-k5v4`）各テール長の最悪 p（律速 = 指標・深度）:

| テール長 | 最悪 p | 律速 | メモリ優先（p ≤ 1） |
|---|---:|---|---|
| 未指定 | 0.874 | 4096 mean KL | PASS |
| 256 | 0.600 | 4096 大偏差率 | PASS |
| 1024 | 0.433 | 32768 top-1 不一致率 | PASS |
| 2048 | 0.533 | 32768 top-1 不一致率 | PASS |

使い分け（`kvarn-k5v4` テール 2048 / FP16 比）:

| 深度 | AR tg (t/s) | AR/FP16 | DFlash2 tg (t/s) | DFlash2/FP16 |
|---|---:|---:|---:|---:|
| 0 | 7.67 | 0.996 | 23.34 | 0.994 |
| 4,096 | 7.52 | 0.984 | 16.00 | 0.890 |
| 32,768 | 6.55 | 0.910 | 23.26 | 0.606 |
| 131,072 | 4.54 | 0.752 | 5.57 | 0.737 |

→ 全深度で DFlash2 が AR より速い（design D20 / README の「超長文脈では AR」記述を修正した根拠）。

---

## 3. 発生した障害と対処（時系列）

### 3.1 4.4 実行中の障害（解決済み）

1. **見積もりの約 10 倍の実時間**: 各リクエストが深度プレフィルを再充填する（実測 273 t/s、`ttft_ms≈130,000` @ 35K トークン）ため、単位あたり 0/4K/32K/128K = 6.3/21.3/135.3/約 510 秒。記録した見積もり（2.5 h）は decode 側速度のみから出しており誤りだった。
   - 対処: 利用者の決定（2026-10-06）で範囲を縮小（AR は実行中の単位で停止、DFlash2 は完了単位のみ、128K は 8 単位）。
2. **較正ファイルの破壊**: 128K 配置プローブがパスと `schema_version` を誤り（親ディレクトリ・値 2、正しくは `<label>/calibration.json` と `kc.SCHEMA_VERSION=1`）、ハーネスが較正を全却下して AR 181 単位の再実行を始めた。
   - 対処: ハーネスを即時停止し、AR 成果物 290 件の `chars`/`prompt_tokens` から較正を復元。プローブを修正。
3. **配置モデルの 1 ブロックずれ**と**剰余のみ保存**（正しくは `tokens(k·L+r) = probe(r) + k·E`、較正には全長 `chars`）。旧実装の実プロンプトは約 4.3K トークンだった。
   - 対処: v2（式の修正）→ v3（剰余を直接解く探索）で **8/8 を窓内に配置**。
4. **旧い 16 単位計画（bg_54）の残骸**が既定 filler で 128K の AR 成果物 10 件を書いていた。
   - 対処: 該当成果物と不整合 unit を削除。
5. **比較が vacuous だった欠陥（最重要）**: チャット応答は `reasoning_content`（推論）と `content`（本文）に分かれ、32 トークンの生成が推論内で尽きると本文が空になる。ハーネスは `content` のみを比較していたため、**全 299 単位が「空 == 空」で一致**と判定されていた（`completion_tokens=32` と矛盾して発覚。実機プローブで応答形状を確認）。
   - 対処（コミット `e9e6533`）: `reasoning_content` + `content` を連結して比較、生成トークン > 0 で本文が空ならエラー、識別子に `response_text=2` を追加して旧成果物を自動無効化。**再実行で 299/299 一致（不一致 0）を実テキスト比較で確認**。

### 3.2 6.1（T-08 縮小版）の障害（未解決）

1. **候補名の誤り**: `--candidates` に内部名 `pseudo-kvarn-*` を渡していた（ハーネスの受理名は `kvarn-k4v4/k5v4/k6v5`、`baseline-q8/q4` は自動生成）。修正済み。
2. **25 時間ハング（10-07 11:14 〜 10-08 10:35）**: run 1 が教師 24 ジョブ完了（24/48）直後の最初の候補ジョブで停止。プロセスは `State: R`、utime+stime ≈ **50 CPU 時間**、ログ増加なし。停止（SIGTERM）時に `MES failed to respond` → `MES might be in unrecoverable state` → `GPU reset begin!. Source: 3` → `MODE2 reset` → 復帰。
3. **GPU ウェッジの再現（T2/T3）**: 利用者の切り分け手順（design D22）に従い 1 テストずつ実行。

| テスト | 構成 | 見積もり | 結果 |
|---|---|---:|---|
| T1 | dmesg/journalctl の確認 | 20 s | **pass**（ハング中はカーネル記録なし。dmesg は権限不可 → journalctl に置換） |
| T2 | `kvarn-k4v4` t256, corpus-0, 深度 4096 | 300 s | 失敗（既存 label の manifest と競合 → 新 label でやり直し） |
| T2b | 同上（新 label） | 300 s | **候補ジョブが停止**（教師 47.0 s は成功）。`gpu_busy_percent` は 5 サンプル連続 **100%**、`State: R`（≈2 コア）。停止時に `GPU reset(29)` |
| T2-FP16 | 教師のみ（corpus-0, 深度 4096） | 120 s | **pass**（47.16 s / 計 62 s、exit 0、udev イベントなし） |
| T3 | `kvarn-k5v4` t2048, 同じ corpus-0・深度 4096 | 300 s | **候補ジョブが停止**（教師 47.15 s は成功、11 分以上未完了、`gpu_busy_percent=100`、`State: R`） |

- 煙テスト（隔離 label）も同一の候補ジョブで停止 → **本番 KVarN の候補ジョブは 4/4 で停止**。
- GPU リセットは 3 回とも自動復帰: 10-08 **10:35:13**（reset begin/成功）、**10:46:52**（`GPU reset(28) succeeded!` / `device wedged, but recovered through reset`）、**11:06:57**（`GPU reset(29) succeeded!` / 同文）。
- カーネルログはハング中に**何も記録しない**（ドライバがウェッジを検知せず、プロセス停止時にのみリセットが走る）。`gpu_busy_percent` はハング中も 100%（GPU 側カーネルが回り続けている）。CPU 側スタックは `ptrace_scope=1`・gdb なしのため取得できず。

---

## 4. 現在の状態と判断材料

- **T3 は停止させずに走らせたまま**（利用者判断 2026-10-08、A4: 見積もり超過時は止めずに報告して判断を待つ）。GPU は 100% のまま占有。
- 利用者の分岐「T2 と T3 の両方が止まった」に該当 → デバイスのリセット／再起動の要否を報告する段階。
- **再起動は不要と見る**根拠:
  1. 3 回のウェッジはいずれも amdgpu の自動リセットで復帰している。
  2. **同じハーネスの FP16 経路（教師）は毎回正常**（47.0〜47.2 秒、G0 実測と同じ）。
  3. **本番 KVarN 経路自体は `serve llm` 経由で 299 単位を不一致 0 で完走**（4.4、527 分＋167 分）。
  → 停止は「品質ハーネス（gpu-test の `qwen_kvarn_quality_eval`）× 本番 KVarN 候補」の組み合わせに固有。この組み合わせは今回が初めてで、G0 は疑似量子化経路だった（テール 256 も G1・G2・4.4 で本番経路に通していない）。
- 未取得の結果: 6.1 の 5 構成（`kvarn-k4v4` t256 / `kvarn-k6v5` t2048 / `kvarn-k5v4` t2048 / E_q8 / E_q4）の p と窓単位ブートストラップ 95% 区間、および比較コード検証（T4）と深度別再開（T5〜T10）。

---

## 5. 実行方法（再現）

- テスト実行の共通ラッパー: `tools/qwen27b/ledger_run.py`（design D22）。`artifacts/test-ledger.jsonl` に IN / PROGRESS / OUT を追記し、`artifacts/test-runs/<id>/` にログと `udev.log`（`udevadm monitor --kernel --property --subsystem-match=drm`）を残す。OUT のカーネル抜粋は udev と `journalctl -k`（dmesg は root 権限が必要なため不使用）。
- 4.4: `run-44-spec.sh`（浅い 3 深度）→ `run-44-128k.sh`（128K 8 単位、`--filler filler-128k.txt`）→ `kvarn_consistency.py report --gate`。
- 6.1: `build/gpu-test/tests/models/qwen/qwen_kvarn_quality_eval ... --candidates kvarn-k4v4 --tails 256 --depths 4096 --window 256 --chunk 256`（教師 FP16 は自動、基準線 E_q8/E_q4 も自動生成）。障害時の切り分けは本報告書 3.2 のとおり。
- 記録の所在: `artifacts/test-ledger.jsonl`、`artifacts/test-runs/`、`~/.cache/gufo/kvarn-consistency/44-reduced/`、`~/.cache/gufo/kvarn-quality/{t08,t2diag,t3diag}/`、`docs/models/qwen3.8-27b/EXPERIMENTS.md`、`openspec/changes/add-kvarn-kv-cache-quantization/{design,tasks}.md`。

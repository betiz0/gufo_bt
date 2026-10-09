# Spec Delta

## ADDED Requirements

### Requirement: サーバオプションとの共存

システムは、`serve llm` で `--kv-cache` と `--kv-cache-tail` を、サーバのログ・トレース・キューのオプション（`--log-level`、`-v/--verbose`、`--log-progress`、`--trace`、`--max-pending`）と同じコマンド行で同時に受理しなければならない。(SHALL) 同時に指定された場合も、`--kv-cache` と `--kv-cache-tail` の検証はモデルの読み込み前に完結し、不正な指定は非ゼロで終了しなければならない。(SHALL) `serve llm --help` は、`--kv-cache`・`--kv-cache-tail` と上記のサーバオプションをすべて表示しなければならず、`--kv-cache`・`--kv-cache-tail` の別名を表示してはならない。(SHALL)

#### Scenario: サーバオプションと併存して起動する
- **WHEN** Qwen3.8 27B に対して `serve llm --kv-cache kvarn-k6v5 --kv-cache-tail 2048 --log-level=info --log-progress` を指定する
- **THEN** 起動に成功し、起動ログに KV 形式 `kvarn-k6v5` とテール長 2048 が報告される

#### Scenario: 不正な KV 指定は他のオプションの指定があっても拒否される
- **WHEN** `serve llm` で `--log-progress --log-level=info` または `--trace <path>` と同時に、`--kv-cache kvarn-k9v9`、`--kv-cache-tail 255`、`--kv-cache fp16 --kv-cache-tail 1024` のいずれかを指定する
- **THEN** いずれもモデルを読み込まずにエラーで非ゼロ終了する

#### Scenario: help に KV オプションとサーバオプションが並ぶ
- **WHEN** `serve llm --help` を実行する
- **THEN** 出力に `--kv-cache`・`--kv-cache-tail` と `--log-level`・`-v/--verbose`・`--log-progress`・`--trace`・`--max-pending` が含まれ、`--kv-cache`・`--kv-cache-tail` の別名は含まれない

## MODIFIED Requirements

### Requirement: 既定の KV 形式

Qwen3.8 27B の KV 形式の既定値は `fp16` でなければならない。(SHALL) KVarN は非可逆な形式であるため、既定値の昇格は品質契約の改定を伴う別の判断として扱い、この変更では既定を変更してはならない。(SHALL) `--kv-cache` を指定しない場合、すべてのモデルは、同じコミットの upstream のビルドと同じ KV 形式・出力・スナップショット・互換同一性キーを使わなければならない。(SHALL)

#### Scenario: 指定なしでは既存の FP16 経路が使われる
- **WHEN** Qwen3.8 27B で `--kv-cache` を指定せずに実行する
- **THEN** FP16 KV が使われ、同じコミットの upstream のビルドとの互換同一性キーの一致はコード上の等価確認で、出力とスナップショットのバイト列に退行がないことは既存の FP16 テストの退行確認で担保される

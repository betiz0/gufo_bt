# kv-cache-selection Specification

## Purpose
KV キャッシュ形式と KVarN のテール長をユーザーが明示的に選択できるようにし、選択結果を起動ログ・ベンチマーク出力・容量計算に一貫して反映する。

## Requirements

### Requirement: KV 形式の正規オプション

システムは、`serve llm`・`chat`・`prompt`・`bench` に共通の正規ロングオプション `--kv-cache` を提供し、値として `fp16` または KV プリセット名を受け付けなければならない。(SHALL) 1 つの挙動に対して正規のオプション名は 1 つだけとし、エイリアスを設けてはならない。(SHALL) `--kv-cache` は各コマンドの `--help` に表示されなければならない。(SHALL) 未知の値または未知のオプションは起動時にエラーとして報告し、非ゼロで終了しなければならない。(SHALL)

#### Scenario: 各コマンドで同一オプションが受理される
- **WHEN** Qwen3.8 27B を対象に `serve llm`、`chat`、`prompt`、`bench` のそれぞれで `--kv-cache kvarn-k4v4` を指定する
- **THEN** いずれのコマンドも KV 形式 `kvarn-k4v4` を選択して起動し、起動ログに `kvarn-k4v4` を報告する

#### Scenario: help に表示される
- **WHEN** `serve llm`、`chat`、`prompt`、`bench` のそれぞれで `--help` を実行する
- **THEN** 出力に `--kv-cache` と受け付ける値の一覧が含まれる

#### Scenario: 不正な値がエラーになる
- **WHEN** `--kv-cache kvarn-k9v9` のように未定義の値を指定する
- **THEN** 起動時にエラーとなって非ゼロで終了し、モデルは読み込まれない

### Requirement: テール長の正規オプション

システムは、`serve llm`・`chat`・`prompt`・`bench` に共通の正規ロングオプション `--kv-cache-tail` を提供し、KVarN のテール長を 256 以上 2048 以下の整数で受け付けなければならない。(SHALL) エイリアスを設けてはならず、各コマンドの `--help` に受け付ける範囲とともに表示しなければならない。(SHALL) 範囲外または整数でない値は起動時エラーとして報告し、モデルを読み込まずに非ゼロで終了しなければならない。(SHALL) `--kv-cache` で KVarN プリセットを選択していない場合（未指定または `fp16`）に `--kv-cache-tail` を指定したときは、起動時エラーとして非ゼロで終了しなければならない。(SHALL) `--kv-cache-tail` を指定しない場合、KVarN はテールトークンを追加してはならない。(SHALL)

#### Scenario: 各コマンドでテール長が受理される
- **WHEN** Qwen3.8 27B を対象に `serve llm`、`chat`、`prompt`、`bench` のそれぞれで `--kv-cache kvarn-k4v4 --kv-cache-tail 1024` を指定する
- **THEN** いずれのコマンドもテール長 1024 で起動し、起動ログにテール長 1024 を報告する

#### Scenario: 範囲外のテール長がエラーになる
- **WHEN** `--kv-cache-tail 255`、`--kv-cache-tail 2049`、`--kv-cache-tail abc` のいずれかを指定する
- **THEN** 起動時にエラーとなって非ゼロで終了し、モデルは読み込まれない

#### Scenario: KVarN 以外との組み合わせがエラーになる
- **WHEN** `--kv-cache fp16 --kv-cache-tail 1024`、または `--kv-cache` を指定せずに `--kv-cache-tail 1024` を指定する
- **THEN** 起動時にエラーとなって非ゼロで終了する

#### Scenario: 未指定ではテールトークンを追加しない
- **WHEN** Qwen3.8 27B で `--kv-cache kvarn-k4v4` だけを指定する
- **THEN** テールトークンを追加せずに起動し、起動ログにテール長「なし」を報告する

### Requirement: 既定の KV 形式

Qwen3.8 27B の KV 形式の既定値は `fp16` でなければならない。(SHALL) KVarN は非可逆な形式であるため、既定値の昇格は品質契約の改定を伴う別の判断として扱い、この変更では既定を変更してはならない。(SHALL) `--kv-cache` を指定しない場合、すべてのモデルは、同じコミットの upstream のビルドと同じ KV 形式・出力・スナップショット・互換同一性キーを使わなければならない。(SHALL)

#### Scenario: 指定なしでは既存の FP16 経路が使われる
- **WHEN** Qwen3.8 27B で `--kv-cache` を指定せずに実行する
- **THEN** FP16 KV が使われ、同じコミットの upstream のビルドとの互換同一性キーの一致はコード上の等価確認で、出力とスナップショットのバイト列に退行がないことは既存の FP16 テストの退行確認で担保される

### Requirement: 非対応モデルでのエラー

システムは、Qwen3.8 27B 以外のモデルに対して `--kv-cache` または `--kv-cache-tail` が明示的に指定された場合、値（`fp16` を含む）にかかわらず起動時エラーとし、非ゼロで終了しなければならない。(SHALL) 非対応モデルで指定値を無視して既定の KV 形式で実行してはならない。(SHALL)

#### Scenario: 非対応モデルでは KVarN を指定すると起動しない
- **WHEN** Qwen3.8 Flash-Next または DeepSeek V4 Flash に対して `--kv-cache kvarn-k4v4`、または `--kv-cache kvarn-k4v4 --kv-cache-tail 1024` を指定する
- **THEN** 起動時にエラーとなって非ゼロで終了し、既定の KV 形式で黙って実行されない

#### Scenario: 非対応モデルでは fp16 の明示指定も拒否される
- **WHEN** DeepSeek V4 Flash（KV は FP8）に対して `--kv-cache fp16` を指定する
- **THEN** 起動時にエラーとなって非ゼロで終了する

### Requirement: KV 形式の報告

システムは、起動ログと `gufo bench` の出力に、実際に使用する KV 形式を含めなければならない。(SHALL) Qwen3.8 27B では、あわせてテール長（未指定のときは「なし」）、実効 bpv、セッションあたりの KV 容量を含めなければならない。(SHALL) Qwen3.8 27B 以外のモデルでは、モデル固有の KV 形式（Flash-Next は `fp16`、DeepSeek V4 Flash は `fp8`）を報告しなければならない。(SHALL) メモリ見積と `--sessions` × `--context` の容量計算は、選択された KV 形式のサイズ式を用いなければならない。(SHALL)

#### Scenario: 起動ログと bench に形式と容量が出る
- **WHEN** Qwen3.8 27B を `--kv-cache kvarn-k4v4 --kv-cache-tail 1024` で `serve llm` 起動し、同じ指定で `gufo bench` を実行する
- **THEN** 両方の出力に KV 形式 `kvarn-k4v4`、テール長 1024、実効 bpv、セッションあたり KV 容量が含まれ、容量の値は選択形式とテール長のサイズ式による見積と一致する

#### Scenario: 他モデルは固有の KV 形式を報告する
- **WHEN** DeepSeek V4 Flash を `--kv-cache` なしで起動する
- **THEN** 起動ログの KV 形式は `fp8` と報告される

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

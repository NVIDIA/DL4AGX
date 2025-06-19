# Test

## How to run `test_vad_integration`

### 前提条件

1. **必要なファイルの存在確認**
   - TensorRTエンジンファイル（`demo/engines/`ディレクトリ内）
   - プラグインファイル（`demo/libplugins.so`）
   - テスト設定ファイル（`test/test_config.yaml`）
   - テストデータファイル（`test/`ディレクトリ内のバイナリファイル）

2. **CUDA環境**
   - CUDA対応GPUが利用可能
   - CUDAドライバーとランタイムがインストール済み

### 実行方法

#### 1. テスト設定ファイルの準備

テスト実行前に、設定ファイルを正しい場所にコピーする必要があります：

```bash
# ビルドディレクトリに設定ファイルをコピー
cp test/test_config.yaml build/autoware_tensorrt_vad/
```

#### 2. 通常のビルドとテスト実行

```bash
# パッケージのビルド
colcon build --packages-select autoware_tensorrt_vad

# テスト実行（lint系テストを除く）
colcon test --packages-select autoware_tensorrt_vad --ctest-args -R test_vad_integration
```

#### 3. テスト結果の確認

```bash
# テスト結果の詳細表示
colcon test-result --all

# 特定のテスト結果のみ表示
colcon test-result --verbose
```

### テスト内容

`test_vad_integration`は以下の3つのテストを実行します：

1. **VadIntegrationTest.ModelInitializationWithRealEngines**
   - 実際のTensorRTエンジンファイルを使用したVadModelの初期化
   - プラグインのロード、エンジンのデシリアライズ、CUDAコンテキストの初期化を検証

2. **VadInferIntegrationTest.ModelInitialization**
   - 推論用のVadModelの初期化
   - エラーなしで初期化できることを確認

3. **VadInferIntegrationTest.RealInferExecution**
   - 実際のデータを使用したVAD推論の実行
   - エンドツーエンドの推論パイプラインを検証
   - 予測された軌道の出力を確認

### トラブルシューティング

#### よくあるエラー

1. **設定ファイルが見つからない**
   ```
   C++ exception with description "Failed to load config from YAML: bad file: ../test_config.yaml"
   ```
   **解決方法**: `test_config.yaml`を`build/autoware_tensorrt_vad/`ディレクトリにコピー

2. **lint系テストの失敗**
   ```
   copyright (Failed)
   cpplint (Failed)
   lint_cmake (Failed)
   uncrustify (Failed)
   ```
   **解決方法**: 機能テストのみ実行する場合は`--ctest-args -R test_vad_integration`オプションを使用

#### debug方法

##### 直接実行する

```bash
# ビルドディレクトリに移動
cd build/autoware_tensorrt_vad

# テストを直接実行
./test_vad_integration
```



### 注意事項

- **GMOCK WARNING**: テスト実行時に表示される警告は、期待されていないmock関数の呼び出しに関するもので、テストの成功には影響しません
- **実行時間**: テストは約2秒程度で完了します
- **メモリ使用量**: TensorRTエンジンのロードにより、一時的にGPUメモリを大量に使用します

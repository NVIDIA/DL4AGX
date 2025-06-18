# VadModel テストスイート

## 概要

このディレクトリには、DL4AGX内で定義された`VadModel`クラスの包括的なテストスイートが含まれています。これらのテストは、`VadModel`をautoware_tensorrt_vadパッケージに移行する際の品質保証として機能します。

## ファイル構成

```
test/
├── CMakeLists.txt           # テスト用のCMake設定
├── .gitignore               # ビルド成果物除外設定
├── README.md                # このファイル
├── mock_vad_logger.hpp      # テスト用モックロガー
├── test_vad_model.cpp       # 基本単体テスト
└── test_vad_integration.cpp # 統合テスト
```

## ビルドと実行

### 推奨方法（Out-of-sourceビルド）

```bash
# testディレクトリに移動
cd test

# buildディレクトリを作成
mkdir build
cd build

# CMakeで設定（TensorRTパスを指定）
cmake .. -DTRT_ROOT=/path/to/tensorrt

# ビルド実行
make vad_tests

# テスト実行
./vad_tests
```

### 他の実行方法

```bash
# makeターゲットを使用
make run_tests

# CTestを使用
ctest

# ヘルプの表示
make test_help
```

## テスト構成

### 1. `mock_vad_logger.hpp`

- テスト用のモックVadLoggerクラス
- ログメッセージを記録して、テストで確認可能
- `VadLogger`インターフェースの実装例

### 2. `test_vad_model.cpp`

基本的な単体テスト：
- **VadInputDataStructure**: 入力データ構造体のテスト
- **VadOutputDataStructure**: 出力データ構造体のテスト
- **LoggerFunctionality**: ロガー機能のテスト
- **VadConfigStructure**: 設定構造体のテスト
- **NetworkParamClass**: NetworkParamクラスのテスト
- **TensorRTLoggerClass**: TensorRT用Loggerクラスのテスト
- **VadModelTypeConstraints**: 型安全性のテスト
- **TrajectoryPostprocessAlgorithm**: 軌道計算アルゴリズムのテスト

### 3. `test_vad_integration.cpp`
統合テスト：
- **CreateRealisticInputData**: 現実的な入力データ生成のテスト
- **MultipleCommandTrajectoryTest**: 複数コマンドに対する軌道計算テスト
- **DetailedLoggingTest**: 詳細なログ出力テスト
- **ConfigurationValidationTest**: 設定検証テスト
- **MemoryUsageEstimationTest**: メモリ使用量推定テスト

## テストの制限事項

### 実際のTensorRTエンジンファイルが必要な機能

以下の機能は実際のTensorRTエンジンファイルが必要なため、現在のテストでは制限的にテストされています：

1. **VadModelの実際の初期化**: 
   - `create_runtime()`
   - `load_plugin()`
   - `init_engines()`

2. **実際の推論**:
   - `infer()` メソッドの完全なテスト
   - GPU メモリ操作
   - CUDA ストリーム処理

### テストされている機能

1. **データ構造**: すべての入出力データ構造が正しく動作
2. **ロガー機能**: ログの記録と取得が正常に動作
3. **設定管理**: 設定の検証と管理が正常に動作
4. **軌道計算アルゴリズム**: postprocessの計算ロジックが正確に動作
5. **型安全性**: テンプレート制約が正しく動作

## autoware_tensorrt_vadパッケージへの移行

これらのテストは、VadModelを`autoware_tensorrt_vad`パッケージに移行する際の品質保証として設計されています。

### 移行時のテスト活用方法

1. **移行前**: 現在のテストをすべて実行して基準を確立
2. **移行中**: 各コンポーネントの移行後にテストを実行
3. **移行後**: すべてのテストが通ることを確認

### 実際のエンジンファイルを使用したテスト

実際のTensorRTエンジンファイルがある環境では、以下のように設定を変更してより包括的なテストを実行できます：

```cpp
// test_vad_integration.cpp内で実際のパスに変更
config_.plugins_path = "/path/to/real/bevfusion_plugin.so";
NetConfig backbone_config;
backbone_config.engine_file = "/path/to/real/backbone.engine";
// ...
```

## トラブルシューティング

### よくある問題

1. **CUDA関連のエラー**:
   ```
   cannot open source file "cuda_runtime.h"
   ```
   解決策: CUDA Toolkitが正しくインストールされ、`CUDA_TOOLKIT_ROOT_DIR`が設定されていることを確認

2. **TensorRT関連のエラー**:
   ```
   cannot open source file "NvInfer.h"
   ```
   解決策: TensorRTが正しくインストールされ、`TRT_ROOT`環境変数が設定されていることを確認

3. **リンクエラー**:
   ```
   undefined reference to nvinfer1::createInferRuntime
   ```
   解決策: TensorRTライブラリのパスが正しく設定されていることを確認

### デバッグモード
```bash
# デバッグ情報付きでビルド
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
make vad_tests

# Valgrindを使用したメモリリークチェック（実際のGPU操作以外）
valgrind --leak-check=full ./test/vad_tests --gtest_filter="*Logger*"
```

## 今後の拡張

### 追加すべきテスト
1. **パフォーマンステスト**: 推論時間の測定
2. **負荷テスト**: 大量データでのメモリ使用量
3. **並行性テスト**: マルチスレッド環境での動作
4. **エラーハンドリングテスト**: 異常系のより詳細なテスト

### モックの改善
1. **MockTensorRTEngine**: 実際のエンジンファイルなしでの完全なテスト
2. **MockCUDA**: GPU操作のシミュレーション
3. **ConfigurationBuilder**: テスト用設定の簡単な生成 

## テストファイル構成

| ファイル名 | 説明 | テスト内容 |
|------------|------|------------|
| `test_vad_model.cpp` | VadModelクラスの基本機能テスト | コンストラクタ、初期化、ログ機能 |
| `test_vad_integration.cpp` | 統合テスト | 複数コンポーネント間の連携 |
| `test_vad_infer.cpp` | **NEW** infer機能の単体テスト | 入力検証、postprocess、メモリ使用量 |
| `test_vad_infer_integration.cpp` | **NEW** infer機能の統合テスト | 実際のGPU推論、性能測定、メモリリーク |
| `mock_vad_logger.hpp` | テスト用モックロガー | VadLoggerのテスト実装 |

## infer機能のテスト詳細

### `test_vad_infer.cpp` - 軽量単体テスト
**目的**: GPU/TensorRTに依存しない部分のテスト

**テストケース**:
1. **InputDataValidation** - 入力データの妥当性検証
   - カメラ画像サイズ（6×3×256×704）
   - シフトデータ（3要素）
   - LiDAR2Image変換行列（6×4×4）
   - CAN-BUSデータ（18要素）
   - コマンドインデックス（0-2範囲）

2. **MockInferWithoutGPU** - postprocessロジックのテスト
   - モック推論結果からの軌道生成
   - 累積和計算の正確性
   - 期待される出力形式の確認

3. **InvalidInputHandling** - エラーハンドリング
   - 無効なサイズの入力データ
   - 範囲外のコマンドインデックス

4. **MultipleFrameProcessing** - フレーム処理シミュレーション
   - 初回フレーム（head_no_prev）から継続フレーム（head）への遷移
   - 状態管理の正確性

5. **MemoryUsageValidation** - メモリ使用量検証
   - 各入力データのメモリサイズ計算
   - 総メモリ使用量のしきい値チェック（50MB制限）

### `test_vad_infer_integration.cpp` - 実環境統合テスト
**目的**: 実際のGPU/TensorRTを使用した完全なテスト

**前提条件**:
- TensorRTエンジンファイル: `backbone.engine`, `head.engine`, `head_no_prev.engine`
- BEVFusionプラグイン: `bevfusion_plugins.so`
- CUDA対応GPU

**テストケース**:
1. **PrerequisitesCheck** - 実行環境確認
   - エンジンファイルの存在確認
   - プラグインファイルの確認
   - CUDA GPUの利用可能性

2. **VadModelInitialization** - モデル初期化テスト
   - 実際のエンジンファイルを使用した初期化
   - プラグインロードの成功確認

3. **RealInferExecution** - 実際の推論実行
   - リアルなサイズの入力データで推論
   - 初回フレームと継続フレームの処理
   - 異なるコマンドでの軌道生成差異確認
   - 軌道点の合理性検証（100m範囲内）

4. **InferencePerformance** - 性能測定
   - ウォームアップ後の推論時間測定
   - 平均推論時間の期待値チェック（100ms以下）
   - 複数回実行での安定性確認

5. **MemoryLeakTest** - メモリリークテスト
   - 複数回のモデル作成・破棄サイクル
   - GPU メモリ使用量の監視
   - メモリ増加量の許容範囲チェック（100MB以内）

## 実行方法

### 基本テスト（GPU不要）
```bash
cd test/build
cmake .. -DTRT_ROOT=/path/to/tensorrt
make vad_tests
./vad_tests --gtest_filter="VadInferTest.*"
```

### 統合テスト（GPU + エンジンファイル必要）
```bash
# 環境変数設定（オプション）
export VAD_ENGINE_DIR=/path/to/engines
export VAD_PLUGIN_PATH=/path/to/bevfusion_plugins.so

# テスト実行
./vad_tests --gtest_filter="VadInferIntegrationTest.*"
```

**注意**: 統合テストは要件が満たされない場合、自動的にスキップされます。 
#ifndef MOCK_VAD_LOGGER_HPP_
#define MOCK_VAD_LOGGER_HPP_

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "autoware/tensorrt_vad/vad_model.hpp" // VadLoggerの定義を含む

namespace autoware::tensorrt_vad {

// nvinfer1::ILogger::Severity を短いエイリアスにする
using Severity = nvinfer1::ILogger::Severity;

/**
 * @class MockVadLogger
 * @brief VadLoggerのモッククラス。テスト中にログ出力をキャプチャし、検証するために使用します。
 */
class MockVadLogger : public VadLogger {
public:
    MockVadLogger() = default;
    ~MockVadLogger() override = default;

    // gmockを使ってVadLoggerのインターフェースをモック化
    MOCK_METHOD(void, debug, (const std::string& message), (override));
    MOCK_METHOD(void, info, (const std::string& message), (override));
    MOCK_METHOD(void, warn, (const std::string& message), (override));
    MOCK_METHOD(void, error, (const std::string& message), (override));
};

}  // namespace autoware::tensorrt_vad

#endif  // MOCK_VAD_LOGGER_HPP_

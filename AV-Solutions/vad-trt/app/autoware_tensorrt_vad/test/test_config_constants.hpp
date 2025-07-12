#ifndef AUTOWARE_TENSORRT_VAD_TEST_CONFIG_CONSTANTS_HPP_
#define AUTOWARE_TENSORRT_VAD_TEST_CONFIG_CONSTANTS_HPP_

#include <vector>
#include <string>
#include <filesystem>
#include <cstdlib>

namespace autoware::tensorrt_vad::test {

// default path for the test configuration file
inline const std::string DEFAULT_TEST_CONFIG_PATH = "../../install/autoware_tensorrt_vad/share/autoware_tensorrt_vad/test/test_config.yaml";

/**
 * @brief this func gets the path to the test configuration file
 * 
 * @return path to the test configuration file
 */
inline std::string getTestConfigPath() {
    return DEFAULT_TEST_CONFIG_PATH;
}

}  // namespace autoware::tensorrt_vad::test

#endif  // AUTOWARE_TENSORRT_VAD_TEST_CONFIG_CONSTANTS_HPP_

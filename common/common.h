/**************************************************************************
 * Copyright (c) 2018-2019 NVIDIA Corporation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * File: DL4AGX/common/common.h
 * Description: Aggrigate other common files and general file operation utils
 *************************************************************************/
#pragma once
#include "common/argparser.h"
#include "common/datasets/coco/cocoJSON.h"
#include "common/image/PGMUtils.h"
#include "common/image/PPMUtils.h"
#include "common/macros.h"
#include "common/tensorrt/Logger.h"
#include "common/tensorrt/Profiler.h"
#include "common/tensorrt/utils.h"
#include "common/timers.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <new>
#include <numeric>
#include <ratio>
#include <string>
#include <utility>
#include <vector>

namespace common
{
inline void loadEngine(std::string filename, std::string& data)
{

    std::ifstream file(filename.c_str(), std::ios::binary);
    if (file.good())
    {
        std::stringstream buffer;
        buffer << file.rdbuf();
        data = buffer.str();
        file.close();
    }
    else
    {
        std::cerr << "Engine file not valid" << std::endl;
        return;
    }
}

// Locate path to file, given its filename or filepath suffix and possible dirs it might lie in
// Function will also walk back MAX_DEPTH dirs from CWD to check for such a file path
inline std::string locateFile(const std::string& filepathSuffix, const std::vector<std::string>& directories)
{
    const int MAX_DEPTH{10};
    bool found{false};
    std::string filepath;

    for (auto& dir : directories)
    {
        filepath = dir + filepathSuffix;

        for (int i = 0; i < MAX_DEPTH && !found; i++)
        {
            std::ifstream checkFile(filepath);
            found = checkFile.is_open();
            if (found)
                break;
            filepath = "../" + filepath; // Try again in parent dir
        }

        if (found)
        {
            break;
        }

        filepath.clear();
    }

    if (filepath.empty())
    {
        std::string directoryList = std::accumulate(directories.begin() + 1, directories.end(), directories.front(),
                                                    [](const std::string& a, const std::string& b) { return a + "\n\t" + b; });
        throw std::runtime_error("Could not find " + filepathSuffix + " in data directories:\n\t" + directoryList);
    }
    return filepath;
}

inline void* safeCudaMalloc(size_t memSize)
{
    void* deviceMem;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}

template <class Iter>
inline std::vector<size_t> argsort(Iter begin, Iter end, bool reverse = false)
{
    std::vector<size_t> inds(end - begin);
    std::iota(inds.begin(), inds.end(), 0);
    if (reverse)
    {
        std::sort(inds.begin(), inds.end(), [&begin](size_t i1, size_t i2) {
            return begin[i2] < begin[i1];
        });
    }
    else
    {
        std::sort(inds.begin(), inds.end(), [&begin](size_t i1, size_t i2) {
            return begin[i1] < begin[i2];
        });
    }
    return inds;
}

//...LG returns top K indices, not values.
template <typename T>
inline std::vector<size_t> topK(const std::vector<T> inp, const size_t k)
{
    std::vector<size_t> result;
    std::vector<size_t> inds = common::argsort(inp.cbegin(), inp.cend(), true);
    result.assign(inds.begin(), inds.begin() + k);
    return result;
}

inline std::string getFileType(const std::string& filepath)
{
    return filepath.substr(filepath.find_last_of(".") + 1);
}

inline std::string toLower(const std::string& inp)
{
    std::string out = inp;
    std::transform(out.begin(), out.end(), out.begin(), ::tolower);
    return out;
}
} // namespace common

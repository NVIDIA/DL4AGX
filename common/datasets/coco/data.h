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
 * File: DL4AGX/common/datasets/coco/data.h
 * Description: Struct to hold coco data
 *************************************************************************/
#pragma once
#include "third_party/json/json.hpp"
#include <assert.h>
#include <stdint.h>
#include <vector>

using json = nlohmann::json;

namespace common
{
namespace datasets
{
namespace coco
{
namespace data
{
typedef struct
{
    int id;
    int width;
    int height;
    std::string file_name;
    int license;
    std::string flickr_url;
    std::string coco_url;
    std::string date_captured;
} image;

void to_json(json& j, const image& i);
void from_json(const json& j, image& i);
} // namespace data
} // namespace coco
} // namespace datasets
} // namespace common

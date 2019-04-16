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
 * File: DL4AGX/common/datasets/coco/results.cpp
 * Description: De/Serialize Results struct to/from JSON
 *************************************************************************/
#include "cocoJSON.h"
#include "third_party/json/json.hpp"
#include <assert.h>
#include <stdint.h>
#include <vector>

using json = nlohmann::json;
using namespace common::datasets::coco;

void results::to_json(json& j, const objectDetection& i)
{
    j = json{{"image_id", i.image_id},
             {"category_id", i.category_id},
             {"bbox", {i.bbox[0], i.bbox[1], i.bbox[2], i.bbox[3]}},
             {"score", i.score}};
}

void results::from_json(const json& j, objectDetection& i)
{
    assert(0); // NOT IMPLEMENTED
}
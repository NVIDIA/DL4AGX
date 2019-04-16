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
 * File: DL4AGX/common/datasets/coco/data.cpp
 * Description: De/Serialize Data struct to/from JSON
 *************************************************************************/
#include "cocoJSON.h"
#include "third_party/json/json.hpp"
#include <assert.h>
#include <stdint.h>
#include <vector>

using json = nlohmann::json;
using namespace common::datasets::coco;

void data::to_json(json& j, const image& i)
{
    j = json{{"id", i.id},
             {"width", i.width},
             {"height", i.height},
             {"file_name", i.file_name},
             {"license", i.license},
             {"flickr_url", i.flickr_url},
             {"coco_url", i.coco_url},
             {"date_captured", i.date_captured}};
}

void data::from_json(const json& j, image& i)
{
    i.id = j.at("id").get<int>();
    i.width = j.at("width").get<int>();
    i.height = j.at("height").get<int>();
    i.file_name = j.at("file_name").get<std::string>();
    //i.license = j.at("license").get<int>();
    //i.flickr_url = j.at("flickr_url").get<std::string>();
    //i.coco_url = j.at("coco_url").get<std::string>();
    //i.date_captured = j.at("date_captured").get<std::string>();
}

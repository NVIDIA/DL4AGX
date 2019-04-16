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
 * File: DL4AGX/common/argparser.h
 * Description: An argument parser for CLIs
 *************************************************************************/
#pragma once
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace common
{
namespace argparser
{
inline bool parseBool(const char* arg, const char* longName, bool& value, char shortName = 0)
{
    bool match = false;

    if (shortName)
    {
        match = (arg[0] == '-') && (arg[1] == shortName);
    }
    if (!match && longName)
    {
        const size_t n = strlen(longName);
        match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, longName, n);
    }
    if (match)
    {
        value = true;
        if (shortName)
        {
            std::cout << shortName << ": " << value << std::endl;
        }
        else if (longName)
        {
            std::cout << longName << ": " << value << std::endl;
        }
    }
    return match;
}

inline bool parseString(const char* arg, const char* name, std::string& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match)
    {
        value = arg + n + 3;
        std::cout << name << ": " << value << std::endl;
    }
    return match;
}

inline bool parseInt(const char* arg, const char* name, int& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match)
    {
        value = atoi(arg + n + 3);
        std::cout << name << ": " << value << std::endl;
    }
    return match;
}

inline bool parseFloat(const char* arg, const char* name, float& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match)
    {
        value = atof(arg + n + 3);
        std::cout << name << ": " << value << std::endl;
    }
    return match;
}

inline std::vector<std::string> split(const std::string& s, char delim)
{
    std::vector<std::string> res;
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim))
    {
        res.push_back(item);
    }
    return res;
}
} // namespace argparser
} // namespace common

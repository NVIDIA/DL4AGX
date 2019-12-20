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
#include <getopt.h>

namespace common
{
namespace argparser
{

struct StandardArgs
{
    bool runInInt8{false};
    bool runInFp16{false};
    bool help{false};
    int useDLACore{-1};
    int batch{1};
    std::vector<std::string> dataDirs;
    bool useILoop{false};
};
    
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

    
//!
//! \brief Populates the Args struct with the provided command-line parameters.
//!
//! \throw invalid_argument if any of the arguments are not valid
//!
//! \return boolean If return value is true, execution can continue, otherwise program should exit
//!
inline bool parseArgs(StandardArgs& args, int argc, char* argv[])
{
    while (1)
    {
        int arg;
        static struct option long_options[] = {{"help", no_argument, 0, 'h'}, {"datadir", required_argument, 0, 'd'},
            {"int8", no_argument, 0, 'i'}, {"fp16", no_argument, 0, 'f'}, {"useILoop", no_argument, 0, 'l'},
            {"useDLACore", required_argument, 0, 'u'}, {"batch", required_argument, 0, 'b'}, {nullptr, 0, nullptr, 0}};
        int option_index = 0;
        arg = getopt_long(argc, argv, "hd:iu", long_options, &option_index);
        if (arg == -1)
        {
            break;
        }

        switch (arg)
        {
        case 'h': args.help = true; return true;
        case 'd':
            if (optarg)
            {
                args.dataDirs.push_back(optarg);
            }
            else
            {
                std::cerr << "ERROR: --datadir requires option argument" << std::endl;
                return false;
            }
            break;
        case 'i': args.runInInt8 = true; break;
        case 'f': args.runInFp16 = true; break;
        case 'l': args.useILoop = true; break;
        case 'u':
            if (optarg)
            {
                args.useDLACore = std::stoi(optarg);
            }
            break;
        case 'b':
            if (optarg)
            {
                args.batch = std::stoi(optarg);
            }
            break;
        default: return false;
        }
    }
    return true;
}
} // namespace argparser
} // namespace common

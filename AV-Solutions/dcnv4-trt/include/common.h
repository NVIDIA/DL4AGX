/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#if DEBUG
#define print_log(...) {\
    char str[100];\
    sprintf(str, __VA_ARGS__);\
    std::cout << "CUSTOM PLUGIN TRACE----> call " << "[" \
              << __FILE__ << ":" << __LINE__ \
              << ", " << __FUNCTION__ << " " << str << std::endl;\
    }
#else
#define print_log(...)
#endif

#if __CUDACC__
#define __hostdev__ __host__ __device__
#else
#define __hostdev__
#endif // __CUDACC__

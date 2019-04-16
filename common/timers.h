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
 * File: DL4AGX/common/timers.h
 * Description: Timer implementations
 *************************************************************************/
#pragma once
#include "common/macros.h"
#include <chrono>
#include <cuda_runtime_api.h>

namespace common
{
namespace timers
{
class TimerBase
{
public:
    virtual void start() {}
    virtual void stop() {}
    float microseconds() const noexcept { return mMs * 1000.f; }
    float milliseconds() const noexcept { return mMs; }
    float seconds() const noexcept { return mMs / 1000.f; }
    void reset() noexcept { mMs = 0.f; }

protected:
    float mMs{0.0f};
};

class GpuTimer : public TimerBase
{
public:
    GpuTimer(cudaStream_t stream)
        : mStream(stream)
    {
        CHECK(cudaEventCreate(&mStart));
        CHECK(cudaEventCreate(&mStop));
    }
    ~GpuTimer()
    {
        CHECK(cudaEventDestroy(mStart));
        CHECK(cudaEventDestroy(mStop));
    }
    void start() { CHECK(cudaEventRecord(mStart, mStream)); }
    void stop()
    {
        CHECK(cudaEventRecord(mStop, mStream));
        float ms{0.0f};
        CHECK(cudaEventSynchronize(mStop));
        CHECK(cudaEventElapsedTime(&ms, mStart, mStop));
        mMs += ms;
    }

private:
    cudaEvent_t mStart, mStop;
    cudaStream_t mStream;
}; // class GpuTimer

template <typename Clock>
class CpuTimer : public TimerBase
{
public:
    using clock_type = Clock;

    void start() { mStart = Clock::now(); }
    void stop()
    {
        mStop = Clock::now();
        mMs += std::chrono::duration<float, std::milli>{mStop - mStart}.count();
    }

private:
    std::chrono::time_point<Clock> mStart, mStop;
}; // class CpuTimer

using PreciseCpuTimer = CpuTimer<std::chrono::high_resolution_clock>;
} // namespace timers
} // namespace common
/***************************************************************************************************
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
 * Project: MultiDeviceInferencePipeline > Inference 
 * 
 * File: DL4AGX/MultiDeviceInferencePipeline/inference/utils/profiling.h
 * 
 * Description: CUPTI Profiling utilites
 ***************************************************************************************************/
#pragma once
#include "common/cuptiUtils.h"

using namespace common;

namespace multideviceinferencpipelines
{
namespace inference
{
namespace utils
{
namespace profiling
{
const std::vector<std::string> collectedMetrics{"sm_efficiency", "flop_count_sp", "dram_utilization"};
CUpti_SubscriberHandle cuptiSubscriber;
CUcontext cuptiCtx;
cuptiUtils::MetricData_t metricData;

inline bool enableProfiling()
{
    char deviceName[32];
    int deviceCount;
    int deviceNum;

    std::cout << "DALI pipeline will be profiled for GPU execution." << std::endl;

    // make sure activity is enabled before any CUDA API
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));

    DRIVER_API_CALL(cuInit(0));
    DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));
    if (deviceCount == 0)
    {
        cout << "There is no device supporting CUDA." << endl;
        return false;
    }

    DRIVER_API_CALL(cuDeviceGet(&cuptiUtils::device, deviceNum));
    DRIVER_API_CALL(cuDeviceGetName(deviceName, 32, cuptiUtils::device));
    printf("CUDA Device Name: %s\n", deviceName);

    DRIVER_API_CALL(cuDevicePrimaryCtxRetain(&cuptiCtx, cuptiUtils::device));
    return true;
}

inline void setupCuptiSubscriptions()
{
    CUPTI_CALL(cuptiActivityFlushAll(0));
    // setup launch callback for event collection
    CUPTI_CALL(cuptiSubscribe(&cuptiSubscriber, (CUpti_CallbackFunc) cuptiUtils::getMetricValueCallback, &metricData));
    CUPTI_CALL(cuptiEnableCallback(1, cuptiSubscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                   CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
    CUPTI_CALL(cuptiEnableCallback(1, cuptiSubscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                   CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));

    // allocate space to hold all the events needed for the metric
    CUPTI_CALL(cuptiMetricGetIdFromName(cuptiUtils::device, cuptiUtils::metricName, &cuptiUtils::metricId));
    CUPTI_CALL(cuptiMetricGetNumEvents(cuptiUtils::metricId, &metricData.numEvents));
    metricData.device = cuptiUtils::device;

    metricData.eventIdArray.resize(metricData.numEvents);
    metricData.eventValueArray.resize(metricData.numEvents);
    metricData.eventIdx = 0;
    CUPTI_CALL(cuptiMetricCreateEventGroupSets(cuptiCtx, sizeof(cuptiUtils::metricId), &cuptiUtils::metricId, &cuptiUtils::passData));
}

inline void cleanupCuptiSubscriptions()
{
    CUPTI_CALL(cuptiEnableCallback(0, cuptiSubscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                   CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
    CUPTI_CALL(cuptiEnableCallback(0, cuptiSubscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                   CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));
    CUPTI_CALL(cuptiUnsubscribe(cuptiSubscriber));
}
} // namespace profiling
} // namespace utils
} // namespace inference
} // namespace multideviceinferencepipeline
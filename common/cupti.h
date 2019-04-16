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
 * File: DL4AGX/common/cupti.h
 * Description: Utility functions for cupti
 *************************************************************************/
#pragma once
#include "cuda.h"
#include "cupti.h"
#include "nvToolsExt.h"
#include <iostream>

#define DRIVER_API_CALL(apiFuncCall)                                                    \
    do                                                                                  \
    {                                                                                   \
        CUresult _status = apiFuncCall;                                                 \
        if (_status != CUDA_SUCCESS)                                                    \
        {                                                                               \
            std::cerr << __FILE__ << ":" << __LINE__ << " error: function "             \
                      << #apiFuncCall << " failed with error " << _status << std::endl; \
            return (-1);                                                                \
        }                                                                               \
    } while (0)

#define CUPTI_CALL(call)                                                        \
    do                                                                          \
    {                                                                           \
        CUptiResult _status = call;                                             \
        if (_status != CUPTI_SUCCESS)                                           \
        {                                                                       \
            const char* errstr;                                                 \
            cuptiGetResultString(_status, &errstr);                             \
            std::cerr << __FILE__ << ":" << __LINE__ << " error: function "     \
                      << #call << " failed with error " << errstr << std::endl; \
            exit(-1);                                                           \
        }                                                                       \
    } while (0)

#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align) \
    (((uintptr_t)(buffer) & ((align) -1)) ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align) -1))) : (buffer))

namespace common
{
namespace cupti
{
// User data for event collection callback
typedef struct MetricData_st
{
    // the device where metric is being collected
    CUdevice device;
    // the set of event groups to collect for a pass
    CUpti_EventGroupSet* eventGroups;
    // the current number of events collected in eventIdArray and
    // eventValueArray
    uint32_t eventIdx;
    // the number of entries in eventIdArray and eventValueArray
    uint32_t numEvents;
    // array of event ids
    std::vector<CUpti_EventID> eventIdArray;
    // array of event values
    std::vector<uint64_t> eventValueArray;
    // Timestamp collection holders
    uint64_t startTimestamp;
    uint64_t endTimestamp;
} MetricData_t;

static uint64_t kernelDuration;
const char* metricName;
CUpti_MetricID metricId;
CUpti_EventGroupSets* passData;
CUpti_MetricValue metricValue;
CUdevice device = 0; //need to check for device once code is integrated

void CUPTIAPI
getMetricValueCallback(void* userdata, CUpti_CallbackDomain domain,
                       CUpti_CallbackId cbid, const CUpti_CallbackData* cbInfo)
{
    uint64_t startTimestamp;
    uint64_t endTimestamp;

    MetricData_t* metricData = (MetricData_t*) userdata;
    unsigned int i, j, k;
    // This callback is enabled only for launch so we shouldn't see
    // anything else.
    if ((cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) && (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000))
    {
        std::cout << __FILE__ << ":" << __LINE__ << ": unexpected cbid " << cbid << std::endl;
        return;
    }

    // on entry, enable all the event groups being collected this pass,
    // for metrics we collect for all instances of the event
    if (cbInfo->callbackSite == CUPTI_API_ENTER)
    {
        cudaDeviceSynchronize();

        // Collect timestamp for API start
        CUPTI_CALL(cuptiDeviceGetTimestamp(cbInfo->context, &startTimestamp));
        metricData->startTimestamp = startTimestamp;

        CUPTI_CALL(cuptiSetEventCollectionMode(cbInfo->context,
                                               CUPTI_EVENT_COLLECTION_MODE_KERNEL));

        for (i = 0; i < metricData->eventGroups->numEventGroups; i++)
        {
            uint32_t all = 1;
            CUPTI_CALL(cuptiEventGroupSetAttribute(metricData->eventGroups->eventGroups[i],
                                                   CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
                                                   sizeof(all), &all));
            CUPTI_CALL(cuptiEventGroupEnable(metricData->eventGroups->eventGroups[i]));
        }
    }
    // on exit, read and record event values
    if (cbInfo->callbackSite == CUPTI_API_EXIT)
    {
        cudaDeviceSynchronize();

        // Collect timestamp for API exit
        CUPTI_CALL(cuptiDeviceGetTimestamp(cbInfo->context, &endTimestamp));

        metricData->endTimestamp = endTimestamp;

        // for each group, read the event values from the group and record
        // in metricData
        for (i = 0; i < metricData->eventGroups->numEventGroups; i++)
        {
            CUpti_EventGroup group = metricData->eventGroups->eventGroups[i];
            CUpti_EventDomainID groupDomain;
            uint32_t numEvents, numInstances, numTotalInstances;
            CUpti_EventID* eventIds;
            size_t groupDomainSize = sizeof(groupDomain);
            size_t numEventsSize = sizeof(numEvents);
            size_t numInstancesSize = sizeof(numInstances);
            size_t numTotalInstancesSize = sizeof(numTotalInstances);
            uint64_t normalized, sum;
            size_t valuesSize, eventIdsSize;

            CUPTI_CALL(cuptiEventGroupGetAttribute(group,
                                                   CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID,
                                                   &groupDomainSize, &groupDomain));
            CUPTI_CALL(cuptiDeviceGetEventDomainAttribute(metricData->device, groupDomain,
                                                          CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT,
                                                          &numTotalInstancesSize, &numTotalInstances));
            CUPTI_CALL(cuptiEventGroupGetAttribute(group,
                                                   CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
                                                   &numInstancesSize, &numInstances));
            CUPTI_CALL(cuptiEventGroupGetAttribute(group,
                                                   CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
                                                   &numEventsSize, &numEvents));
            eventIdsSize = numEvents * sizeof(CUpti_EventID);
            eventIds = new CUpti_EventID[eventIdsSize];
            CUPTI_CALL(cuptiEventGroupGetAttribute(group,
                                                   CUPTI_EVENT_GROUP_ATTR_EVENTS,
                                                   &eventIdsSize, eventIds));

            std::vector<uint64_t> values(numInstances);
            valuesSize = values.size() * sizeof(uint64_t);

            //Uncomment this if events information is needed
            //std::cout << "Number of events : " << numEvents << std::endl;
            for (j = 0; j < numEvents; j++)
            {
                CUPTI_CALL(cuptiEventGroupReadEvent(group, CUPTI_EVENT_READ_FLAG_NONE,
                                                    eventIds[j], &valuesSize, values.data()));
                if (metricData->eventIdx >= metricData->numEvents)
                {
                    cerr << "error: too many events collected, metric expects only " << (int) metricData->numEvents << std::endl;
                    return;
                }

                // sum collect event values from all instances
                sum = 0;
                for (k = 0; k < numInstances; k++)
                    sum += values[k];

                // normalize the event value to represent the total number of
                // domain instances on the device
                normalized = (sum * numTotalInstances) / numInstances;

                metricData->eventIdArray[metricData->eventIdx] = eventIds[j];
                metricData->eventValueArray[metricData->eventIdx] = normalized;
                metricData->eventIdx++;

                // print collected value
                {
                    char eventName[128];
                    size_t eventNameSize = sizeof(eventName) - 1;
                    CUPTI_CALL(cuptiEventGetAttribute(eventIds[j], CUPTI_EVENT_ATTR_NAME,
                                                      &eventNameSize, eventName));
                    eventName[127] = '\0';
                    // Uncomment this block below if events information is needed
                    /*
                    std::cout << "	" << eventName << " = " << (unsigned long long) sum << " (";
                    if (numInstances > 1)
                    {
                        for (k = 0; k < numInstances; k++)
                        {
                            if (k != 0)
                                std::cout << ", ";
                            std::cout << (unsigned long long) values[k];
                        }
                    }

                    std::cout << ")" << std::endl;
                    std::cout << "	" << eventName << " (normalized) (" << (unsigned long long) sum << " * " << numTotalInstances << ") / " << numInstances << " = " << (unsigned long long) normalized << std::endl;*/
                }
            }
        }

        if (metricData->eventIdx != metricData->numEvents)
        {
            cout << "error: expected " << metricData->numEvents << " metric events, got " << metricData->eventIdx << endl;
            exit(-1);
        }

        CUPTI_CALL(cuptiMetricGetValue(device, metricId, metricData->numEvents * sizeof(CUpti_EventID), metricData->eventIdArray.data(), metricData->numEvents * sizeof(uint64_t), metricData->eventValueArray.data(), kernelDuration, &metricValue));

        // print metric value, we format based on the value kind

        CUpti_MetricValueKind valueKind;
        size_t valueKindSize = sizeof(valueKind);
        CUPTI_CALL(cuptiMetricGetAttribute(metricId, CUPTI_METRIC_ATTR_VALUE_KIND,
                                           &valueKindSize, &valueKind));
        std::cout << "=================================================================================" << std::endl;
        std::cout << "Profiling Op / Kernel: " << cbInfo->symbolName << std::endl;
        switch (valueKind)
        {
        case CUPTI_METRIC_VALUE_KIND_DOUBLE:
            std::cout << "Metric " << metricName << " = " << metricValue.metricValueDouble << std::endl;
            ;
            break;
        case CUPTI_METRIC_VALUE_KIND_UINT64:
            std::cout << "Metric " << metricName << " = " << metricValue.metricValueUint64 << std::endl;
            break;
        case CUPTI_METRIC_VALUE_KIND_INT64:
            std::cout << "Metric " << metricName << " =  " << metricValue.metricValueInt64 << std::endl;
            break;
        case CUPTI_METRIC_VALUE_KIND_PERCENT:
            std::cout << "Metric " << metricName << " =  " << metricValue.metricValuePercent << std::endl;
            break;
        case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
            std::cout << "Metric " << metricName << " = " << metricValue.metricValueThroughput << " bytes/sec" << std::endl;
            break;
        case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
            std::cout << "Metric " << metricName << (unsigned int) metricValue.metricValueUtilizationLevel << " = utilization level " << std::endl;
            break;
        default:
            fprintf(stderr, "error: unknown value kind\n");
            exit(-1);
        }
        std::cout << std::fixed << std::setprecision(3) << "Execution Time for Kernel: " << (metricData->endTimestamp - metricData->startTimestamp) / 1000000.0 << " ms" << endl;

        for (i = 0; i < metricData->eventGroups->numEventGroups; i++)
            CUPTI_CALL(cuptiEventGroupDisable(metricData->eventGroups->eventGroups[i]));
        metricData->eventIdx = 0;
    }
}

static void CUPTIAPI
bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords)
{
    uint8_t* rawBuffer;

    //size of the activity buffer used by CUPTI to store activity records
    *size = 32 * 2048;
    rawBuffer = new uint8_t[*size + ALIGN_SIZE];
    //std::vector<uint8_t> rawBuffer(*size + ALIGN_SIZE);

    *buffer = ALIGN_BUFFER(rawBuffer, ALIGN_SIZE);
    *maxNumRecords = 0;

    if (*buffer == NULL)
    {
        std::cout << "Error: out of memory " << std::endl;
        return;
    }
}

static void CUPTIAPI
bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t* buffer, size_t size, size_t validSize)
{
    CUpti_Activity* record = NULL;
    CUpti_ActivityKernel4* kernel;

    CUPTI_CALL(cuptiActivityGetNextRecord(buffer, validSize, &record));

    kernel = (CUpti_ActivityKernel4*) record;
    if (kernel->kind != CUPTI_ACTIVITY_KIND_KERNEL)
    {
        cerr << "Error: expected kernel activity record, got  " << (int) kernel->kind << std::endl;
        return;
    }

    kernelDuration = kernel->end - kernel->start;
    delete[] buffer;
}
}
}

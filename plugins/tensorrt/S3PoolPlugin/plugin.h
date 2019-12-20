#ifndef TRT_CUSTOM_PLUGIN_H
#define TRT_CUSTOM_PLUGIN_H

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <iostream>
#include <cstring>

namespace nvinfer1
{
namespace plugin
{

class BaseCreator : public IPluginCreator
{
public:
    void setPluginNamespace(const char* libNamespace) override
    {   
        mNamespace = libNamespace;
    }
    
    const char* getPluginNamespace() const override
    {   
        return mNamespace.c_str();
    }

protected:
    std::string mNamespace = "";
};
}
}

#endif

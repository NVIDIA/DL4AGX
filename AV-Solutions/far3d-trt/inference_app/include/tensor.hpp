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

#ifndef NV_TENSOR_HPP
#define NV_TENSOR_HPP
#include <NvInferRuntime.h>

#include <array>
#include <exception>
#include <memory>
#include <stdexcept>

namespace nv
{
    size_t numel(const nvinfer1::Dims& dims);
    void cudaFreeWrapper(void* ptr);
    void cudaFreeHostWrapper(void* ptr);

    inline int reverseIndex(int idx, const int size)
    {
        if(idx < 0)
        {
            idx = size + idx;
        }
        return idx;
    }

    template<size_t N>
    size_t numel(const std::array<int32_t, N> dims, int start = 0, int end = -1)
    {
        size_t output = 1;
        end = reverseIndex(end, N);
        for(int32_t i = start; i <= end; ++i)
        {
            output *= dims[i];
        }
        return output;
    }

    class GPU;
    class CPU;

    class GPU
    {
    public:
        using Other_t = CPU;
        template<class T>
        static std::shared_ptr<T> allocate(size_t size)
        {
            T* ptr = nullptr;
            cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&ptr), size * sizeof(T));
            if(err != cudaSuccess)
            {
                return nullptr;
            }
            return std::shared_ptr<T>(ptr, &cudaFreeWrapper);
        }
        
    };

    class CPU
    {
    public:
        using Other_t = GPU;
        template<class T>
        static std::shared_ptr<T> allocate(size_t size)
        {
            T* ptr = nullptr;
            cudaError_t err = cudaMallocHost(reinterpret_cast<void**>(&ptr), size * sizeof(T));
            if(err != cudaSuccess)
            {
                return nullptr;
            }
            return std::shared_ptr<T>(ptr, &cudaFreeHostWrapper);
        }
    };

    template<class ... ARGS>
    auto makeShape(ARGS&&... args) -> std::array<int32_t, sizeof...(ARGS)>{
        return std::array<int32_t, sizeof...(ARGS)>{std::forward<ARGS>(args)...};
    }

    template<class T, class LAYOUT, class XPU = GPU>
    class Tensor;
    template<class T, class LAYOUT, class XPU = GPU>
    class OwningTensor;

    template<class T, class LAYOUT, class XPU>
    class Tensor
    {
    public:
        using CPU_t = Tensor<T, LAYOUT, CPU>;
        using GPU_t = Tensor<T, LAYOUT, GPU>;
        static constexpr const uint8_t RANK = LAYOUT::D;

        Tensor(T* data = nullptr, const std::array<int32_t, RANK>& dims = {}):
            m_data{data},
            m_dims{dims}
        {}

        T* getData() { return m_data; }
        const T* getData() const { return m_data; }

        const std::array<int32_t, RANK> getShape() const {return m_dims;}

        void reshape(T* data, const std::array<int32_t, RANK>& dims)
        {
            m_dims = dims;
            m_data = data;
        }

        size_t numel(int start = 0, int end = -1) const{return nv::template numel<RANK>(m_dims, start, end);}
        size_t bytes() const{return this->numel() * sizeof(T); }

        
        bool copyFrom(const Tensor<const T, LAYOUT, typename XPU::Other_t>& other, cudaStream_t stream)
        {
            if(other.bytes() == this->bytes())
            {
                // This overload is only called when OTHER_XPU != XPU and thus it is always host to device or device to host
                const cudaMemcpyKind kind = std::is_same<XPU, GPU>::value ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
                cudaError_t err = cudaMemcpyAsync(this->getData(), other.getData(), other.bytes(), kind);
                return err == cudaSuccess;
            } else {
                // TODO error handling
                return false;
            }
        }

        
        bool copyFrom(const Tensor<const T, LAYOUT, XPU>& other, cudaStream_t stream)
        {
            if(other.bytes() == this->bytes())
            {
                const cudaMemcpyKind kind = std::is_same<XPU, CPU>::value ? cudaMemcpyKind::cudaMemcpyHostToHost : cudaMemcpyKind::cudaMemcpyDeviceToDevice;
                cudaError_t err = cudaMemcpyAsync(this->getData(), other.getData(), other.bytes(), kind);
                return err == cudaSuccess;
            } else {
                // TODO error handling
                return false;
            }
        }


    private:
        T* m_data;
        std::array<int32_t, RANK> m_dims;
    };

    template<class T, class LAYOUT, class XPU>
    class Tensor<const T, LAYOUT, XPU>
    {
    public:
        using CPU_t = Tensor<const T, LAYOUT, CPU>;
        using GPU_t = Tensor<const T, LAYOUT, GPU>;
        static constexpr const uint8_t RANK = LAYOUT::D;
        Tensor(const Tensor<T, LAYOUT, XPU>& other):
            m_data{other.getData()}, m_dims{other.getShape()}
        {
        }

        Tensor(const OwningTensor<T, LAYOUT, XPU>& other):
            m_data{other.getData()}, m_dims{other.getShape()}
        {
        }

        Tensor(const T* data = nullptr, const std::array<int32_t, RANK>& dims = {});

        const T* getData() const { return m_data; }

        const std::array<int32_t, RANK> getShape() const {return m_dims;}
        size_t numel() const{return nv::template numel<RANK>(m_dims);}
        size_t bytes() const{return this->numel() * sizeof(T); }

        void reshape(const T* data, const std::array<int32_t, RANK>& dims)
        {
            m_dims = dims;
            m_data = data;
        }


    private:
        const T* m_data;
        std::array<int32_t, RANK> m_dims;
    };

    template<class T, class LAYOUT, class XPU>
    class OwningTensor: public Tensor<T, LAYOUT, XPU>
    {
    public:
        using Super_t = Tensor<T, LAYOUT, XPU>;
        using CPU_t = OwningTensor<T, LAYOUT, CPU>;
        using GPU_t = OwningTensor<T, LAYOUT, GPU>;

        OwningTensor(const std::array<int32_t, Super_t::RANK>& dims)
        {
            this->reshape(dims);
        }

        OwningTensor(const nvinfer1::Dims& dims)
        {
            this->reshape(dims);
        }

        OwningTensor(std::shared_ptr<T> data = {}, const std::array<int32_t, Super_t::RANK>& dims = {}):
            Super_t{data.get(), dims},
            m_data_ptr{std::move(data)}
        {
        }

        void reshape(std::shared_ptr<T> data, const std::array<int32_t, Super_t::RANK>& dims)
        {
            m_data_ptr = std::move(data);
            Super_t::reshape(m_data_ptr.get(), dims);
        }

        void reshape(const std::array<int32_t, Super_t::RANK>& dims)
        {
            const size_t size = numel(dims);
            const size_t current_size = this->numel();
            if(size > current_size)
            {
                m_data_ptr = XPU::template allocate<T>(size);
            }
            
            Super_t::reshape(m_data_ptr.get(), dims);
        }


        void reshape(const nvinfer1::Dims& dims)
        {
            if(Super_t::RANK +1 == dims.nbDims)
            {
                if(dims.d[0] != 1)
                {
                    throw std::runtime_error("Expected tensorrt batch dimension of 1");
                }
            }
            std::array<int, Super_t::RANK> shape;
            for(int i = 0; i < Super_t::RANK; ++i)
            {
                if(Super_t::RANK == dims.nbDims)
                {
                    shape[i] = dims.d[i];
                }else if(Super_t::RANK +1 == dims.nbDims)
                {
                    // Assumption here is that we are omitting the batch dimension
                    // While tensorrt has a batch dim of 1 which is checked above
                    shape[i] = dims.d[i+1];
                }
            }
            this->reshape(shape);
        }

        bool copyFrom(const Tensor<const T, LAYOUT, typename XPU::Other_t>& other, cudaStream_t stream)
        {
            this->reshape(other.getShape());
            // This overload is only called when OTHER_XPU != XPU and thus it is always host to device or device to host
            const cudaMemcpyKind kind = std::is_same<XPU, GPU>::value ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
            cudaError_t err = cudaMemcpyAsync(this->getData(), other.getData(), other.bytes(), kind);
            return err == cudaSuccess;
        }

        
        bool copyFrom(const Tensor<const T, LAYOUT, XPU>& other, cudaStream_t stream)
        {
            this->reshape(other.getShape());
            const cudaMemcpyKind kind = std::is_same<XPU, CPU>::value ? cudaMemcpyKind::cudaMemcpyHostToHost : cudaMemcpyKind::cudaMemcpyDeviceToDevice;
            cudaError_t err = cudaMemcpyAsync(this->getData(), other.getData(), other.bytes(), kind);
            return err == cudaSuccess;
        }
    private:
        std::shared_ptr<T> m_data_ptr;
    };
}

#endif // NV_TENSOR_HPP
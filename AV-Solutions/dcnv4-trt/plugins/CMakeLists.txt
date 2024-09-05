set(PLUGIN_LIST dcnv4FusePlugin dcnv4Plugin)
foreach(P ${PLUGIN_LIST})
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${P})
endforeach()

if(${USE_PTX})
    add_definitions(-DUSE_PTX -DPTX_ARCH=${PTX_ARCH})
    set(DEPS ${DEPS} libnvptxcompiler_static.a)

    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/dcnv4PtxPlugin)
endif()

message(DCNV4_plugin_cu_SRCS ${DCNV4_plugin_cu_SRCS})

cuda_add_library(DCNv4_plugin_cu ${DCNV4_plugin_cu_SRCS}                 
                 OPTIONS -arch=sm_${ARCH})
set_target_properties(DCNv4_plugin_cu PROPERTIES CUDA_ARCHITECTURES $ARCH)
set_target_properties(DCNv4_plugin_cu PROPERTIES CUDA_STANDARD 14)
set_target_properties(DCNv4_plugin_cu PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(DCNv4_plugin SHARED ${DCNV4_plugin_SRCS})
target_link_libraries(DCNv4_plugin DCNv4_plugin_cu nvinfer ${CUDA_LIBRARIES} ${DEPS})

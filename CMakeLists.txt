cmake_minimum_required(VERSION 3.16)
project(vllm_ascend_C)

# include(CheckCXXcompilerFlag)
# check_cxx_compiler_flag("-std=c++17", COMPILER_SUPPORTS_CXX17)
set(CMAKE_CXX_STANDARD 17)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# Suppress potential warnings about unused manually-specified variables
set(ignoreMe "${VLLM_PYTHON_PATH}")

# TODO: Add 3.12 back when torch-npu support 3.12
set(PYTHON_SUPPORTED_VERSIONS "3.9" "3.10" "3.11")

find_package(pybind11 REQUIRED)

append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")
set(VLLM_ASCEND_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}")

find_package(Torch REQUIRED)

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION ${SOC_VERSION})
message(STATUS "Detected SOC version: ${SOC_VERSION}")

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRINGS "Build type Release/Debug (default Release)" FORCE)
endif()

if (CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRINGS "path to install()")
endif()

set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_HOME_PATH})
if(EXISTS ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_HOME_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the cann package is installed.")
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

include_directories(
  ${pybind11_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_PATH}
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_NPU_PATH}/include
  ${ASCEND_HOME_PATH}/include
  ${ASCEND_HOME_PATH}/aarch64-linux/include/experiment/platform
  ${ASCEND_HOME_PATH}/x86_64-linux/include/experiment/platform
)

set(
  INCLUDES
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_NPU_INCLUDE_DIRS}
  ${ASCEND_HOME_PATH}/include
  ${ASCEND_HOME_PATH}/aarch64-linux/include/experiment/platform
)

file(GLOB KERNEL_FILES
${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/*.cpp)

ascendc_library(npu_kernels STATIC
    ${KERNEL_FILES}
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/bgmv_expand/op_kernel/bgmv_expand_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/bgmv_shrink/op_kernel/bgmv_shrink_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/get_masked_input_and_mask/op_kernel/get_masked_input_and_mask_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/mla_preprocess/op_kernel/mla_preprocess_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/rotary_embedding/op_kernel/rotary_embedding_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/sgmv_expand/op_kernel/sgmv_expand_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/sgmv_shrink/op_kernel/sgmv_shrink_kernel.cpp
)

add_library(vllm_ascend_kernels SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/utils/host_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/bgmv_expand/op_host/bgmv_expand.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/bgmv_shrink/op_host/bgmv_shrink.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/get_masked_input_and_mask/op_host/get_masked_input_and_mask.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/mla_preprocess/op_host/mla_preprocess.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/rotary_embedding/op_host/rotary_embedding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/sgmv_expand/op_host/sgmv_expand.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/sgmv_shrink/op_host/sgmv_shrink.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/swap_blocks/op_host/swap_blocks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/weak_ref_tensor/op_host/weak_ref_tensor.cpp
)

target_link_libraries(vllm_ascend_kernels PRIVATE
    npu_kernels
    ascendcl
    tiling_api
    register
    platform
    ascendalog
    dl
)

message("TORCH_NPU_PATH is ${TORCH_NPU_PATH}")

file(GLOB VLLM_ASCEND_SRC
${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp)

pybind11_add_module(vllm_ascend_C ${VLLM_ASCEND_SRC})

target_link_directories(vllm_ascend_C PRIVATE
  ${TORCH_NPU_PATH}/lib/
  ${ASCEND_HOME_PATH}/lib64
)

target_link_libraries(vllm_ascend_C PUBLIC
  ${TORCH_LIBRARIES}
  libtorch_npu.so
  vllm_ascend_kernels
)

target_link_options(vllm_ascend_C PRIVATE "-Wl,-rpath,$ORIGIN:$ORIGIN/lib")

install(TARGETS vllm_ascend_C vllm_ascend_kernels DESTINATION ${VLLM_ASCEND_INSTALL_PATH})

cmake_minimum_required(VERSION 3.20)

project(npu_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PYBIND11_FINDPYTHON ON)
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(SystemCLanguage CONFIG QUIET)

if(NOT SystemCLanguage_FOUND)
  find_package(SystemC CONFIG QUIET)
endif()

if(TARGET SystemC::systemc)
  set(NPU_SIM_SYSTEMC_TARGET SystemC::systemc)
elseif(TARGET systemc)
  set(NPU_SIM_SYSTEMC_TARGET systemc)
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(SYSTEMC REQUIRED IMPORTED_TARGET systemc)
  set(NPU_SIM_SYSTEMC_TARGET PkgConfig::SYSTEMC)
endif()

add_library(npu_sim_core
  cpp/src/backend/systemc/systemc_backend.cpp
  cpp/src/modules/output_pipeline.cpp
  cpp/src/modules/processing_element.cpp
  cpp/src/modules/sram_scratchpad.cpp
  cpp/src/modules/systolic_array.cpp
  cpp/src/simulator.cpp
)

target_include_directories(npu_sim_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

target_compile_features(npu_sim_core PUBLIC cxx_std_17)
target_link_libraries(npu_sim_core PUBLIC ${NPU_SIM_SYSTEMC_TARGET})

pybind11_add_module(_native
  cpp/bindings/python_bindings.cpp
  cpp/src/systemc_main.cpp
)
target_link_libraries(_native PRIVATE npu_sim_core)

install(TARGETS _native DESTINATION npu_sim)

enable_testing()

add_executable(npu_sim_core_tests
  cpp/src/systemc_main.cpp
  tests/cpp/test_simulator.cpp
)

target_link_libraries(npu_sim_core_tests PRIVATE npu_sim_core)
add_test(NAME npu_sim_core_tests COMMAND npu_sim_core_tests)

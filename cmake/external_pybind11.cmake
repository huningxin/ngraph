# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

#------------------------------------------------------------------------------
# Fetch and configure pybind11
#------------------------------------------------------------------------------

configure_file(${CMAKE_SOURCE_DIR}/cmake/pybind11_fetch.in.cmake ${CMAKE_CURRENT_BINARY_DIR}/pybind11/CMakeLists.txt)
message(STATUS "*********** ${CMAKE_CURRENT_BINARY_DIR}/pybind11")
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/pybind11")
message(STATUS "**")
execute_process(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/pybind11")

# set(TBB_ROOT ${CMAKE_CURRENT_BINARY_DIR}/tbb/tbb-src)
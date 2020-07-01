# ======================================================================== #
# Copyright 2018 Ingo Wald                                                 #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

set(CMAKE_MODULE_PATH
  "${PROJECT_SOURCE_DIR}/cmake"
#  "${CMAKE_CURRENT_SOURCE_DIR}/../cmake"
  ${CMAKE_MODULE_PATH}
  )

find_package(CUDA REQUIRED)
find_package(OptiX REQUIRED VERSION 7.1)

#include_directories(${CUDA_TOOLKIT_INCLUDE}, ${OptiX_INCLUDE})
if (CUDA_TOOLKIT_ROOT_DIR)
	include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
endif()
if (OptiX_INCLUDE)
  include_directories(${OptiX_INCLUDE})
endif()

if (WIN32)
  add_definitions(-DNOMINMAX)
endif()
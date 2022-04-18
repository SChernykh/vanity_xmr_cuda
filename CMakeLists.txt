# Support for this software's development was paid for by Fredrick R. Brennan's Modular Font Editor K Foundation, Inc.
# 
# Copyright (c) 2022 SChernykh <https://github.com/SChernykh>
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

cmake_minimum_required(VERSION 3.8)
project(vanity_xmr_cuda LANGUAGES C CXX CUDA)

include(FindCUDA)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES "50")

	if (CUDA_VERSION VERSION_LESS 9.0)
		list(APPEND CMAKE_CUDA_ARCHITECTURES "20;21")
	endif()

	if (CUDA_VERSION VERSION_LESS 11.0)
		list(APPEND CMAKE_CUDA_ARCHITECTURES "30")
	else()
		list(APPEND CMAKE_CUDA_ARCHITECTURES "35")
	endif()

	if (NOT CUDA_VERSION VERSION_LESS 8.0)
		list(APPEND CMAKE_CUDA_ARCHITECTURES "60")
	endif()

	if (NOT CUDA_VERSION VERSION_LESS 9.0)
		list(APPEND CMAKE_CUDA_ARCHITECTURES "70")
	endif()

	if (NOT CUDA_VERSION VERSION_LESS 10.0)
		list(APPEND CMAKE_CUDA_ARCHITECTURES "75")
	endif()

	if (NOT CUDA_VERSION VERSION_LESS 11.0)
		list(APPEND CMAKE_CUDA_ARCHITECTURES "80")
	endif()

	list(SORT CMAKE_CUDA_ARCHITECTURES ORDER DESCENDING)
endif()

if (${CMAKE_VERSION} VERSION_GREATER "3.5.2") 
	set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT vanity_xmr_cuda)
endif()

set(HEADERS
	src/crypto-ops.h
	src/gpu_crypto.h
	src/gpu_keccak.h
	src/gpu_scan.h
	src/keccak.h
)

set(SOURCES
	src/crypto-ops-data.c
	src/crypto-ops.c
	src/keccak.cpp
	src/kernel.cu
)

include_directories(src)
add_executable(${CMAKE_PROJECT_NAME} ${HEADERS} ${SOURCES})

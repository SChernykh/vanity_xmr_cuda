/*
 * Support for this software's development was paid for by Fredrick R. Brennan's Modular Font Editor K Foundation, Inc.
 *
 * Copyright (c) 2022 SChernykh <https://github.com/SChernykh>
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#pragma once

namespace gpu {

static __device__ __forceinline__ void sync()
{
#if (__CUDACC_VER_MAJOR__ >= 9)
	__syncwarp();
#else
	__syncthreads();
#endif
}

__constant__ static const int c[25][2] = {
	{ 1, 2}, { 2, 3}, { 3, 4}, { 4, 0}, { 0, 1},
	{ 6, 7}, { 7, 8}, { 8, 9}, { 9, 5}, { 5, 6},
	{11,12}, {12,13}, {13,14}, {14,10}, {10,11},
	{16,17}, {17,18}, {18,19}, {19,15}, {15,16},
	{21,22}, {22,23}, {23,24}, {24,20}, {20,21}
};

__constant__ static const int ppi[25][2] = {
	{0, 0},   {6, 44},  {12, 43}, {18, 21}, {24, 14}, {3, 28},  {9, 20}, {10, 3}, {16, 45},
	{22, 61}, {1, 1},   {7, 6},   {13, 25}, {19, 8},  {20, 18}, {4, 27}, {5, 36}, {11, 10},
	{17, 15}, {23, 56}, {2, 62},  {8, 55},  {14, 39}, {15, 41}, {21, 2}
};

__device__ __forceinline__ uint64_t R64(uint64_t a, int b, int c) { return (a << b) | (a >> c); }

#define ROUND(k) \
do { \
	C[t] = A[s] ^ A[s + 5] ^ A[s + 10] ^ A[s + 15] ^ A[s + 20]; \
	A[t] ^= C[s + 4] ^ R64(C[s + 1], 1, 63); \
	C[t] = R64(A[at], ro0, ro1); \
	A[t] = (C[t] ^ ((~C[c1]) & C[c2])) ^ (k1 & (k)); \
} while (0)

__global__ void __launch_bounds__(32) step1_keccak_12_rounds(const uint8_t* input_data, uint32_t input_size, uint64_t offset, uint64_t* data)
{
	const uint32_t t = threadIdx.x;
	const uint32_t g = blockIdx.x;

	if (t >= 25) {
		return;
	}

	const uint64_t* input = (const uint64_t*)(input_data);

	__shared__ uint64_t A[25];
	__shared__ uint64_t C[25];

	const uint32_t input_words = input_size / sizeof(uint64_t);
	A[t] = (t < input_words) ? input[t] : 0;

	sync();

	if (t == 0) {
		A[0] ^= offset + g;

		const uint32_t tail_size = input_size % sizeof(uint64_t);
		A[input_words] ^= 1ULL << (tail_size * 8);
		A[16] ^= 0x8000000000000000ULL;
	}

	sync();

	const uint32_t s = t % 5;
	const int at = ppi[t][0];
	const int ro0 = ppi[t][1];
	const int ro1 = 64 - ro0;
	const int c1 = c[t][0];
	const int c2 = c[t][1];
	const uint64_t k1 = (t == 0) ? (uint64_t)(-1) : 0;

	ROUND(0x0000000000000001ULL); ROUND(0x0000000000008082ULL); ROUND(0x800000000000808AULL);
	ROUND(0x8000000080008000ULL); ROUND(0x000000000000808BULL); ROUND(0x0000000080000001ULL);
	ROUND(0x8000000080008081ULL); ROUND(0x8000000000008009ULL); ROUND(0x000000000000008AULL);
	ROUND(0x0000000000000088ULL); ROUND(0x0000000080008009ULL); ROUND(0x000000008000000AULL);

	if (t < 4) {
		data[g * 4 + t] = A[t];
	}
}

#undef ROUND

} // gpu

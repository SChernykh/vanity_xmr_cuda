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

__constant__ const char alphabet[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
__constant__ const char alphabet_nocase[] = "123456789abcdefghjklmnpqrstuvwxyzabcdefghijkmnopqrstuvwxyz";

__global__ void step4_scan(const uint8_t* patterns, size_t pattern_count, bool case_sensitive, const void* data, uint32_t* results)
{
	const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	p += index * 32;

	uint64_t k = 0x12;
	for (int i = 0; i < 7; ++i) {
		k = (k << 8) | p[i];
	}

	const char* abc = case_sensitive ? alphabet : alphabet_nocase;

	uint8_t encoded_data[11];
	for (int i = 10; i >= 0; --i) {
		encoded_data[i] = abc[k % 58];
		k /= 58;
	}

	for (const uint8_t* p = patterns, *e = patterns + pattern_count * PATTERN_SIZE; p < e; p += PATTERN_SIZE) {
#define CHECK(index) { const uint8_t c = p[index]; if ((c != '?') && (c != encoded_data[index])) continue; }
		CHECK(0);
		CHECK(1);
		CHECK(2);
		CHECK(3);
		CHECK(4);
		CHECK(5);
		CHECK(6);
		CHECK(7);
		CHECK(8);
		CHECK(9);
		CHECK(10);
#undef CHECK

		const uint32_t k = atomicAdd(results, 1) + 1;
		if (k < 256) {
			results[k] = index;
		}
	}
}

} // gpu

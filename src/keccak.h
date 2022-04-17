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

#include <cstdint>

void keccak(const void* in, int inlen, uint8_t* md, int mdlen, int rounds);

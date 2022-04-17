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

#include <iostream>
#include <cstdint>
#include <random>
#include <chrono>
#include <thread>
#include <atomic>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "keccak.h"
#include "gpu_keccak.h"
#include "gpu_crypto.h"

constexpr size_t PATTERN_SIZE = 11;
#include "gpu_scan.h"

extern "C" {
#include "crypto-ops.h"
}

using namespace std::chrono;

constexpr char alphabet[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
constexpr char alphabet_nocase[] = "123456789abcdefghjklmnpqrstuvwxyzabcdefghijkmnopqrstuvwxyz";

constexpr size_t BATCH_SIZE = 1 << 20;

int main(int argc, char** argv)
{
    std::string patterns_str;
    bool case_sensitive = true;

    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];

        if (s == "-i") {
            case_sensitive = false;
            continue;
        }

        if (s.length() > PATTERN_SIZE) {
            s.resize(PATTERN_SIZE);
        }
        else {
            while (s.length() < PATTERN_SIZE) {
                s += '?';
            }
        }

        const char* abc = case_sensitive ? alphabet : alphabet_nocase;

        bool good = true;
        for (int j = 0; j < PATTERN_SIZE; ++j) {
            if (s[j] == '?') {
                continue;
            }
            bool found = false;
            for (int k = 0; k < 58; ++k) {
                if (s[j] == abc[k]) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                good = false;
                break;
            }
        }
        if (good) {
            patterns_str += s;
        }
        else {
            std::cout << "Invalid pattern \"" << argv[i] << "\"" << std::endl;
        }
    }

    if (patterns_str.empty()) {
        printf(
            "Usage:\n\n"
            "./vanity_xmr_cuda [-i] pattern1 [pattern_2] [pattern_3] ... [pattern_n]\n\n"
            "-i         case insensitive search (you can't use capital letters in patterns in this mode)\n\n"
            "Each pattern can have \"?\" symbols which match any character\n\n"
            "Example:\n\t./vanity_xmr_cuda -i 4?xxxxx 433333 455555 477777 499999\n\n"
            "If the vanity generator finds a match, it will print the spend secret key and the resulting Monero address.\n"
            "Copy the spend key and run \"./monero-wallet-cli --generate-from-spend-key WALLET_NAME\" to create this wallet.\n"
            "Paste the spend key when the wallet asks you for it.\n\n"
        );
        return 0;
    }

#define CHECKED_CALL(X) do { \
        const cudaError_t err = X; \
        if (err != cudaSuccess) { \
            std::cerr << #X " (line " << __LINE__ << ") failed, error " << err; \
            return __LINE__; \
        } \
    } while(0)

    int device_count;
    CHECKED_CALL(cudaGetDeviceCount(&device_count));

    // Get some entropy from the random device
    std::random_device::result_type rnd_buf[256];
    std::random_device rd;
    for (int i = 0; i < 256; ++i) {
        rnd_buf[i] = rd();
    }

    std::atomic<uint64_t> keys_checked;
    std::vector<std::thread> threads;

    for (int i = 0; i < device_count; ++i) {
        threads.emplace_back([i, case_sensitive, &rnd_buf, &patterns_str, &keys_checked]()
        {
            CHECKED_CALL(cudaSetDevice(i));

            cudaDeviceProp prop;
            CHECKED_CALL(cudaGetDeviceProperties(&prop, i));
            printf("Thread %d: running on %s\n", i, prop.name);

            CHECKED_CALL(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

            // Mix entropy into 32-byte secret spend key template
            uint8_t tmp_buf[sizeof(rnd_buf)];
            memcpy(tmp_buf, rnd_buf, sizeof(rnd_buf));

            // Mix in thread number
            tmp_buf[0] ^= i;

            // Mix all bits of the random buffer into the key template
            uint8_t key_template[32];
            keccak(tmp_buf, sizeof(tmp_buf), key_template, sizeof(key_template), 24);

            uint8_t* input_buf;
            CHECKED_CALL(cudaMalloc((void**)&input_buf, 32));
            CHECKED_CALL(cudaMemcpy(input_buf, key_template, sizeof(key_template), cudaMemcpyHostToDevice));

            uint64_t* data;
            CHECKED_CALL(cudaMalloc((void**)&data, BATCH_SIZE * 32));

            uint8_t* patterns;
            CHECKED_CALL(cudaMalloc((void**)&patterns, patterns_str.length()));
            CHECKED_CALL(cudaMemcpy(patterns, patterns_str.data(), patterns_str.length(), cudaMemcpyHostToDevice));

            uint32_t* results;
            CHECKED_CALL(cudaMalloc((void**)&results, 256 * sizeof(uint32_t)));

            for (uint64_t offset = 0;; offset += BATCH_SIZE) {
                CHECKED_CALL(cudaMemset(results, 0, sizeof(uint32_t)));

                gpu::step1_keccak_12_rounds<<<BATCH_SIZE, 32>>>(input_buf, 32, offset, data);
                CHECKED_CALL(cudaGetLastError());

                gpu::step2_reduce<<<BATCH_SIZE / 32, 32>>>(data);
                CHECKED_CALL(cudaGetLastError());

                gpu::step3_gen_public_key<<<BATCH_SIZE / 32, 32>>>(data);
                CHECKED_CALL(cudaGetLastError());

                gpu::step4_scan<<<BATCH_SIZE / 32, 32>>>(patterns, patterns_str.length() / PATTERN_SIZE, case_sensitive, data, results);
                CHECKED_CALL(cudaGetLastError());

                CHECKED_CALL(cudaDeviceSynchronize());

                uint32_t results_host[256];
                CHECKED_CALL(cudaMemcpy(results_host, results, sizeof(results_host), cudaMemcpyDeviceToHost));

                for (uint32_t i = 1, n = std::min(255u, results_host[0]); i <= n; ++i) {
                    uint8_t buf[32];

                    *((uint64_t*)key_template) ^= offset + results_host[i];
                    keccak(key_template, 32, buf, 32, 12);
                    *((uint64_t*)key_template) ^= offset + results_host[i];

                    sc_reduce32(buf);

                    uint8_t spend_secret_key[32];
                    memcpy(spend_secret_key, buf, 32);

                    uint8_t encoded_spend_secret_key[65];
                    for (int i = 0; i < 32; ++i) {
                        encoded_spend_secret_key[i * 2    ] = "0123456789abcdef"[spend_secret_key[i] >> 4];
                        encoded_spend_secret_key[i * 2 + 1] = "0123456789abcdef"[spend_secret_key[i] & 15];
                    }
                    encoded_spend_secret_key[64] = '\0';

                    uint8_t view_secret_key[32];
                    keccak(spend_secret_key, 32, view_secret_key, 32, 24);
                    sc_reduce32(view_secret_key);

                    uint8_t wallet_address[128];
                    wallet_address[0] = 0x12;

                    ge_p3 point;
                    ge_scalarmult_base(&point, spend_secret_key);
                    ge_p3_tobytes(wallet_address + 1, &point);

                    ge_scalarmult_base(&point, view_secret_key);
                    ge_p3_tobytes(wallet_address + 33, &point);

                    keccak(wallet_address, 65, buf, 32, 24);
                    memcpy(wallet_address + 65, buf, 4);

                    char encoded_wallet_address[128] = {};

                    constexpr char alphabet58[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

                    for (int i = 0; i < 64; i += 8) {
                        uint64_t k = 0;
                        for (int j = 0; j < 8; ++j) {
                            k = (k << 8) | wallet_address[i + j];
                        }

                        for (int j = 10; j >= 0; --j) {
                            encoded_wallet_address[j + (i / 8) * 11] = alphabet58[k % 58];
                            k /= 58;
                        }
                    }

                    uint64_t k = 0;
                    for (int j = 0; j < 5; ++j) {
                        k = (k << 8) | wallet_address[64 + j];
                    }

                    for (int j = 6; j >= 0; --j) {
                        encoded_wallet_address[j + 88] = alphabet58[k % 58];
                        k /= 58;
                    }

                    printf("%s %s\n", encoded_spend_secret_key, encoded_wallet_address);
                }

                keys_checked += BATCH_SIZE;
            }
        });
    }

    auto t1 = high_resolution_clock::now();

    uint64_t prev_keys_checked = 0;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(15));

        const uint64_t cur_keys_checked = keys_checked;
        const auto t2 = high_resolution_clock::now();

        const double dt = duration_cast<nanoseconds>(t2 - t1).count() * 1e-9;
        std::cout << (cur_keys_checked - prev_keys_checked) / dt * 1e-6 << " million keys/second" << std::endl;

        t1 = t2;
        prev_keys_checked = cur_keys_checked;
    }

    return 0;
}

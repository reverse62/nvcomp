/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#define CATCH_CONFIG_MAIN
// #define LOSSY_TEST_FLAG

#include "nvcomp.hpp"
#include "nvcomp/bitcomp.hpp"

#include "catch.hpp"

#include <assert.h>
#include <stdlib.h>
#include <vector>
#include <stdio.h>
#include <string>
#include <fstream>
#include <streambuf>
#include <iostream>
#include <dirent.h>
#include <thrust/extrema.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

// Test GPU decompression with Bitcomp compression API //

using namespace std;
using namespace nvcomp;

#define CUDA_CHECK(cond)                                                       \
  do {                                                                         \
    cudaError_t err = cond;                                                    \
    REQUIRE(err == cudaSuccess);                                               \
  } while (false)

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{

// template <typename T>
// std::vector<T> buildRuns(const size_t numRuns, const size_t runSize)
// {
//   std::vector<T> input;
//   for (size_t i = 0; i < numRuns; i++) {
//     for (size_t j = 0; j < runSize; j++) {
//       input.push_back(static_cast<T>(i));
//     }
//   }

//   return input;
// }

// template <typename T>
// void test_bitcomp(const std::vector<T>& input, nvcompType_t data_type, uint8_t is_lossy = 0)
// {
//   // create GPU only input buffer
//   T* d_in_data;
//   const size_t in_bytes = sizeof(T) * input.size();
//   CUDA_CHECK(cudaMalloc((void**)&d_in_data, in_bytes));
//   CUDA_CHECK(
//       cudaMemcpy(d_in_data, input.data(), in_bytes, cudaMemcpyHostToDevice));

//   cudaStream_t stream;
//   cudaStreamCreate(&stream);

//   BitcompManager manager{data_type, 0, stream};
//   // manager.is_lossy = is_lossy;
//   auto comp_config = manager.configure_compression(in_bytes);

//   // Allocate output buffer
//   uint8_t* d_comp_out;
//   CUDA_CHECK(cudaMalloc(&d_comp_out, comp_config.max_compressed_buffer_size));

//   manager.compress(
//       reinterpret_cast<const uint8_t*>(d_in_data),
//       d_comp_out,
//       comp_config);

//   CUDA_CHECK(cudaStreamSynchronize(stream));

//   size_t comp_out_bytes = manager.get_compressed_output_size(d_comp_out);

//   cudaFree(d_in_data);

//   // Test to make sure copying the compressed file is ok
//   uint8_t* copied = 0;
//   CUDA_CHECK(cudaMalloc(&copied, comp_out_bytes));
//   CUDA_CHECK(
//       cudaMemcpy(copied, d_comp_out, comp_out_bytes, cudaMemcpyDeviceToDevice));
//   cudaFree(d_comp_out);
//   d_comp_out = copied;

//   auto decomp_config = manager.configure_decompression(d_comp_out);

//   T* out_ptr;
//   cudaMalloc(&out_ptr, decomp_config.decomp_data_size);

//   // make sure the data won't match input if not written to, so we can verify
//   // correctness
//   cudaMemset(out_ptr, 0, decomp_config.decomp_data_size);

//   manager.decompress(
//       reinterpret_cast<uint8_t*>(out_ptr),
//       d_comp_out,
//       decomp_config);
//   CUDA_CHECK(cudaStreamSynchronize(stream));

//   // Copy result back to host
//   std::vector<T> res(input.size());
//   cudaMemcpy(
//       &res[0], out_ptr, input.size() * sizeof(T), cudaMemcpyDeviceToHost);

//   if(is_lossy == 1){
//     for(int ind = 0; ind < res.size(); ind ++){
//       printf("the value of index %d is %f\n", ind, res[ind]);
//     }
//   }

//   // Verify correctness
//   REQUIRE(res == input);
  
//   cudaFree(d_comp_out);
//   cudaFree(out_ptr);
// }

template <typename T>
void test_lossy_bitcomp(const T* input, size_t dtype_len)
{
  // find the range of data
  float range = *max_element(input, input + dtype_len) - *min_element(input, input + dtype_len);

  // compute the delta based on range and write to configuration file
  // ofstream MyFile("/home/boyuan.zhang1/bitcomp_lossy_config.txt");
  // float delta = range*1e-2;
  // MyFile << "1 1 1 "+std::to_string(delta);
  // MyFile.close();

  // create GPU only input buffer
  T* d_in_data;
  
  const size_t in_bytes = sizeof(T) * dtype_len;
  CUDA_CHECK(cudaMalloc((void**)&d_in_data, in_bytes));
  CUDA_CHECK(
      cudaMemcpy(d_in_data, input, in_bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  BitcompManager manager{NVCOMP_TYPE_INT, 0, stream};
  auto comp_config = manager.configure_compression(in_bytes);

  // Allocate output buffer
  uint8_t* d_comp_out;
  CUDA_CHECK(cudaMalloc(&d_comp_out, comp_config.max_compressed_buffer_size));

  manager.compress(
      reinterpret_cast<const uint8_t*>(d_in_data),
      d_comp_out,
      comp_config);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  size_t comp_out_bytes = manager.get_compressed_output_size(d_comp_out);

  cudaFree(d_in_data);

  // Test to make sure copying the compressed file is ok
  uint8_t* copied = 0;
  CUDA_CHECK(cudaMalloc(&copied, comp_out_bytes));
  CUDA_CHECK(
      cudaMemcpy(copied, d_comp_out, comp_out_bytes, cudaMemcpyDeviceToDevice));
  cudaFree(d_comp_out);
  d_comp_out = copied;

  auto decomp_config = manager.configure_decompression(d_comp_out);

  T* out_ptr;
  cudaMalloc(&out_ptr, decomp_config.decomp_data_size);

  // make sure the data won't match input if not written to, so we can verify
  // correctness
  cudaMemset(out_ptr, 0, decomp_config.decomp_data_size);

  manager.decompress(
      reinterpret_cast<uint8_t*>(out_ptr),
      d_comp_out,
      decomp_config);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Copy result back to host
  std::vector<T> res(dtype_len);
  cudaMemcpy(
      &res[0], out_ptr, dtype_len * sizeof(T), cudaMemcpyDeviceToHost);

  uint8_t flg = 1;
  for (auto i = 0; i < dtype_len; i++){
    if (input[i] != res[i]) {
      printf("the value of index %d is different, input is: %e, output is: %e\n", i, input[i], res[i]);
      flg = 0;
    }
  }
  //Verify correctness
  REQUIRE(flg == 1);
  
  cudaFree(d_comp_out);
  cudaFree(out_ptr);
}

template <typename T>
T* read_binary_to_new_array(const std::string& fname, size_t dtype_len)
{
    std::ifstream ifs(fname.c_str(), std::ios::binary | std::ios::in);
    if (not ifs.is_open()) {
        std::cerr << "fail to open " << fname << std::endl;
        exit(1);
    }
    auto _a = new T[dtype_len]();
    ifs.read(reinterpret_cast<char*>(_a), std::streamsize(dtype_len * sizeof(T)));
    ifs.close();
    return _a;
}

} // namespace

/******************************************************************************
 * UNIT TESTS *****************************************************************
 *****************************************************************************/

// TEST_CASE("comp/decomp Bitcomp-small", "[nvcomp]")
// {
//   using T = int;

//   std::vector<T> input = {0, 2, 2, 3, 0, 0, 0, 0, 0, 3, 1, 1, 1, 1, 1, 2, 3, 3};

//   test_bitcomp(input, NVCOMP_TYPE_INT);
// }

// TEST_CASE("comp/decomp Bitcomp-1", "[nvcomp]")
// {
//   using T = int;

//   const int num_elems = 500;
//   std::vector<T> input;
//   for (int i = 0; i < num_elems; ++i) {
//     input.push_back(i >> 2);
//   }

//   test_bitcomp(input, NVCOMP_TYPE_INT);
// }

// TEST_CASE("comp/decomp Bitcomp-all-small-sizes", "[nvcomp][small]")
// {
//   using T = uint8_t;

//   for (int total = 1; total < 4096; ++total) {
//     std::vector<T> input = buildRuns<T>(total, 1);
//     test_bitcomp(input, NVCOMP_TYPE_UCHAR);
//   }
// }

// TEST_CASE("comp/decomp Bitcomp-multichunk", "[nvcomp][large]")
// {
//   using T = int;

//   for (int total = 10; total < (1 << 24); total = total * 2 + 7) {
//     std::vector<T> input = buildRuns<T>(total, 10);
//     test_bitcomp(input, NVCOMP_TYPE_INT);
//   }
// }

// TEST_CASE("comp/decomp Bitcomp-small-uint8", "[nvcomp][small]")
// {
//   using T = uint8_t;

//   for (size_t num = 1; num < 1 << 18; num = num * 2 + 1) {
//     std::vector<T> input = buildRuns<T>(num, 3);
//     test_bitcomp(input, NVCOMP_TYPE_UCHAR);
//   }
// }

// TEST_CASE("comp/decomp Bitcomp-small-uint16", "[nvcomp][small]")
// {
//   using T = uint16_t;

//   for (size_t num = 1; num < 1 << 18; num = num * 2 + 1) {
//     std::vector<T> input = buildRuns<T>(num, 3);
//     test_bitcomp(input, NVCOMP_TYPE_USHORT);
//   }
// }

// TEST_CASE("comp/decomp Bitcomp-small-uint32", "[nvcomp][small]")
// {
//   using T = uint32_t;

//   for (size_t num = 1; num < 1 << 18; num = num * 2 + 1) {
//     std::vector<T> input = buildRuns<T>(num, 3);
//     test_bitcomp(input, NVCOMP_TYPE_UINT);
//   }
// }

// TEST_CASE("comp/decomp Bitcomp-small-uint64", "[nvcomp][small]")
// {
//   using T = float;

//   for (size_t num = 1; num < 1 << 18; num = num * 2 + 1) {
//     std::vector<T> input = buildRuns<T>(num, 3);
//     // NVCOMP_TYPE_ULONGLONG currently unsupported
//     test_bitcomp(input, NVCOMP_TYPE_UINT);
//   }
// }

// TEST_CASE("comp/lossy decomp Bitcomp-small", "[nvcomp]")
// {
//   using T = float;

//   std::vector<T> input = {0, 2.0, 2.0, 3.0, 0, 0, 0, 0, 0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0};

//   test_bitcomp(input, NVCOMP_TYPE_INT, 1);
// }

// TEST_CASE("comp/lossy decomp Bitcomp-small with decimal num", "[nvcomp]")
// {
//   using T = float;

//   std::vector<T> input = {0, 2.1, 2.2, 3.3, 0, 0, 0, 0, 0, 3.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.1, 3.2, 3.3};

//   test_bitcomp(input, NVCOMP_TYPE_INT, 1);
// }

TEST_CASE("comp/lossy decomp Bitcomp with our dataset", "[nvcomp]")
// int main(int argc, char *argv[])
{
  using T = float;

  // read bitcomp_lossy_file.txt to get the directory of files and size
  std::ifstream t("/home/boyuan.zhang1/bitcomp_lossy_file.txt");
  t.seekg(0, std::ios::end);
  size_t size = t.tellg();
  std::string s(size, ' ');
  t.seekg(0);
  t.read(&s[0], size); 

  // parse the string to get dir path and size 
  std::string delimiter = " ";
  size_t pos = 0;
  std::string dir_name;
  size_t dtype_len;
  pos = s.find(delimiter);
  dir_name = s.substr(0, pos);
  s.erase(0, pos + delimiter.length());
  dtype_len = stoi(s);
  
  // iterate the dir to get file with .dat and .f32 to compress
  struct dirent *entry = nullptr;
  DIR *dp = nullptr;
  std::string extension1 = ".dat";
  std::string extension2 = ".f32";
  std::string fname;
  T* arr;

  dp = opendir(dir_name.c_str());
  if (dp != nullptr) {
    while ((entry = readdir(dp))){
      fname = entry->d_name;
      if(fname.find(extension1, (fname.length() - extension1.length())) != std::string::npos || 
          fname.find(extension2, (fname.length() - extension2.length())) != std::string::npos){
        printf ("%s\n", entry->d_name);
        arr = read_binary_to_new_array<T>(dir_name + "/" + fname, dtype_len);
        test_lossy_bitcomp(arr, dtype_len);
        delete arr;
      }
    }
  }
  closedir(dp);
  // return 0;
}


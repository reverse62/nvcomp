/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include <assert.h>
#include <stdlib.h>
#include <vector>
#include <stdio.h>
#include <string>
#include <fstream>
#include <sstream>
#include <streambuf>
#include <iostream>
#include <dirent.h>
#include <algorithm>
#include <cmath>
#include <sys/stat.h>

long GetFileSize(std::string filename)
{
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

#include "nvcomp.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"
#include "nvcomp/cascaded.hpp"

using namespace std;
using namespace nvcomp;

#define REQUIRE(a)                                                             \
  do {                                                                         \
    if (!(a)) {                                                                \
      std::cerr << "Failure" << std::endl;                                \
      exit(1);                                                        \
    }                                                                          \
  } while (0)

#define CUDA_CHECK(cond)                                                       \
  do {                                                                         \
    cudaError_t err = cond;                                                    \
    if (err != cudaSuccess) {                                               \
      std::cerr << "Failure" << std::endl;                                \
      return;                                                              \
    }                                                                         \
  } while (false)

namespace
{

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

template <typename T>
void test_lz4(const T* input, size_t dtype_len, nvcompType_t data_type, const size_t chunk_size = 1 << 16)
{
  // create GPU only input buffer
  T* d_in_data;
  const size_t in_bytes = sizeof(T) * dtype_len;
  CUDA_CHECK(cudaMalloc((void**)&d_in_data, in_bytes));
  CUDA_CHECK(
      cudaMemcpy(d_in_data, input, in_bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  LZ4Manager manager{chunk_size, data_type, stream};
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
  printf("the original data size is: %d\nthe compressed data size is: %d\nthe compress ratio is :%f\n", dtype_len, comp_out_bytes, float(dtype_len) / float(comp_out_bytes));


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

  cudaFree(d_comp_out);
  cudaFree(out_ptr);
}


} // namespace

int main(int argc, char *argv[])
{
  using T = unsigned char;

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
  std::string extension1 = argv[1];
  // std::string extension2 = ".f32";
  std::string fname;
  T* arr;
  size_t file_size;

  dp = opendir(dir_name.c_str());
  if (dp != nullptr) {
    while ((entry = readdir(dp))){
      fname = entry->d_name;
      if(fname.find(extension1, (fname.length() - extension1.length())) != std::string::npos){
        printf ("%s\n", entry->d_name);
        file_size = GetFileSize(dir_name + "/" + fname);
        arr = read_binary_to_new_array<T>(dir_name + "/" + fname, file_size);
        test_lz4<T>(arr, file_size, NVCOMP_TYPE_UCHAR);
        delete arr;
      }
    }
  }
  closedir(dp);
  return 0;
}
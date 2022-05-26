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
#include "nvcomp/snappy.h"

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
      exit(1);                                                              \
    }                                                                         \
  } while (false)

namespace
{
// typedef struct Stat {
//   double min_odata{}, max_odata{}, rng_odata{}, std_odata{};
//   double min_xdata{}, max_xdata{}, rng_xdata{}, std_xdata{};
//   double PSNR{}, MSE{}, NRMSE{};
//   double coeff{};
//   double user_set_eb{}, max_abserr_vs_rng{}, max_pwrrel_abserr{};

//   size_t len{}, max_abserr_index{};
//   double max_abserr{};

// } stat_t;

// template <typename T>
// void verify_data(stat_t* stat, const T* xdata, const T* odata, size_t len)
// {
//     double max_odata = odata[0], min_odata = odata[0];
//     double max_xdata = xdata[0], min_xdata = xdata[0];
//     double max_abserr = max_abserr = fabs(xdata[0] - odata[0]);

//     double sum_0 = 0, sum_x = 0;
//     for (size_t i = 0; i < len; i++) sum_0 += odata[i], sum_x += xdata[i];

//     double mean_odata = sum_0 / len, mean_xdata = sum_x / len;
//     double sum_var_odata = 0, sum_var_xdata = 0, sum_err2 = 0, sum_corr = 0, rel_abserr = 0;

//     double max_pwrrel_abserr = 0;
//     size_t max_abserr_index  = 0;
//     for (size_t i = 0; i < len; i++) {
//         max_odata = max_odata < odata[i] ? odata[i] : max_odata;
//         min_odata = min_odata > odata[i] ? odata[i] : min_odata;

//         max_xdata = max_xdata < odata[i] ? odata[i] : max_xdata;
//         min_xdata = min_xdata > xdata[i] ? xdata[i] : min_xdata;

//         float abserr = fabs(xdata[i] - odata[i]);
//         if (odata[i] != 0) {
//             rel_abserr        = abserr / fabs(odata[i]);
//             max_pwrrel_abserr = max_pwrrel_abserr < rel_abserr ? rel_abserr : max_pwrrel_abserr;
//         }
//         max_abserr_index = max_abserr < abserr ? i : max_abserr_index;
//         max_abserr       = max_abserr < abserr ? abserr : max_abserr;
//         sum_corr += (odata[i] - mean_odata) * (xdata[i] - mean_xdata);
//         sum_var_odata += (odata[i] - mean_odata) * (odata[i] - mean_odata);
//         sum_var_xdata += (xdata[i] - mean_xdata) * (xdata[i] - mean_xdata);
//         sum_err2 += abserr * abserr;
//     }
//     double std_odata = sqrt(sum_var_odata / len);
//     double std_xdata = sqrt(sum_var_xdata / len);
//     double ee        = sum_corr / len;

//     stat->len               = len;
//     stat->max_odata         = max_odata;
//     stat->min_odata         = min_odata;
//     stat->rng_odata         = max_odata - min_odata;
//     stat->std_odata         = std_odata;
//     stat->max_xdata         = max_xdata;
//     stat->min_xdata         = min_xdata;
//     stat->rng_xdata         = max_xdata - min_xdata;
//     stat->std_xdata         = std_xdata;
//     stat->coeff             = ee / std_odata / std_xdata;
//     stat->max_abserr_index  = max_abserr_index;
//     stat->max_abserr        = max_abserr;
//     stat->max_abserr_vs_rng = max_abserr / stat->rng_odata;
//     stat->max_pwrrel_abserr = max_pwrrel_abserr;
//     stat->MSE               = sum_err2 / len;
//     stat->NRMSE             = sqrt(stat->MSE) / stat->rng_odata;
//     stat->PSNR              = 20 * log10(stat->rng_odata) - 10 * log10(stat->MSE);
// }

// template <typename Data>
// void print_data_quality_metrics(
//     stat_t* stat,
//     size_t  archive_nbyte = 0,
//     bool    gpu_checker   = false
//     // size_t  bin_scale    = 1 // TODO
// )
// {
//     auto checker = (not gpu_checker) ? string("(using CPU checker)") : string("(using GPU checker)");
//     auto nbyte   = (stat->len * sizeof(Data) * 1.0);

//     auto print_ln = [](const char* s, double n1, double n2, double n3, double n4) {
//         printf("  %-12s\t%15.8g\t%15.8g\t%15.8g\t%15.8g\n", s, n1, n2, n3, n4);
//     };
//     auto print_head = [](const char* s1, const char* s2, const char* s3, const char* s4, const char* s5) {
//         printf("  \e[1m\e[31m%-12s\t%15s\t%15s\t%15s\t%15s\e[0m\n", s1, s2, s3, s4, s5);
//     };

//     printf("\nquality metrics %s:\n", checker.c_str());
//     printf(
//         "  %-12s\t%15lu\t%15s\t%15lu\n",  //
//         const_cast<char*>("data-len"), stat->len, const_cast<char*>("data-byte"), sizeof(Data));

//     print_head("", "min", "max", "rng", "std");
//     print_ln("origin", stat->min_odata, stat->max_odata, stat->rng_odata, stat->std_odata);
//     print_ln("eb-lossy", stat->min_xdata, stat->max_xdata, stat->rng_xdata, stat->std_xdata);

//     print_head("", "abs-val", "abs-idx", "pw-rel", "VS-RNG");
//     print_ln("max-error", stat->max_abserr, stat->max_abserr_index, stat->max_pwrrel_abserr, stat->max_abserr_vs_rng);

//     print_head("", "CR", "NRMSE", "corr-coeff", "PSNR");
//     print_ln("metrics", nbyte / archive_nbyte, stat->NRMSE, stat->coeff, stat->PSNR);

//     printf("\n");
// };

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

// template <typename T>
// void test_bitcomp(const T* input, size_t dtype_len, nvcompType_t data_type)
// {
//   // create GPU only input buffer
//   T* d_in_data;
//   const size_t in_bytes = sizeof(T) * dtype_len;
//   CUDA_CHECK(cudaMalloc((void**)&d_in_data, in_bytes));
//   CUDA_CHECK(
//       cudaMemcpy(d_in_data, input, in_bytes, cudaMemcpyHostToDevice));

//   cudaStream_t stream;
//   cudaStreamCreate(&stream);

//   BitcompManager manager{data_type, 0, stream};
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
//   printf("the original data size is: %d\nthe compressed data size is: %d\nthe compress ratio is :%f\n", dtype_len * 4, comp_out_bytes, float(dtype_len) * 4 / float(comp_out_bytes));

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
//   std::vector<T> res(dtype_len);
//   cudaMemcpy(
//       &res[0], out_ptr, dtype_len * sizeof(T), cudaMemcpyDeviceToHost);

//   auto stat = new stat_t;
//   verify_data<unsigned short int>(stat, res.data(), input, dtype_len);
//   print_data_quality_metrics<unsigned short int>(stat);

//   cudaFree(d_comp_out);
//   cudaFree(out_ptr);
// }

template <typename T>
void test_snappy(const T* input, size_t dtype_len)
{
  const size_t batch_size = 1;

  // prepare input and output on host
  size_t* host_batch_sizes = (size_t*)malloc(batch_size * sizeof(size_t));
  for (size_t i = 0; i < batch_size; ++i) {
    host_batch_sizes[i] = dtype_len;
  }

  size_t* host_batch_bytes = (size_t*)malloc(batch_size * sizeof(size_t));
  size_t max_chunk_size = 0;
  for (size_t i = 0; i < batch_size; ++i) {
    host_batch_bytes[i] = sizeof(T) * host_batch_sizes[i];
    if (host_batch_bytes[i] > max_chunk_size) {
      max_chunk_size = host_batch_bytes[i];
    }
  }

  T** host_input = (T**)malloc(sizeof(T*) * batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    host_input[i] = (T*)malloc(sizeof(T) * host_batch_sizes[i]);
    for (size_t j = 0; j < host_batch_sizes[i]; ++j) {
      // make sure there should be some repeats to compress
      host_input[i][j] = input[j];
    }
  }
  free(host_batch_sizes);

  T** host_output = (T**)malloc(sizeof(T*) * batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    host_output[i] = (T*)malloc(host_batch_bytes[i]);
  }

  // prepare gpu buffers
  void** host_in_ptrs = (void**)malloc(sizeof(void*) * batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    CUDA_CHECK(cudaMalloc(&host_in_ptrs[i], host_batch_bytes[i]));
    CUDA_CHECK(cudaMemcpy(
        host_in_ptrs[i],
        host_input[i],
        host_batch_bytes[i],
        cudaMemcpyHostToDevice));
  }
  void** device_in_pointers;
  CUDA_CHECK(cudaMalloc(
      (void**)&device_in_pointers, sizeof(*device_in_pointers) * batch_size));
  CUDA_CHECK(cudaMemcpy(
      device_in_pointers,
      host_in_ptrs,
      sizeof(*device_in_pointers) * batch_size,
      cudaMemcpyHostToDevice));

  size_t* device_batch_bytes;
  CUDA_CHECK(cudaMalloc(
      (void**)&device_batch_bytes, sizeof(*device_batch_bytes) * batch_size));
  CUDA_CHECK(cudaMemcpy(
      device_batch_bytes,
      host_batch_bytes,
      sizeof(*device_batch_bytes) * batch_size,
      cudaMemcpyHostToDevice));

  nvcompStatus_t status;

  // Compress on the GPU using batched API
  size_t comp_temp_bytes;
  status = nvcompBatchedSnappyCompressGetTempSize(batch_size, max_chunk_size, nvcompBatchedSnappyDefaultOpts, &comp_temp_bytes);
  
  // if (max_chunk_size > 1<<16) printf("max_chunk_size = %zu\n", max_chunk_size);
  REQUIRE(status == nvcompSuccess);

  void* d_comp_temp;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

  size_t max_comp_out_bytes;
  status = nvcompBatchedSnappyCompressGetMaxOutputChunkSize(max_chunk_size, nvcompBatchedSnappyDefaultOpts, &max_comp_out_bytes);
  REQUIRE(status == nvcompSuccess);

  void** host_comp_out = (void**)malloc(sizeof(void*) * batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    CUDA_CHECK(cudaMalloc(&host_comp_out[i], max_comp_out_bytes));
  }
  void** device_comp_out;
  CUDA_CHECK(cudaMalloc(
      (void**)&device_comp_out, sizeof(*device_comp_out) * batch_size));
  CUDA_CHECK(cudaMemcpy(
      device_comp_out,
      host_comp_out,
      sizeof(*device_comp_out) * batch_size,
      cudaMemcpyHostToDevice));

  size_t* device_comp_out_bytes;
  CUDA_CHECK(cudaMalloc(
      (void**)&device_comp_out_bytes,
      sizeof(*device_comp_out_bytes) * batch_size));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  status = nvcompBatchedSnappyCompressAsync(
      (const void* const*)device_in_pointers,
      device_batch_bytes,
      max_chunk_size,
      batch_size,
      d_comp_temp,
      comp_temp_bytes,
      device_comp_out,
      device_comp_out_bytes,
      nvcompBatchedSnappyDefaultOpts, 
      stream);
  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  size_t* tmp_size;
  tmp_size = (size_t*)malloc(sizeof(*device_comp_out_bytes) * batch_size);
  cudaMemcpy(
      tmp_size,
      device_comp_out_bytes,
      sizeof(*device_comp_out_bytes) * batch_size,
      cudaMemcpyDeviceToHost);
  std::cout<< "original data size: " << dtype_len << endl;
  std::cout<< "compressed data size: " << tmp_size[0] << endl;
  std::cout<< "compression ratio:  " << float(dtype_len) / float(tmp_size[0]) << endl;

  CUDA_CHECK(cudaFree(d_comp_temp));
  for (size_t i = 0; i < batch_size; ++i) {
    CUDA_CHECK(cudaFree(host_in_ptrs[i]));
  }
  cudaFree(device_in_pointers);
  free(host_in_ptrs);

  size_t temp_bytes;
  status = nvcompBatchedSnappyDecompressGetTempSize(batch_size, max_chunk_size, &temp_bytes);

  void* device_temp_ptr;
  CUDA_CHECK(cudaMalloc(&device_temp_ptr, temp_bytes));

  size_t* device_decomp_out_bytes;
  CUDA_CHECK(cudaMalloc(
      (void**)&device_decomp_out_bytes,
      sizeof(*device_decomp_out_bytes) * batch_size));

  status = nvcompBatchedSnappyGetDecompressSizeAsync(
      (const void* const*)device_comp_out,
      device_comp_out_bytes,
      device_decomp_out_bytes,
      batch_size,
      stream);
  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(stream));
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
        test_snappy<T>(arr, file_size);
        delete arr;
      }
    }
  }
  closedir(dp);
  return 0;
}
//===- resnet18_multibatch_test.cc ---------------------------------------===//
//
// Copyright (C) 2019-2021 Alibaba Group Holding Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

// clang-format off
// Testing CXX Code Gen using ODLA on NPU(HgEngine)
// RUN: %halo_compiler -target cxx -batch-size=16 %src_dir/tests/parser/onnx/resnet18_v1_hgai.onnx -o %t_nc.cc  -disable-broadcasting -input-shape=data:16x3x224x224
// RUN: %cxx -O0 -g -I%odla_path/include -c -o %t_nc.o %t_nc.cc
// RUN: %cxx -O0 -g -DBATCH=16 -DTIMING_TEST -DITER_TEST=1 %include %s %t_nc.o %t_nc.bin -I%odla_path/include -I%dnnl_path/include -L%hgai_path/lib -L%dnnl_path/lib %odla_link -lodla_hgai -lhg_profiler -lhgrt -lalinpu -Wl,-rpath=%dnnl_path/lib:%hgai_path/lib -o %t_nc.exe
// RUN: %t_nc.exe | FileCheck %s

// Testing CXX Code Gen using ODLA on NPU(HgEngine), (Enable Auto-fuse)
// RUN: %halo_compiler --tvm-fuse-on --cutlass-fuse-on -target cxx -batch-size=16 %src_dir/tests/parser/onnx/resnet18_v1_hgai.onnx -o %t_nc.cc  -disable-broadcasting -input-shape=data:16x3x224x224
// RUN: %cxx -O0 -g -I%odla_path/include -c -o %t_nc.o %t_nc.cc
// RUN: %cxx -O0 -g -DBATCH=16 -DTIMING_TEST -DITER_TEST=1 %include %s %t_nc.o %t_nc.bin -I%odla_path/include -I%dnnl_path/include -L%hgai_path/lib -L%dnnl_path/lib %odla_link -lodla_hgai -lhg_profiler -lhgrt -lalinpu -Wl,-rpath=%dnnl_path/lib:%hgai_path/lib -o %t_nc.exe
// RUN: %t_nc.exe | FileCheck %s
// clang-format on
// clang-format on

// CHECK: Result verified

#define TEST_SET 1
#include <vector>

#include "resnet_data.h"
#include "test_util.h"

extern "C" {
#ifdef USE_INFERENCE_FUNC_SIG
void model_run(int num_inputs, const void* inputs[], int num_outputs,
               void* outputs[]);
#else
void resnet18_v1_hgai(const float* in, float* out);
#endif
}

#ifndef COMPARE_ERROR
#define COMPARE_ERROR 1e-8
#endif

#ifndef BATCH
#define BATCH 1
#endif

int main(int argc, char** argv) {
  float out[1000 * BATCH];

  size_t input_size = sizeof(test_input) / sizeof(test_input[0]);
  size_t output_size = sizeof(test_output_ref) / sizeof(out[0]);
  std::vector<float> batch_input;
  std::vector<float> batch_output_ref;
  batch_input.reserve(input_size * BATCH);
  batch_output_ref.reserve(output_size * BATCH);

  for (int i = 0; i < BATCH; ++i) {
    batch_input.insert(batch_input.end(), test_input, test_input + input_size);
    batch_output_ref.insert(batch_output_ref.end(), test_output_ref,
                            test_output_ref + output_size);
  }

  resnet18_v1_hgai(batch_input.data(), out);
  if (Verify(out, batch_output_ref.data(), batch_output_ref.size(),
             COMPARE_ERROR)) {
    std::cout << "Batch:" << BATCH << "\n";
    std::cout << "Result verified\n";
#ifdef TIMING_TEST
    for (int i = 0; i < 200; ++i) {
      auto begin_time = Now();
      resnet18_v1_hgai(batch_input.data(), out);
      auto end_time = Now();
      std::cout << "Elapse time: " << GetDuration(begin_time, end_time)
                << " seconds\n";
    }
#endif
    return 0;
  }

  std::cout << " Failed\n";
  return 1;
}

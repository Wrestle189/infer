#include <ODLA/odla.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

#include "../include/test_util.h"
#include "data/resnet18_data.h"

#ifndef COMPARE_ERROR
#define COMPARE_ERROR 2e-1
#endif

static bool error_flag = false;

int main(int argc, char **argv) {

  int batch = argc < 2 ? 1 : atoi(argv[1]);
  int total_iters = argc < 3 ? 1 : atoi(argv[2]);
  std::vector<float> out(1000);
  if (batch > 1) {
    auto resize = [batch](auto& data) {
      auto n = data.size();
      data.resize(n * batch);
      for (int i = 1; i < batch; ++i) {
        std::copy(data.begin(), data.begin() + n, data.begin() + i * n);
      }
    };
    resize(test_input);
    resize(test_output_gt);
    resize(out);
  }

  // int input_size = sizeof(test_input) / sizeof(test_input[0]); // 3*224*224
  // int output_size = sizeof(test_output_gt) / sizeof(test_output_gt[0]); // 1000
  // std::vector<float> batch_input;
  // batch_input.reserve(batch * input_size);

  // for (int i = 0; i < batch; i++) {
  //   batch_input.insert(batch_input.end(), test_input, test_input + input_size);
  // }
  // std::vector<float> output(batch * 1000);

  odla_vendor vendor = nullptr;
  odla_device_name device_name = ODLA_DEVICE_NVIDIA_TENSORRT;
  int device_idx = 0;
  odla_device device = nullptr;
  ODLA_RET_CHECK(odla_AllocateDevice(vendor, device_name, device_idx, &device),1);

  std::string trt_engine_file = xstr(ENGINE);
  odla_resource_location loc;
  loc.location_type = ODLA_LOCATION_PATH;
  loc.location = trt_engine_file.c_str();

  odla_executable exec = nullptr;

  ODLA_RET_CHECK(odla_LoadExecutable(loc, device, &exec),1);

  std::array<odla_value, 1> input_vals;
  for (int i = 0; i < input_vals.size(); ++i) {
    ODLA_RET_CHECK(odla_GetArgFromExecutableByIdx(exec, i, &input_vals[i]),1);
  }
  std::array<odla_value, 1> output_vals;
  for (int i = 0; i < output_vals.size(); ++i) {
    ODLA_RET_CHECK(odla_GetOutputFromExecutableByIdx(exec, i, &output_vals[i]),1);
  }

  odla_context ctx = nullptr;
  ODLA_RET_CHECK(odla_CreateContext(&ctx),1);
  ODLA_RET_CHECK(
      odla_SetContextItem(ctx, ODLA_RUN_BATCH_SIZE, (odla_item_value)&batch),1);

  std::array<void*, 1> input_data{test_input.data()};
  std::array<void*, 1> output_data{out.data()};
  auto ts_begin = Now();
  bool verified = true;

  for (int i = 0; i < total_iters; ++i) {
    for (int idx = 0; idx < input_data.size(); ++idx) {
      ODLA_RET_CHECK(
          odla_BindToArgument(input_vals[idx], input_data[idx], ctx),1);
    }
    ODLA_RET_CHECK(odla_LaunchExecutable(exec, ctx),1);
    for (int idx = 0; idx < output_data.size(); ++idx) {
      ODLA_RET_CHECK(
          odla_GetValueData(output_vals[idx], output_data[idx], ctx),1);
    }

    if (!(Verify(out.data(), test_output_gt.data(), test_output_gt.size(), COMPARE_ERROR))) {
      std::cout << "Incorrect results\n";
      ODLA_RET_CHECK(odla_DestroyContext(ctx),1);
      ODLA_RET_CHECK(odla_DestroyExecutable(exec),1);
      ODLA_RET_CHECK(odla_DestroyDevice(device),1);
      return 1;
    }
  }

  auto dur = GetDuration(ts_begin, Now());
  int64_t queries = total_iters * batch;
  std::cout << "Queries:" << queries << ", Time:" << dur
            << ", QPS:" << queries / dur << std::endl;

  ODLA_RET_CHECK(odla_DestroyContext(ctx),1);
  ODLA_RET_CHECK(odla_DestroyExecutable(exec),1);
  ODLA_RET_CHECK(odla_DestroyDevice(device),1);

  std::cout << "Result verified\n";
  return 0;
}
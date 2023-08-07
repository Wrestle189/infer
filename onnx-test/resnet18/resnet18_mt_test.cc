#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <thread>
#include <atomic>

#include <ODLA/odla.h>
#include "../include/test_util.h"
#include "data/resnet18_data.h"

#ifndef COMPARE_ERROR
#define COMPARE_ERROR 5e-2
#endif

static bool error_flag = false;

bool Verify(const std::vector<float> &output, const std::vector<float> &gt) {
  bool accuracy = true;
  int output_size = gt.size(); 
  int batch = output.size() / output_size;

  for (int i = 0; i < batch; i++) {
    accuracy &= Verify(output.data() + i * output_size, gt.data(),
                       gt.size(), COMPARE_ERROR);
  }
  return accuracy;
}

class Worker {
 private:
  bool Init(int batch, void* arg0_data) {
    input_data[0] = arg0_data;
    output.resize(batch * test_output_gt.size());
    output_data[0] = output.data();

    for (int i = 0; i < input_vals.size(); ++i) {
        ODLA_RET_CHECK(odla_GetArgFromExecutableByIdx(exec, i, &input_vals[i]),false);
    }
    for (int i = 0; i < output_vals.size(); ++i) {
        ODLA_RET_CHECK(odla_GetOutputFromExecutableByIdx(exec, i, &output_vals[i]),false);
    }

    ODLA_RET_CHECK(odla_CreateContext(&ctx), false);
    ODLA_RET_CHECK(
        odla_SetContextItem(ctx, ODLA_RUN_BATCH_SIZE, (odla_item_value)&batch), false);
    return true;
  }

 public:
  // All instances share an executable.
  Worker(odla_device device, odla_executable exec, int batch, std::atomic_int& cnts,
         void* arg0_data, int total_iters)
      : device(device), 
        exec(exec),
        cnts(cnts),
        total_iters(total_iters),
        share_exec(true) {
    Init(batch, arg0_data);
  }
  // Each instance uses its own executable.
  Worker(odla_device device, odla_resource_location loc, int batch,
         std::atomic_int& cnts, void* arg0_data, int total_iters)
      : device(device),
        cnts(cnts),
        batch(batch),
        total_iters(total_iters),
        share_exec(false) {

    if (odla_LoadExecutable(loc, device, &exec) != ODLA_SUCCESS){
      error_flag = true;
      constructor_error_flag = true;
      return;
    }

    Init(batch, arg0_data);
  }

  ~Worker() {
    
    if (!constructor_error_flag){
      ODLA_RET_CHECK(odla_DestroyContext(ctx), );
    }
    if (!share_exec) {
      ODLA_RET_CHECK(odla_DestroyExecutable(exec), );
    }
  }

  bool Wait() {
    if (thread.joinable()) {
      thread.join();
    }
    return accuracy;
  }

  void Run() {
    while ((cnts.fetch_add(1) < total_iters) && !error_flag) {
      for (int idx = 0; idx < input_data.size(); ++idx) {
        ODLA_RET_CHECK(odla_BindToArgument(input_vals[idx], input_data[idx], ctx),);
      }
      ODLA_RET_CHECK(odla_LaunchExecutable(exec, ctx),);

      for (int idx = 0; idx < output_data.size(); ++idx) {
        ODLA_RET_CHECK(odla_GetValueData(output_vals[idx], output_data[idx], ctx),);
      }

      accuracy &= Verify(output, test_output_gt);
    }
  }

  void RunAsync() {
    thread = std::thread([this]() { this->Run(); });
  }

 private:
  std::atomic_int &cnts;
  std::thread thread;
  int total_iters;
  int batch = 1;
  bool share_exec;

  odla_executable exec = nullptr;
  odla_context ctx = nullptr;
  odla_device device = nullptr;

  bool accuracy = true;
  bool constructor_error_flag = false;

  std::vector<float> output;
  std::array<void*, 1> input_data;
  std::array<void*, 1> output_data;
  std::array<odla_value, 1> input_vals;
  std::array<odla_value, 1> output_vals;
};

int main(int argc, char** argv) {

  if (argc < 5) {
    std::cout << "Usage: " << "$exe batch iters thread_num enginefile DYN(opt)" << std::endl;
    std::cout << "Use OEF eg: " << "/tmp/exes/resnet 4 1000 4 /tmp/oefs/resnet.oef DYN " << std::endl;
    std::cout << "Use trt-engine eg: " << "/tmp/exes/resnet 4 10000 4 /host/trt-engines/engines/resnet-dyn-max_b512-fp16-NV_A10.engine DYN" << std::endl;
    return 1;
  }

  int batch = atoi(argv[1]);
  int total_iters = atoi(argv[2]);
  int num_threads = atoi(argv[3]);
  std::string trt_engine_file = std::string(argv[4]);
  bool dyn_batch = argc >= 6 && std::string(argv[5]) == "DYN";

  if (batch > 1) {
    auto resize = [batch](auto& data) {
      auto n = data.size();
      data.resize(n * batch);
      for (int i = 1; i < batch; ++i) {
        std::copy(data.begin(), data.begin() + n, data.begin() + i * n);
      }
    };
    resize(test_input);
  }


  // timestamp
  auto total_start = Now();

  odla_vendor vendor = nullptr;
  odla_device_name device_name = ODLA_DEVICE_NVIDIA_TENSORRT;
  int device_idx = 0;
  odla_device device = nullptr;
  ODLA_RET_CHECK(odla_AllocateDevice(vendor, device_name, device_idx, &device), 1);

  odla_resource_location loc;
  loc.location_type = ODLA_LOCATION_PATH;
  loc.location = trt_engine_file.c_str();

  std::atomic_int cnt{0};
  std::vector<std::unique_ptr<Worker>> workers;

  odla_executable exec = nullptr;

  std::cout << "Dynamic Batch:" << dyn_batch << "\n";

  ODLA_RET_CHECK(odla_LoadExecutable(loc, device, &exec), 1);
    
  for (int i = 0; (!error_flag) && (i < num_threads); ++i) {
    std::unique_ptr<Worker> w;
    if (dyn_batch && i > 0) {
      w = std::make_unique<Worker>(device, loc, batch, cnt, test_input.data(),
                                   total_iters);
    } else {
      w = std::make_unique<Worker>(device, exec, batch, cnt, test_input.data(),
                                   total_iters);
    }

    workers.push_back(std::move(w));
  }

  for (auto& worker : workers) {
    worker->RunAsync();
  }

  bool accuracy = true;
  for (auto& worker : workers) {
    accuracy &= worker->Wait();
  }

  // timestamp
  auto total_end = Now();
  auto total_time = GetDuration(total_start, total_end);

  int64_t queries = total_iters * batch;
  int QPS = (int)(queries / total_time);

  workers.clear();

  ODLA_RET_CHECK(odla_DestroyExecutable(exec), 1);

  ODLA_RET_CHECK(odla_DestroyDevice(device), 1);

  if (accuracy && !error_flag) {
    std::cout << "Batchsize: " << batch << std::endl;
    std::cout << "Iterations: " << total_iters << std::endl;
    std::cout << "Queries: " << queries << std::endl;
    std::cout << "Threads: " << num_threads << std::endl;
    std::cout << "Total time: " << total_time << "s" << std::endl;
    std::cout << "QPS: " << QPS << std::endl;
    std::cout << "Result verified\n";
    return 0;
  }
  std::cout << " Failed\n";
  return 1;
}

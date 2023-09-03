

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os
import time
import sys
# Define a function to load the ONNX model and convert it to a TensorRT engine
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
ERROR = 1e-2

# Define a function to load the TensorRT engine and perform inference
def inference_with_tensorrt(engine_file_path, input_data, output_std):
    output_data = np.empty((1, 166, 2, 5531), dtype = np.float32)
    flag = True
    with open(engine_file_path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        with engine.create_execution_context() as context:
            # Allocate GPU memory
            d_input = cuda.mem_alloc(input_data.nbytes)
            d_output = cuda.mem_alloc(output_data.nbytes)
            
            # Create a CUDA stream
            stream = cuda.Stream()
            
            # Copy the input data from the host memory to the GPU
            cuda.memcpy_htod_async(d_input, input_data, stream)
            
            # Execute inference
            context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
            
            # Copy the output data from GPU memory back to the host memory
            cuda.memcpy_dtoh_async(output_data, d_output, stream)

            # compare the output results with standard output
            for i in range(1):
                for j in range(166):
                    for k in range(2):
                        for l in range(5531):
                            if abs(output_data[i][j][k][l] - output_std[i][j][k][l]) > ERROR:
                                flag = False
                                break
            
            # Synchronize and wait for the CUDA stream to finish
            stream.synchronize()
    
    if(not flag):
        print("ERROR: The output results are not correct!")
    else:
        print("The output results are correct!")
    return flag
    # print(output_data)


def get_engine(onnx_file_path, batch_size,engine_file_path=""):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:
            config.max_workspace_size = 1 << 28 # 256MiB
            builder.max_batch_size = 2
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [batch_size, 3, 32, 665]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    # if os.path.exists(engine_file_path):
    #     # If a serialized engine exists, use it instead of building an engine.
    #     print("Reading engine from file {}".format(engine_file_path))
    #     with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    #         return runtime.deserialize_cuda_engine(f.read())
    # else:
    return build_engine()
   
def resize_data_for_batch(data, batch):
    # if batch > 1:
    #     n = len(data)
    #     # for i in range(1, batch):
    #         data[n: 2 * n] = data[:n]
    return data   
    
def get_input_data(input_file,batch_size):
    with open(input_file, "r") as file:
        data_lines = file.readlines()

    # for item in data_lines:
    #     if not item:
    #         print("----")
    #         continue
    #     print(item.strip().replace(",", ""))
    #     # print("------")
    #     print(float(item.strip().replace(",", "")))
    # for item in data_lines:
    #     # 过滤 // 注释, 删除其后的内容
    #     if "//" in item:
    #         item = item.split("//")[0]
        # 过滤 # 注释, 删除其后的内容

    custom_input_data = [float(line.split("//")[0].strip().replace(",", "")) for line in data_lines]
    # input_shape = session.get_inputs()[0].shape
    print("length of custom_input_data",len(custom_input_data))

    input_shape = [batch_size, 3, 32, 665]

    # 根据batchsize调整input_data的大小
    custom_input_data = resize_data_for_batch(custom_input_data, batch_size)
    input_data = np.array(custom_input_data, dtype=np.float32).reshape(input_shape)    
    return input_data

def get_output_data(output_file, batch_size):
    with open(output_file, "r") as file:
        data_lines = file.readlines()
    
    custom_output_data = [float(line.split("//")[0].strip().replace(",", "")) for line in data_lines]

    output_data = np.array(custom_output_data, dtype=np.float32).reshape([1, 166, 2, 5531])
    return output_data

if __name__ == "__main__":
    # if len(sys.argv) < 3:
    #     print("用法：python script.py 参数1 参数2")
    #     sys.exit(1)  # 退出脚本，返回状态码 1 表示错误

    # 获取命令行参数
    # model_name=sys.argv[1]
    
    # batch = sys.argv[2]
    # total_iters = sys.argv[3]
    # num_threads = sys.argv[4]
    # trt_engine_file = sys.argv[5]
    model_name="crnn"
    engine_file_path = "/vodla_workload/engines_self/"+model_name+".engine"
    onnx_file_path="/vodla_workload/"+model_name+"/models/crnn_lite_lstm.onnx"
    input_file="/vodla_workload/"+model_name+"/data/crnn_input2.txt"
    # output_std_file="/vodla_workload/"+model_name+"/data/crnn_output.txt"
    batch_size=1
    num_inferences = 1  # 假设您要进行10次推理
    
    input_data=get_input_data(input_file,batch_size)        
    # output_std = get_output_data(output_std_file, batch_size)
    output_std = np.empty((1, 166, 2, 5531), dtype = np.float32)

    get_engine(onnx_file_path,batch_size, engine_file_path)
    # engine_file_path="/resnet18/resnet-dyn-max_b512-fp16-NV_V100.engine"
    
    inference_times = []  # 用于存储每次推理的运行时间

    for i in range(num_inferences):
        start_time = time.time()  # 开始计时

        inference_with_tensorrt(engine_file_path, input_data, output_std)

        end_time = time.time()  # 结束计时
        inference_time = end_time - start_time
        inference_times.append(inference_time)    

        # 输出每次推理的结果
        # print("Inference {}: Output shape: {}, Output data: {}".format(i+1, outputs[0].shape, outputs[0]))

    average_inference_time = sum(inference_times) / (batch_size * num_inferences)

    print("Batch_size: {}, Num_inferences: {}, All time: {:.6f} seconds, Average_inference_time:{:.6f}".format(batch_size, num_inferences, sum(inference_times),average_inference_time))

import torch
import onnxruntime as rt
import numpy as np
import time

def resize_data_for_batch(data, batch):
    if batch > 1:
        n = len(data)
        data = data * batch
        for i in range(1, batch):
            data[i * n: (i + 1) * n] = data[:n]
    return data

# 加载ONNX模型，并指定使用CUDAExecutionProvider执行提供者
model_path = "/resnet18/models/resnet18-v2-7.onnx"  # 替换为您的ONNX模型文件路径
providers = ['CUDAExecutionProvider']
session = rt.InferenceSession(model_path, providers=providers)

# 确保您的GPU设备被识别
print("Available GPUs:", rt.get_available_providers())

# 准备自定义的文本数据并转换为适用于模型输入的张量形式
# 假设您有一个名为"custom_data.txt"的文本文件，每行包含一个数字（示例）
with open("/resnet18/data/data_input.txt", "r") as file:
    data_lines = file.readlines()
custom_input_data = [float(line.strip().replace(",", "")) for line in data_lines]

# 将输入数据转换为NumPy数组，并根据模型的输入要求进行适当的调整
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

print(input_shape)

batch_size = 100
input_shape = [batch_size, 3, 224, 224]

# 根据batchsize调整input_data的大小
custom_input_data = resize_data_for_batch(custom_input_data, batch_size)
input_data = np.array(custom_input_data, dtype=np.float32).reshape(input_shape)

# 执行推理
output_name = session.get_outputs()[0].name

inference_times = []  # 用于存储每次推理的运行时间

# 设置进行推理的次数
num_inferences = 1  # 假设您要进行10次推理

for i in range(num_inferences):
    start_time = time.time()  # 开始计时

    outputs = session.run([output_name], {input_name: input_data})

    end_time = time.time()  # 结束计时
    inference_time = end_time - start_time
    inference_times.append(inference_time)

    # 输出每次推理的结果
    # print("Inference {}: Output shape: {}, Output data: {}".format(i+1, outputs[0].shape, outputs[0]))

# 输出平均推理时间
average_inference_time = sum(inference_times) / len(inference_times)

print("Batch_size: {}, Num_inferences: {}, All time: {:.6f} seconds".format(batch_size, num_inferences, sum(inference_times)))

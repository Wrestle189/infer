# 基于NVIDIA的TensorRT镜像构建Docker容器
FROM nvcr.io/nvidia/tensorrt:21.11-py3

# 设置工作目录
WORKDIR /app

# 将当前目录中的代码复制到容器中的/app目录下
# COPY . /app

# 安装依赖
RUN pip install numpy torch

# 运行Python脚本
# CMD ["python", "your_tensorrt_inference_script.py"]

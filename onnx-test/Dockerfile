# 基础镜像，选择一个包含Python的镜像
FROM python:3.9

# 安装所需的Python库和ONNX Runtime
RUN pip install torch torchvision onnx optimum[onnxruntime-gpu]

# 将您的应用程序代码复制到容器中
# COPY run-resnet18.py /app/run-resnet18.py

# 设置工作目录
WORKDIR /resnet18

# 运行应用程序
# CMD ["python", "run-resnet18.py"]
CMD ["/bin/bash"]


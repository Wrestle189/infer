sudo docker run --rm --runtime=nvidia --gpus all -v /home/jinzhen/infer/onnx-test/resnet18:/resnet18 -v /home/jinzhen/infer/tensorrt:/code  -v /home/jinzhen/vodla/vodla_workload:/vodla_workload  -it run-trt-image \
# python /code/run-trt.py 

# sudo docker run --rm --runtime=nvidia --gpus all -v /home/jinzhen/infer/onnx-test/resnet18:/resnet18 -v /home/jinzhen/infer/tensorrt:/code  -v /home/jinzhen/vodla/vodla_workload:/vodla_workload  -it run-trt-image
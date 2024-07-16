# <div align="center"> YOLO Model Conversion </div>

## A. Introduction
This part provides a framework for converting PyTorch model formats into TensorRT and ONNX, primarily servicing the YOLOv5 architecture.

## B. Feature
- Supports multiple model formats: ONNX, PyTorch, TensorRT, and Torch2trt.
- Easy management of multiple models with automatic handling of device allocation (CPU/GPU).

## C. Prerequisites
- Python 3.6 or higher
- OpenCV
- PyTorch (with CUDA support)
- torchvision
- NumPy
- yaml
- ONNX Runtime (for ONNX models. and >= python3.8)
- TensorRT (for TensorRT models)
- CUDA and cuDNN (for GPU support)

## D. Usage
### 1. Set Up Configuration
#### PyTorch -> ONNX -> TensorRT Configurations
Main parameters are set in a YAML file, edit this file to configure the conversion:
```YAML
model_converter:
    model_path: "yolov5s.pt" # Path to your model
    image_path: "test.jpg" # Example image for the conversion process, optional
    onnx_output_path: "yolov5s.onnx" # Filename for the ONNX format output
    trt_output_path: "yolov5s.engine" # Filename for the TensorRT format output
    work_space: 2 # GB # Space setting for TensorRT optimization strategy
    image_size: # Inference size after conversion, consistent with the original model
        - 640 
        - 640
    batch_size: 1 # Batch size format after conversion
    dynamic: False # Enable dynamic batch size
    simplify: False # Simplify ONNX model
    half: False # Convert to half precision
    opset_version: 12
```
Start the model conversion process:
```python=
from modelconvert import ModelConverter
modelconverter = ModelConverter('./configs/example/convert.yaml')
modelconverter.convert_torch_to_onnx() # torch->onnx
modelconverter.convert_onnx_to_tensorrt() # torch->onnx->tensorrt
```
#### PyTorch -> TensorRT Configurations
```python=
from modelconvert import torch2trt
model_path = 'yolov5s.pt'
trt_path = 'yolov5s_trt.pth'
input_shape=(1, 3, 640, 640)
fp16_mode=False

torch2trt(model_path, trt_path, input_shape, fp16_mode)
```
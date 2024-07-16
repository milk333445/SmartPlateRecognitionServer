# <div align="center"> YOLO Detector </div>

## A. Introduction
The YOLO Detection Package provides a versatile and user-friendly platform for object detection, supporting YOLO models in formats like ONNX, PyTorch, and TensorRT. With its **DetectorBuilder** class, users can easily manage multiple models through a YAML configuration, offering the flexibility to run detections globally or on individual models to meet diverse needs.

## B. Feature
- Supports multiple model formats: ONNX, PyTorch, TensorRT, and Torch2trt.
- Customizable detection settings such as confidence thresholds and IOU thresholds.
- Easy management of multiple models with automatic handling of device allocation (CPU/GPU).

## C. Prerequisites
- Python 3.6 or higher
- OpenCV
- PyTorch (with CUDA support)
- torchvision
- NumPy
- yaml
- ONNX Runtime (for ONNX models)
- TensorRT (for TensorRT models)
- CUDA and cuDNN (for GPU support)

## D. Usage
### 1. Set Up Configuration
Create a YAML configuration file specifying the model paths, sizes, thresholds, and other settings. Example structure for config.yml:
```YAML
model_A:
    weight: 'path/to/yolov5s.pt'
    size: 640
    conf_threshold: 0.25
    iou_threshold: 0.45
    classes: [0, 1]  # Optional: specific classes to detect
    fp16: false

model_B:
    weight: 'path/to/another_yolov5s.pt'
    size: 640
    conf_threshold: 0.5
    iou_threshold: 0.5
    classes: 
    fp16: false
...
```
**Model type:**
- **.pt / .pth**: original PyTorch model
- **.onnx**: ONNX
- **.engine**: TensorRT
- **_trt.pt / _trt.pth**: TensorRT from torch2trt

&nbsp;

### 2. Load and Run
Use the DetectorBuilder class to load the models specified in your configuration file.  Then, run detection on your images as shown in the following example:
```python
from nexva.detect import DetectorBuilder

# Initialize the detector
detector = DetectorBuilder('config.yml')

# Sample image loading (add your own image loading mechanism)
images = [cv2.imread('path/to/image.jpg')]

# Run all detection models
detector.run(images)

# Or execute detection using only the 'model_A' specified by its key in the YAML configuration
detector.model_A.run(images)

# Clean up resources
detector.destroy_detectors()
```
**API reference:**
- **DetectorBuilder(file)**: Class to manage the loading and running of specified detection models based on a YAML configuration file.
- **run(images, roi_list=None)**: Method to process a list of images through the detection models. Optionally accepts a list of regions of interest.
    + images: list of image nparrays.
    + roi_list: list of bboxes with the format [x1, y1, x2, y2] or None.
- **run.{model_name}(images, roi_list=None)**: Method to process a list of images through a specified models. Optionally accepts a list of regions of interest.
- **destroy_detectors()**: Method to clean up loaded models and free resources.
# LPR Engine
## Links to the overall organization
https://gitmind.com/app/docs/fk4m7tt9
## Installation
- To install the LPR Server, clone the repository and related tool packages as shown below:
```python=
https://github.com/milk333445/SmartPlateRecognitionServer.git
```
## Configurations
### UDP Configurations
- Set up IP addresses and ports for both the server and the client to facilitate communication.
#### Client Configuration
```YAML
./configs/lprclient_config.yaml
LPR_UDPClient:
  host: "10.0.0.xx" # your own ip
  port: 5000
  draw: False # Visualize inference results
  timeout: 10 # UDP send time limit
  send_mode: 'path' # 'image' sends the entire image; 'path' sends the image path
```
#### Server Configuration
```YAML
./configs/lprserver_config.yaml
LPR_UDPServer:
  host: "10.0.0.53"
  port: 5000
  buffer_timeout: 10 # Data processing time limit
  recv_buffer_size: 1048576 # UDP receive buffer size, larger sizes stabilize packet reception
  send_buffer_size: 1048576 # UDP receive buffer size, larger sizes stabilize packet reception
  perspect_transform: False # Apply perspective transformation to license plates for 300 * 100 inference
  continuous_learning:
    reload_port: 8001
    enable: False
    conf_threshold: 0.7
    save_path: './continuous_learning_dataset'
    train_file_path: '/usr/src/app/yolov5'
    train_data_path: '/usr/src/app/yolov5/data'
    train_script_path: '/usr/src/app/yolov5/train.py'
    images_threshold: 100
    check_interval_hours: 2 # 2小時
    cooldown_period_hours: 12 # 12小時
    training_windows:
      - start: "02:00"
        end: "04:00"
      - start: "05:00"
        end: "07:00"  
    train_ratio: 0.95
    train_model_configs_path: '/usr/src/app/lpr_server_v0.1/train_configs/yolov5n.yaml'
    train_data_configs_path: '/usr/src/app/lpr_server_v0.1/train_configs/coco.yaml'
    pretrain_model_weights_path: '/usr/src/app/lpr_server_v0.1/lpr_engine/lpr_weight/char_best_224.pt'
    train_hyp_configs_path: '/usr/src/app/lpr_server_v0.1/train_configs/hyp.scratch-low.yaml'
    train_epochs: 1
    train_batch_size: 8
    train_img_size: 224
    tmp_train_model_weights_path: '/usr/src/app/lpr_server_v0.1/tmp_train_model_weights'
    test_model_results_path: './test_results'
    test_dataset_path: './test_dataset/labeled_images'
    tmp_detection_config_path: './train_configs/detection.yaml'
```



### LPR Engine Configurations
- Supports deploying multiple models simultaneously, including popular formats like .pt, .onnx, .pth (TRT), and .engine (TensorRT).
- Ensure that the input size matches when using .engine (TensorRT) format and onnx engine. 
- Since this is a license plate recognition task, it is assumed that there are weights for both license plate detection and character recognition.
- Supports batch sizes smaller than the fixed batch size for TensorRT inference (consistent batch sizes are recommended for speed).
```YAML
./lpr_engine/configs/detection.yaml
car_plate:
    weight: './lpr_engine/lpr_onnx_trt_weight/bbox_best_224_bt2.engine'
    size: 224
    conf_threshold: 0.5
    iou_threshold: 0.45
    fp16: False # Half-precision inference
    classes: 

car_char:
    weight: './lpr_engine/lpr_onnx_trt_weight/char_best_224_bt2.engine'
    size: 224
    conf_threshold: 0.5
    iou_threshold: 0.45
    fp16: False
    classes: 
```
### Model Conversion Configurations
- Configure model conversion settings to convert models to ONNX or TensorRT engine formats:
```YAML
./lpr_engine/lpr_engine/configs/convert.yaml
model_converter:
  model_path: "./lpr_weight/bbox_best_224.pt"
  image_path: "tmp.jpg" # Example image for conversion, optional
  onnx_output_path: "bbox_best_224_bt2.onnx"
  trt_output_path: "bbox_best_224_bt2.engine"
  work_space: 6 # GB # Maximum temporary memory space available for TensorRT during model optimization and execution
  image_size: 
    - 224
    - 224
  batch_size: 2 
  dynamic: False # Most TensorRT versions do not support dynamic conversion (no INT64 support)
  simplify: False
  half: False # Convert to half-precision
  opset_version: 12
```
## Quick Start
### Convert Model
- To convert a model to both ONNX and TensorRT formats, use the following code:
```python=
from lprtools.lprtools.convert import ModelConverter
config_path = 'configs/convert.yaml'
model_converter = ModelConverter(config_path)
# to onnx and tensorrt at the same time
model_converter.convert_onnx_to_tensorrt()
```

### Initialize LPR Server
- Create an LPR Server instance and start listening:
```python=
from lprserver import LPR_UDPServer
server = LPR_UDPServer(config_path='./configs/lprserver_config.yaml')
server.listen()
```
### Create Client
- Set up a client to send images for license plate recognition:
```python=
from lprclient import LPR_UDPClient
client = LPR_UDPClient(config_path='./configs/lprclient_config.yaml')
img_path = 'tmp.jpg'
img = cv2.imread(img_path)
images = [img_path] or [img] # Choose to send the image path or image itself based on settings in lprclient_config.yaml
# Send the following format, please follow the instructions
- 'mode'
- 'timestamp'
- 'images'
# send udp
mode = 'lprtw'
timestamp = datetime.now().strftime("%y%m%d%H%M%S%f")[:-3]
client.send(images_path, mode, timestamp)
client.close()
```
## Using Continuous learning
#### Continuous learning Configuration
**Make sure you have your training scripts ready**
```YAML
LPR_UDPServer:
  host: "10.0.0.53"
  port: 5000
  buffer_timeout: 10 # Data processing time limit
  recv_buffer_size: 1048576 # UDP receive buffer size, larger sizes stabilize packet reception
  send_buffer_size: 1048576 # UDP receive buffer size, larger sizes stabilize packet reception
  perspect_transform: False # Apply perspective transformation to license plates for 300 * 100 inference
  continuous_learning:
    reload_port: 8001 # Communication between cls and lprserver
    enable: True # Whether to open continuous learning server
    conf_threshold: 0.7 # results conf of training data collection
    save_path: './continuous_learning_dataset' # Path to store training data
    train_file_path: '/usr/src/app/yolov5' # Training folder path (absolute path)
    train_data_path: '/usr/src/app/yolov5/data'
    train_script_path: '/usr/src/app/yolov5/train.py'
    images_threshold: 100
    check_interval_hours: 2 # Checks every two hours.
    cooldown_period_hours: 12 # There will be a 12-hour break after training.
    training_windows: # Trainable Time Window
      - start: "02:00"
        end: "04:00"
      - start: "05:00"
        end: "07:00"  
    train_ratio: 0.95 # Training Test Data Cut ratio
    train_model_configs_path: '/usr/src/app/lpr_server_v0.1/train_configs/yolov5n.yaml' # Customized training configurations
    train_data_configs_path: '/usr/src/app/lpr_server_v0.1/train_configs/coco.yaml' 
    pretrain_model_weights_path: '/usr/src/app/lpr_server_v0.1/lpr_engine/lpr_weight/char_best_224.pt'
    train_hyp_configs_path: '/usr/src/app/lpr_server_v0.1/train_configs/hyp.scratch-low.yaml'
    train_epochs: 1 
    train_batch_size: 8
    train_img_size: 224
    tmp_train_model_weights_path: '/usr/src/app/lpr_server_v0.1/tmp_train_model_weights' # The new model weights will go here first after the training is complete.
    test_model_results_path: './test_results' # path to save test model results
    test_dataset_path: './test_dataset/labeled_images' # Test model good or bad dataset (named correct license plate number)
    tmp_detection_config_path: './train_configs/detection.yaml' # This yaml controls a new model that has just been trained.
```

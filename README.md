# Hello Object Detection
This is a tutorial to help you setup Object Detection using `TensorFlow API 1.x`.

*For latest setup process, please refer to the following sites:*
-   [tensorflow/models/object_detection/g3doc/installation.md](<https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md>)
-   [tensorflow/models/object_detection](<https://github.com/tensorflow/models/tree/master/research/object_detection>)
-   [tensorflow/models](<https://github.com/tensorflow/models>)

## Installation

### Setup Tensorflow-GPU
Please refer to the official tutorials:
-   [TensorFlow GPU Support](<https://www.tensorflow.org/install/gpu>)
-   [TensorFlow GPU Support (w/o VPN)](<https://tensorflow.google.cn/install/gpu>)

### Setup PyPi Libraries
Run the following command in shell:
```shell script
# Assume current worspace = ./
python -m pip install --user -r requirements.txt
```

### Setup COCO API
*For latest setup process, please refer to the official site [cocodataset/cocoapi](<https://github.com/cocodataset/cocoapi>).*

1.  Download `cocodataset/cocoapi/PythonAPI` folder.

2.   Extract `PyhonAPI` folder so that the directory looks like:
      ```text
      Hello-Object-Detection
      |-- cocoapi
      |   |-- PythonAPI
      |       |-- setup.py
      |       |-- ...
      |-- main.py
      |-- ...
      ```

3.  ***IMPORTANT!*** For `Windows` users, modify file `./cocoapi/PythonAPI/setup.py` as follow:
    - Delete the line that contains `extra_compile_args`.

4.  Build and install `pycocotools`:
Run the following command in shell:

#### `Windows` users:
```shell script
# Assume current worspace = ./cocoapi/PythonAPI
python setup.py build_ext --inplace
```

#### `Unix` users:
```shell script
# Assume current worspace = ./cocoapi/PythonAPI
make
```
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
Run the following command in shell from folder `./`:
```shell script
python -m pip install --user -r requirements.txt
```

### Setup COCO API
*For latest setup process, please refer to the official site [cocodataset/cocoapi](<https://github.com/cocodataset/cocoapi>).*

1.  Download `cocodataset/cocoapi/PythonAPI` and `cocodataset/cocoapi/common` folder.

2.  Extract `PyhonAPI` and `common` folder so that the directory looks like:
    ```text
    Hello-Object-Detection
    |-- cocoapi
    |   |-- common
    |       |-- maskApi.c
    |       |-- ...
    |   |-- PythonAPI
    |       |-- setup.py
    |       |-- ...
    |-- main.py
    |-- ...
    ```

3.  ***IMPORTANT!*** For `Windows` users, modify file `./cocoapi/PythonAPI/setup.py` as follow:
    - Delete the line that contains `extra_compile_args`.

4.  Build and install `pycocotools`:

    Run the following command in shell from folder `./cocoapi/PythonAPI`:
    
    #### `Windows` users:
    ```shell script
    python setup.py build_ext --inplace
    ```
    
    #### `Unix` users:
    ```shell script
    make
    ```
    
5. Successful output will output `... -> pycocotools` as the last line of output.

### Install Protocol Buffer
1.  Download the corresponding binary zipped release from [protocolbuffers/protobuf](<https://github.com/protocolbuffers/protobuf/releases>).
2.  Unzip the folder.
3.  Add `protoc` executable to Environment Variable PATH then restart terminal. (Or you will need to replace every `protoc` call with `PATH_TO_PROTOC/protoc` later.)

### Compile protobuf models and parameters
1.  Download `tensorflow/models/research` folder.

2.  Extract `research` folder so that the directory looks like:
    ```text
    Hello-Object-Detection
    |-- models
    |   |-- research
    |       |-- setup.py
    |       |-- ...
    |-- main.py
    |-- ...
    ```

3.  Run the following command in shell from folder `./models/research`:
    ```shell script
    protoc object_detection/protos/*.proto --python_out=.
    ```
    
4.  Successful execution will output nothing.

### Add Libraries to `PYTHONPATH`
The following path shall be appended to Environment Variable `PYTHONPATH`:
1.  `__PWD__/models/research`
2.  `__PWD__/models/research/slim`

***Note: change `__PWD__` to the absolute path of `Hello-Object-Detection` folder before touching `PYTHONPATH`!***

### Test Installation
1.  Run the following command in shell from folder `./models/research`:
    ```shell script
    python object_detection/builders/model_builder_test.py
    ```
    
2.  Successful execution will output `OK` or `OK (skipped=...)` as the last line of output.

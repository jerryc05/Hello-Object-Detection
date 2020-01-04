# Hello Object Detection
This is a tutorial to help you setup Object Detection using `TensorFlow Object Detection API`.

Author: [@jerryc05](<https://github.com/jerryc05>)

*For latest setup process, please refer to the following sites:*
-   [tensorflow/models/object_detection](<https://github.com/tensorflow/models/tree/master/research/object_detection>)
-   [tensorflow/models](<https://github.com/tensorflow/models>)

**Table of Contents:**
1.  [Installation](#installation)
    1.  [Setup Tensorflow-GPU](#setup-tensorflow-gpu)
    2.  [Setup PyPi Libraries](#setup-pypi-libraries)
    3.  [Setup COCO API](#setup-coco-api)
    4.  [Install Protocol Buffer](#install-protocol-buffer)
    5.  [Compile protobuf models and parameters](#compile-protobuf-models-and-parameters)
    6.  [Add Libraries to PYTHONPATH](#add-libraries-to-pythonpath)
    7.  [Test Installation](#test-installation)
2.  [Label Image](#label-image)
    1.  [Install labelImg](#install-labelimg)
    2.  [Use labelImg](#use-labelimg)
    3.  [Parse labels](#parse-labels)

## Installation
*For latest setup process, please refer to the following site:*
-   [tensorflow/models/object_detection/g3doc/installation.md](<https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md>)

### Setup Tensorflow-GPU
Please refer to the official tutorials:
-   [TensorFlow GPU Support](<https://www.tensorflow.org/install/gpu>)
-   [TensorFlow GPU Support (Mainland China)](<https://tensorflow.google.cn/install/gpu>)

### Setup PyPi Libraries
Run the following command in shell from folder `./`:
```shell script
python -m pip install --user -r requirements.txt
```

### Setup COCO API
*For latest setup process, please refer to the official site [cocodataset/cocoapi](<https://github.com/cocodataset/cocoapi>).*

1.  Download `cocodataset/cocoapi/PythonAPI` and `cocodataset/cocoapi/common` folders.

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

## Label Image
*For latest setup process, please refer to the following sites:*
-   [tzutalin/labelImg](<https://github.com/tzutalin/labelImg>)

### Install labelImg
*You should have this installed already via [1.2) Setup PyPi Libraries](#setup-pypi-libraries).*

### Use labelImg
*No Bullshit here, please refer to official site [tzutalin/labelImg](<https://github.com/tzutalin/labelImg>).*

### Parse labels
1.  Parse labels from `xml` to `csv`:
    
    Run the following command in shell from folder `./`:
    ```shell script
    python xml_to_csv.py -i __PATH_TO_XML__ -o __PATH_TO_CSV__
    ```
    ***Note: change the following paths before running the script:***
    1.  `__PATH_TO_XML__` <- path of xml labels created by `labelImg`. 
    2.  `__PATH_TO_CSV__` <- path of csv that the script will create. 

2.  Parse `csv` to `TFRecord`:
    
    Run the following command in shell from folder `./`:
    ```shell script
    python csv_to_tfrecord.py -i __PATH_TO_CSV__ -o __PATH_TO_TFRECORD__
    ```
    ***Note: change the following paths before running the script:***
    1.  `__PATH_TO_CSV__` <- path of csv file created by `xml_to_csv.py`. 
    2.  `__PATH_TO_TFRECORD__` <- path of TFRecord file that the script will create. 







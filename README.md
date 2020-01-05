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

#### `Windows` users:
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

3.  ***IMPORTANT!*** Modify the file `./cocoapi/PythonAPI/setup.py` as follow:
    - Delete the line that contains `extra_compile_args`.

4.  Build and install `pycocotools`:

    Run the following command in shell from folder `./cocoapi/PythonAPI`:
    ```shell script
    python setup.py install
    ```
5. Successful output will output `Finished processing dependencies for pycocotools==...` as the last line of output.

#### `Unix` users:
Run the following command in shell from folder `./`:                      
```shell script
python -m pip install pycocotools
```

### Install Protocol Buffer
1.  Download the corresponding binary zipped release from [protocolbuffers/protobuf](<https://github.com/protocolbuffers/protobuf/releases>).
2.  Unzip the folder.
3.  Add `protoc` executable to Environment Variable PATH then restart terminal. (Or you will need to replace every `protoc` call with `PATH_TO_PROTOC/protoc` later.)

### Compile protobuf models and parameters
1.  Download `tensorflow/models/research/object_detection` and `tensorflow/models/research/slim` folders.

2.  Extract `object_detection` and `slim` folders so that the directory looks like:
    ```text
    Hello-Object-Detection
    |-- models
    |   |-- research
    |       |-- object_detection
    |           |-- model_main.py
    |           |-- ...
    |       |-- slim
    |           |-- setup.py
    |           |-- ...
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
*For latest setup process, please refer to site: [tzutalin/labelImg](<https://github.com/tzutalin/labelImg>)*

### Install labelImg
*You should have this installed already via [1.2) Setup PyPi Libraries](#setup-pypi-libraries).*

### Use labelImg
*No Bullshit here, please refer to official site [tzutalin/labelImg](<https://github.com/tzutalin/labelImg>).*

### Parse labels
1.  Parse labels from `xml` to `csv`:

    Run the following command in shell from folder `./`:
    ```shell script
    python xml_to_csv.py -i ${PATH_TO_XML_FOLDER} -o ${PATH_TO_CSV_FOLDER}
    ```
    ***Note: change the following paths before running the script:***
    1.  `${PATH_TO_XML_FOLDER}` <- path of xml labels created by `labelImg`. 
    2.  `${PATH_TO_CSV_FOLDER}` <- path where csv file will be saved. 

2.  Parse `csv` to `TFRecord`:

    Run the following command in shell from folder `./`:
    ```shell script
    python csv_to_tfrecord.py -c ${PATH_TO_CSV_FILE} -i ${PATH_TO_IMG_FOLDER} -o ${PATH_TO_TFRECORD_FILE_FOLDER}
    ```
    ***Note: change the following paths before running the script:***
    1.  `${PATH_TO_CSV_FILE}` <- path of csv file created by `xml_to_csv.py`. 
    2.  `${PATH_TO_IMG_FOLDER}` <- path of image files. 
    3.  `${PATH_TO_TFRECORD_FILE_FOLDER}` <- path of TFRecord file that the script will create. 

    ***Note: you might need to run it twice for both `train.csv` and `eval.csv` respectively.***

## Start Training

### Configure Training
1.  Select a pre-configure model config from `./models/research/object_detection/samples/configs`.

2.  Copy the `.config` file to somewhere else.
    
    For example, I use `./training/faster_rcnn_inception_resnet_v2_atrous_coco.config`.
    
    So the directory looks like:
    ```text
    Hello-Object-Detection
    |-- training
    |   |-- faster_rcnn_inception_resnet_v2_atrous_coco.config
    |-- main.py
    |-- ...
    ```

3.  Open the config file.

4.  Replace the value of `num_classes` with your number of classes.

    If you forgot the number, check your `.pbtxt` file that contains label map.

5.  Replace the value of `fine_tune_checkpoint` with the path of model file to save.

    For example, I use `"./training/faster_rcnn_inception_resnet_v2_atrous_coco.ckpt"`.
    
6.  Replace the value of `input_path` under section `train_input_reader` with the path of 
training TFRecord file.

    *Note: wildcard char `?` can be used to match multiple file names.*

7.  Replace the value of `label_map_path` under section `train_input_reader` with the path 
of training label map `.pbtxt`file.

8.  Replace the value of `num_examples` with the number of images to train.
    
9.  Replace the value of `input_path` under section `eval_input_reader` with the path of 
eval TFRecord file.

    *Note: wildcard char `?` can be used to match multiple file names.*

10.  Replace the value of `label_map_path` under section `eval_input_reader` with the path 
of eval label map `.pbtxt`file.

### Run Training
Run the following command in shell from folder `./models/research/object_detection`:
```shell script
python model_main.py --pipeline_config_path=${PIPELINE_CONFIG_PATH} --model_dir=${MODEL_DIR} --alsologtostderr
```
***Note: change the following paths before running the script:***
1.  `${PIPELINE_CONFIG_PATH}` <- path of pre-configured model config file. 
2.  `${MODEL_DIR}` <- path where training checkpoints and events will be saved.






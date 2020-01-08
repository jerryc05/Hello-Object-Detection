r"""
An asyncio-powered csv to TFRecord parser. It also generates class ids and
    the label pbtxt for you automatically.

Author: @jerryc05 - https://github.com/jerryc05

Example Usage:
--------------
python csv_to_tfrecord.py \
       -c ${PATH_TO_CSV_FILE} \
       -i ${PATH_TO_IMG_FOLDER} \
       -o ${PATH_TO_TFRECORD_FILE_FOLDER}
"""

import argparse
import asyncio
import csv
import io
import os
import cv2
from typing import List, Dict, Any
from utils.log_helper import str_error


async def main():
    arg_parser = argparse.ArgumentParser(description=
                                         'An asyncio-powered csv to TFRecord parser. '
                                         'It also generates class ids and the label pbtxt '
                                         'for you automatically.')
    arg_parser.add_argument('-c', '--csv', required=True,
                            help='the path to folder containing input csv files.')
    arg_parser.add_argument('-i', '--img', required=True,
                            help='the path to folder containing image files.')
    arg_parser.add_argument('-o', '--output', required=True,
                            help='the path to folder containing output TFRecord file.')
    arg_parser.add_argument('-l', '--label',
                            help='the path to folder containing output TFRecord file.')
    args = arg_parser.parse_args()

    csv_path: str = args.csv.strip()
    if not os.path.isfile(csv_path):
        print(f'{str_error}\n CSV file not found!')
        print(f'Please check your csv input path again: [{csv_path}]!')
        exit(1)

    img_path: str = args.img.strip()

    tfrecord_path: str = args.output.strip()
    if os.path.isdir(tfrecord_path):
        print(f'{str_error}\n ${{tfrecord_path}} should be a file path, not folder path!')
        print(f'Please check your ${{tfrecord_path}} again: [{tfrecord_path}]!')
        exit(1)
    print(111)
    if args.label:
        label_path = args.label.strip()
    else:
        label_path = os.path.join(os.path.dirname(tfrecord_path), 'label_map.pbtxt')

    print(f'Processing {csv_path}!')

    class_list = []
    tasks: List[asyncio.Task] = []

    async def csv_to_pbtxt():
        with open(label_path, 'w') as f:
            for i, name in enumerate(class_list):
                f.write(f"item {{\n  id: {i + 1}\n  name: '{name}'\n}}\n")

    # parse text to id
    async def id_of_class(row_label: str):
        assert isinstance(row_label, str)
        if not class_list:
            _classes = set()
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    _classes.add(row['class'])
            class_list.extend(_classes)
            tasks.append(asyncio.create_task(csv_to_pbtxt()))
        return 1 + class_list.index(row_label)

    import tensorflow as tf
    try:
        import object_detection.utils.dataset_util as dataset_util
    except ImportError:
        print('Cannot import [object_detection.utils.dataset_util]!')
        print('Did you forget to append [./research] to [PYTHONPATH]?')
        exit(1)

    async def write_box_to_tfrecord(group: Dict[str, Any], tfrecord_writer):
        filename: str = group['filename']
        # with tf.gfile.GFile(os.path.join(img_path, filename), 'rb') as fid:
        #     encoded_img = fid.read()
        # image = PIL.Image.open(io.BytesIO(encoded_img))
        image = cv2.imread(os.path.join(img_path, filename))
        width, height, _ = image.size
        encoded_img = cv2.imdecode(image, cv2.IMREAD_COLOR)

        filename_b = filename.encode('utf8')
        image_format = filename_b.split(b'.')[-1].lower()
        xmins = [x / width for x in group['xmin']]
        xmaxs = [x / width for x in group['xmax']]
        ymins = [x / height for x in group['ymin']]
        ymaxs = [x / height for x in group['ymax']]
        classes_text = [x.encode('utf8') for x in group['class']]
        classes = [await id_of_class(x) for x in group['class']]

        tf_example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image/height'            : dataset_util.int64_feature(height),
                    'image/width'             : dataset_util.int64_feature(width),
                    'image/filename'          : dataset_util.bytes_feature(filename_b),
                    'image/source_id'         : dataset_util.bytes_feature(filename_b),
                    'image/encoded'           : dataset_util.bytes_feature(encoded_img),
                    'image/format'            : dataset_util.bytes_feature(image_format),
                    'image/object/bbox/xmin'  : dataset_util.float_list_feature(xmins),
                    'image/object/bbox/xmax'  : dataset_util.float_list_feature(xmaxs),
                    'image/object/bbox/ymin'  : dataset_util.float_list_feature(ymins),
                    'image/object/bbox/ymax'  : dataset_util.float_list_feature(ymaxs),
                    'image/object/class/text' : dataset_util.bytes_list_feature(classes_text),
                    'image/object/class/label': dataset_util.int64_list_feature(classes),
                }))
        tfrecord_writer.write(tf_example.SerializeToString())

    async def to_tfrecord(tfrecord_file_path):
        # data = collections.namedtuple('data', ['filename', 'object'])
        grouped: Dict[str, Dict[str, Any]] = {}
        with open(csv_path) as f:
            # examples = pd.read_csv(csv_path)
            for box in csv.DictReader(f):
                filename = box['filename']
                if filename not in grouped:
                    grouped[filename] = {
                        'filename': filename,
                        'width'   : box['width'],
                        'height'  : box['height'],
                        'class'   : [],
                        'xmin'    : [],
                        'ymin'    : [],
                        'xmax'    : [],
                        'ymax'    : []
                    }
                group = grouped[filename]
                group['class'].append(box['class'])
                group['xmin'].append(int(box['xmin']))
                group['ymin'].append(int(box['ymin']))
                group['xmax'].append(int(box['xmax']))
                group['ymax'].append(int(box['ymax']))

        writer_tasks = []
        with tf.python_io.TFRecordWriter(tfrecord_file_path) as tfrecord_writer:
            for group in grouped.values():
                writer_tasks.append(asyncio.create_task(
                    write_box_to_tfrecord(group, tfrecord_writer)))
            for task in writer_tasks:
                await task

        print()
        print('CSV -> TFRecord successful!')
        print(f'Processed [{len(grouped)}] files in total!')
        print(f'TFRecord file location: [{tfrecord_path}]!')
        print(f'Label pbtxt file location: [{label_path}]!')

    tasks.append(asyncio.create_task(
        to_tfrecord(tfrecord_path)))

    for task in tasks:
        await task


asyncio.run(main())
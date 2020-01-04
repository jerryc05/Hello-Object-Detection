# An asyncio-powered csv to TFRecord parser.
# Author: @jerryc05 - https://github.com/jerryc05

import argparse
import asyncio
import csv
from PIL import Image
import io
from collections import namedtuple
import os
import tensorflow as tf

try:
    import object_detection.utils.dataset_util as dataset_util
except ImportError:
    print('Cannot import [object_detection.utils.dataset_util]!')
    print('Did you forget to append [./research] to [PYTHONPATH]?')
    exit(1)


async def main():
    arg_parser = argparse.ArgumentParser(description='A csv to TFRecord parser.')
    arg_parser.add_argument('-c', '--csv', required=True,
                            help='the path to folder containing input csv files.')
    arg_parser.add_argument('-i', '--img', required=True,
                            help='the path to folder containing input csv files.')
    arg_parser.add_argument('-o', '--output', required=True,
                            help='the path to folder containing output TFRecord file.')
    args = arg_parser.parse_args()
    del arg_parser

    csv_path: str = args.csv.strip()
    if csv_path.endswith('/') or csv_path.endswith('\\'):
        csv_path = csv_path[:-1]
    if not os.path.isfile(csv_path):
        print(f'WARNING! CSV file not found!')
        print(f'Please check your csv input path again: [{csv_path}]!')
        exit(1)
    img_path: str = args.img.strip()
    if img_path.endswith('/') or img_path.endswith('\\'):
        img_path = img_path[:-1]
    tfrecord_path: str = args.output.strip()
    if tfrecord_path.endswith('/') or tfrecord_path.endswith('\\'):
        tfrecord_path = tfrecord_path[:-1]
    del args

    print(f'Processing {csv_path}!')

    exit(0)

    # flags = tf.app.flags
    # flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
    # flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
    # FLAGS = flags.FLAGS

    # parse text to id
    class_list = []

    def id_of_class(row_label):
        if not class_list:
            _classes = set()
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    _classes.add(row['class'])
            class_list.extend(_classes)
            del _classes
        return 1 + class_list.index(row_label)

    def create_tf_example(group, path):
        print(os.path.join(path, '{}'.format(group.filename)))
        with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        width, height = image.size

        filename = (group.filename + '.jpg').encode('utf8')
        image_format = b'jpg'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        for index, row in group.object.iterrows():
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            classes_text.append(row['class'].encode('utf8'))
            classes.append(id_of_class(row['class']))

        return tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image/height'            : dataset_util.int64_feature(height),
                    'image/width'             : dataset_util.int64_feature(width),
                    'image/filename'          : dataset_util.bytes_feature(filename),
                    'image/source_id'         : dataset_util.bytes_feature(filename),
                    'image/encoded'           : dataset_util.bytes_feature(encoded_jpg),
                    'image/format'            : dataset_util.bytes_feature(image_format),
                    'image/object/bbox/xmin'  : dataset_util.float_list_feature(xmins),
                    'image/object/bbox/xmax'  : dataset_util.float_list_feature(xmaxs),
                    'image/object/bbox/ymin'  : dataset_util.float_list_feature(ymins),
                    'image/object/bbox/ymax'  : dataset_util.float_list_feature(ymaxs),
                    'image/object/class/text' : dataset_util.bytes_list_feature(classes_text),
                    'image/object/class/label': dataset_util.int64_list_feature(classes),
                }))

    async def to_tfrecord(csv_input, tfrecord_file_path):
        with tf.python_io.TFRecordWriter(tfrecord_file_path) as writer:
            examples = pd.read_csv(csv_input)
            data = namedtuple('data', ['filename', 'object'])
            gb = examples.groupby('filename')
            grouped = [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]
            for group in grouped:
                tf_example = create_tf_example(group, img_path)
                writer.write(tf_example.SerializeToString())

        print('CSV -> TFRecord successful!')
        # print('Processed', xml_size, 'boxes in total!')

    train_task = to_tfrecord(csv_path, f'{tfrecord_path}/train.record')
    eval_task = to_tfrecord(csv_path, f'{tfrecord_path}/eval.record')
    await train_task
    await eval_task


asyncio.run(main())